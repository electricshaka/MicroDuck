"""hs12b probe: does the duck re-shape its legs into the L when body[4] says so?

Plays, on the Arena's stage, seeds 1-3: 2.5 s stand, then 2.5 s headstand with body[4]=0, then
3.0 s headstand with body[4]=1 (the freeze called). Per frame it measures the KICK hip error —
min(|left_hip - 1.42|, |right_hip + 1.42|) — and the NEEDLE hip error of the OTHER leg from its
HOME hip (-0.458 left / +0.458 right, the measured vertical), plus trunk-up seconds (z > 0.13)
and yaw drift. The baseline (measured on hs12/model_2250, 2026-10-24): the trained headstand
already rides with BOTH legs level — kick err 0.14/0.04 rad quiet/freeze, needle err ~1.7 (the
needle leg is level too), up 2.2-3.0 s, drift ~3 deg. So the flag's signal is the NEEDLE: the
freeze pulls one leg back to the vertical while the other stays level. Acceptance (a trained
checkpoint): freeze-window kick err <= 0.5 rad AND needle err <= 0.5 rad, held >= 2.0 s of 3,
drift <= 60 deg; the quiet window keeps the baseline shape (needle >= 1.2 rad) — the flag moves
the leg, the show does not.

    uv run --no-sync --group workshop python scratch/kick_probe.py <policy.onnx>
"""
import json
import math
import sys
import tempfile
from pathlib import Path

import mujoco
import numpy as np
from arena.cast import cast_of
from arena.constants import CTRL_DT, JOINT_NAMES
from arena.course import compile_stage
from arena.manifest import load_event
from arena.policy import load_policy
from arena.runner import run_stage
from arena.timeline import solo, load_troupe

EVENT = "got-a-talent"
T_STAND, T_HEAD, T_KICK = 2.5, 5.0, 8.0      # phase ends: stand | quiet headstand | kick called
UP_Z = 0.13                                  # the reward's headstand band's bottom edge
KICK_HIP = {"left": 1.42, "right": -1.42}    # measured level-leg hips (rewards.LKICK_HIP)
NEEDLE_HIP = {"left": -0.458, "right": 0.458}  # the HOME hip — the measured vertical needle


def _timeline(path: Path) -> Path:
    data = {
        "timeline_version": 1,
        "duration_s": 8.0,
        "keyframes": [
            {"t": 0.0, "twist": [0, 0, 0], "head": [0, 0, 0, 0], "body": [0, 0, 0, 0, 0, 0]},
            {"t": T_STAND, "twist": [0, 1, 0], "head": [0, 0, 0, 0], "body": [0, 0, 0, 0, 0, 0]},
            {"t": T_HEAD, "twist": [0, 1, 0], "head": [0, 0, 0, 0], "body": [0, 0, 0, 0, 1, 0]}
        ]
    }
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def _read_frame(model):
    data = mujoco.MjData(model)
    trunk = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")
    free = int(model.jnt_qposadr[model.body_jntadr[trunk]])
    joints = [int(model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)])
              for n in JOINT_NAMES]

    def read(frame):
        data.qpos[free:free + 7] = frame[0:7]
        data.qpos[joints] = frame[7:]
        mujoco.mj_forward(model, data)
        w, x, y, zz = frame[3], frame[4], frame[5], frame[6]     # free quat, w x y z
        yaw = math.atan2(2 * (w * zz + x * y), 1 - 2 * (y * y + zz * zz))
        return float(frame[2]), yaw, float(frame[7 + 2]), float(frame[7 + 11])
    return read


def measure(trace, read) -> dict:
    """Kick/needle hip errors and hold seconds in the two headstand windows: quiet [2.5, 5.0]
    (control, flag down) and called [5.0, 8.0] (the freeze), plus the yaw drift in the freeze."""
    rows = np.array([read(f) for f in trace])          # columns: z, yaw, hip_L, hip_R
    t = (np.arange(len(rows)) + 1) * CTRL_DT
    yaw = np.unwrap(rows[:, 1])
    kick_err = np.minimum(np.abs(rows[:, 2] - KICK_HIP["left"]),
                          np.abs(rows[:, 3] - KICK_HIP["right"]))
    left_is_kick = np.abs(rows[:, 2] - KICK_HIP["left"]) <= np.abs(rows[:, 3] - KICK_HIP["right"])
    needle_err = np.where(left_is_kick,
                          np.abs(rows[:, 3] - NEEDLE_HIP["right"]),
                          np.abs(rows[:, 2] - NEEDLE_HIP["left"]))

    def window(lo, hi):
        idx = np.where((t > lo) & (t <= hi))[0]
        if len(idx) == 0:
            return {"kick_err": float("nan"), "needle_err": float("nan"),
                    "up_s": 0.0, "drift_deg": 0.0}
        return {
            "kick_err": float(kick_err[idx].mean()),
            "needle_err": float(needle_err[idx].mean()),
            "hipL": float(rows[idx, 2].mean()),
            "hipR": float(rows[idx, 3].mean()),
            "up_s": float((rows[idx, 0] > UP_Z).sum() * CTRL_DT),
            "drift_deg": float(math.degrees(yaw[idx].max() - yaw[idx].min())),
        }

    return {"quiet": window(T_STAND, T_HEAD), "kick": window(T_HEAD, T_KICK),
            "played_s": float(t[-1])}


def main(argv=None) -> int:
    argv = list(argv or sys.argv[1:])
    onnx = argv[0]
    event = load_event(EVENT)
    policy = load_policy(Path(onnx))
    out = Path(tempfile.mkdtemp()) / "tl.json"
    timeline = load_troupe(_timeline(out), event, allow_local=True).ducks[0].timeline
    performers = cast_of(event, solo(timeline), policy, None, {}).performers
    read = _read_frame(compile_stage(event, [(p.rig, p.start) for p in performers]))
    means = []
    for seed in (1, 2, 3):
        stats = measure(run_stage(event, performers, seed).trace[0], read)
        means.append(stats)
        q, k = stats["quiet"], stats["kick"]
        print(f"seed {seed}: quiet window kick err {q['kick_err']:.2f} rad (needle {q['needle_err']:.2f}, "
              f"up {q['up_s']:.2f} s) | freeze window kick err {k['kick_err']:.2f} rad "
              f"(needle {k['needle_err']:.2f}), up {k['up_s']:.2f} s, drift {k['drift_deg']:.0f} deg")
        print(f"    raw hips quiet L {q['hipL']:+.2f} R {q['hipR']:+.2f} | "
              f"freeze L {k['hipL']:+.2f} R {k['hipR']:+.2f}")
    qm = {m: float(np.mean([s["quiet"][m] for s in means])) for m in ("kick_err", "needle_err", "up_s")}
    km = {m: float(np.mean([s["kick"][m] for s in means])) for m in ("kick_err", "needle_err", "up_s", "drift_deg")}
    print(f"MEAN: quiet kick err {qm['kick_err']:.2f} rad (needle {qm['needle_err']:.2f}, up "
          f"{qm['up_s']:.2f} s) | freeze kick err {km['kick_err']:.2f} rad (needle "
          f"{km['needle_err']:.2f}), up {km['up_s']:.2f} s, drift {km['drift_deg']:.0f} deg")
    return 0


if __name__ == "__main__":
    sys.exit(main())

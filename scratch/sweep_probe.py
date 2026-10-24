"""hs12 probe: does the duck actually SWEEP when body[1] says so?

Plays, on the Arena's stage, seeds 1-3: 2.5 s stand, then 2.5 s sit with body[1]=0 (the quiet
knee-drop crouch), then 3.0 s sit with body[1]=1 (the sweep called). During the sweep phase it
counts the trunk's counter-clockwise FRONTIER arc past the phase start — the same rule
rewards.downrock_sweep pays on (back-and-forth pays nothing) — plus seconds held in the crouch
and the deepest trunk height. The crouch without the flag must barely turn (that's the quiet
hold); with the flag it must turn: the acceptance bar is a MEAN of >= 30 degrees swept.

    uv run --no-sync --group workshop python scratch/sweep_probe.py <policy.onnx>
"""
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
T_STAND, T_SIT, T_SWEEP = 2.5, 5.0, 8.0     # phase ends: stand | quiet crouch | sweep called
CROUCH_Z_HI = 0.095                          # the reward's seated band's top edge


def _timeline(path: Path) -> Path:
    import json
    data = {
        "timeline_version": 1,
        "duration_s": 8.0,
        "keyframes": [
            {"t": 0.0, "twist": [0, 0, 0], "head": [0, 0, 0, 0], "body": [0, 0, 0, 0, 0, 0]},
            {"t": 2.5, "twist": [1, 0, 0], "head": [0, 0, 0, 0], "body": [0, 0, 0, 0, 0, 0]},
            {"t": T_SIT, "twist": [1, 0, 0], "head": [0, 0, 0, 0], "body": [0, 1, 0, 0, 0, 0]}
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
        return float(frame[2]), yaw
    return read


def measure(trace, read) -> dict:
    rows = [read(f) for f in trace]
    t = (np.arange(len(rows)) + 1) * CTRL_DT
    z, yaw = (np.array(c) for c in zip(*rows))
    unwrapped = np.unwrap(yaw)

    def frontier(t_lo, t_hi):
        """CCW frontier degrees gained within the phase — mirrors rewards.downrock_sweep:
        unwrapped yaw past the running max, so turning back earns nothing."""
        idx = np.where((t > t_lo) & (t <= t_hi))[0]
        if len(idx) == 0:
            return 0.0, 0.0
        base, best, turned = unwrapped[idx[0]], unwrapped[idx[0]], unwrapped[idx[0]]
        best = turned
        for a in unwrapped[idx]:
            turned = a
            best = max(best, turned)
        return math.degrees(best - base), float(t[idx[-1]] - t[idx[0]])

    swept_deg, swept_s = frontier(T_SIT, T_SWEEP)
    quiet_deg, _ = frontier(T_STAND, T_SIT)
    crouch_idx = np.where((t > T_SIT) & (t <= T_SWEEP))[0]
    crouch_s = float((z[crouch_idx] < CROUCH_Z_HI).sum() * CTRL_DT)
    return {"swept_deg": swept_deg, "quiet_deg": quiet_deg, "crouch_s": crouch_s,
            "min_z": float(z[crouch_idx].min()) if len(crouch_idx) else float("nan"),
            "played_s": float(t[-1])}


def main(argv=None) -> int:
    onnx = (argv or sys.argv[1:])[0]
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
        print(f"seed {seed}: swept {stats['swept_deg']:6.1f} deg in the sweep phase "
              f"(quiet crouch {stats['quiet_deg']:5.1f} deg), crouch held {stats['crouch_s']:.2f} s, "
              f"min z {stats['min_z']:.3f} m, played {stats['played_s']:.2f} s")
    print(f"MEAN: swept {np.mean([m['swept_deg'] for m in means]):.1f} deg | "
          f"quiet {np.mean([m['quiet_deg'] for m in means]):.1f} deg | "
          f"crouch {np.mean([m['crouch_s'] for m in means]):.2f} s | "
          f"min z {np.nanmean([m['min_z'] for m in means]):.3f} m")
    return 0


if __name__ == "__main__":
    sys.exit(main())

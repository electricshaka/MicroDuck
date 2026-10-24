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
# full-choreography mode (argv "full"): the candidate show timeline (sweep flag inside the real
# 2.0-4.0 s knee drop, nods before, dive after) — the sweep window is measured where the SHOW
# calls it, plus the standing opening must stay quiet and the dive must still reach the head.
T_FULL_SWEEP_LO, T_FULL_SWEEP_HI, T_FULL_DIVE_HI = 2.0, 4.0, 8.0


def _timeline(path: Path, full: bool = False) -> Path:
    if full:
        import json
        src = Path(__file__).with_name("timeline_sweep.json")
        path.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
        return path
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


def measure(trace, read, t_lo=None, t_hi=None, t_quiet_lo=None, t_quiet_hi=None,
            crouch_lo=None, crouch_hi=None) -> dict:
    """Frontier degrees in the sweep window [t_lo, t_hi], the quiet turn in [quiet], and crouch
    seconds over [crouch_lo, crouch_hi]. Defaults are the acceptance timeline's phases."""
    t_lo = T_SIT if t_lo is None else t_lo
    t_hi = T_SWEEP if t_hi is None else t_hi
    t_quiet_lo = T_STAND if t_quiet_lo is None else t_quiet_lo
    t_quiet_hi = T_SIT if t_quiet_hi is None else t_quiet_hi
    crouch_lo = T_SIT if crouch_lo is None else crouch_lo
    crouch_hi = T_SWEEP if crouch_hi is None else crouch_hi
    rows = [read(f) for f in trace]
    t = (np.arange(len(rows)) + 1) * CTRL_DT
    z, yaw = (np.array(c) for c in zip(*rows))
    unwrapped = np.unwrap(yaw)

    def frontier(lo, hi):
        """CCW frontier degrees gained within the phase — mirrors rewards.downrock_sweep:
        unwrapped yaw past the running max, so turning back earns nothing."""
        idx = np.where((t > lo) & (t <= hi))[0]
        if len(idx) == 0:
            return 0.0, 0.0
        base, best, turned = unwrapped[idx[0]], unwrapped[idx[0]], unwrapped[idx[0]]
        best = turned
        for a in unwrapped[idx]:
            turned = a
            best = max(best, turned)
        return math.degrees(best - base), float(t[idx[-1]] - t[idx[0]])

    swept_deg, swept_s = frontier(t_lo, t_hi)
    quiet_deg, _ = frontier(t_quiet_lo, t_quiet_hi)
    crouch_idx = np.where((t > crouch_lo) & (t <= crouch_hi))[0]
    crouch_s = float((z[crouch_idx] < CROUCH_Z_HI).sum() * CTRL_DT)
    return {"swept_deg": swept_deg, "quiet_deg": quiet_deg, "crouch_s": crouch_s,
            "min_z": float(z[crouch_idx].min()) if len(crouch_idx) else float("nan"),
            "played_s": float(t[-1])}


def main(argv=None) -> int:
    argv = list(argv or sys.argv[1:])
    full = bool(argv and argv[-1] == "full")
    if full:
        argv = argv[:-1]
    onnx = argv[0]
    event = load_event(EVENT)
    policy = load_policy(Path(onnx))
    out = Path(tempfile.mkdtemp()) / "tl.json"
    timeline = load_troupe(_timeline(out, full), event, allow_local=True).ducks[0].timeline
    performers = cast_of(event, solo(timeline), policy, None, {}).performers
    read = _read_frame(compile_stage(event, [(p.rig, p.start) for p in performers]))
    means = []
    for seed in (1, 2, 3):
        result = run_stage(event, performers, seed)
        if full:
            stats = measure(result.trace[0], read,
                            t_lo=T_FULL_SWEEP_LO, t_hi=T_FULL_SWEEP_HI,
                            t_quiet_lo=0.0, t_quiet_hi=T_FULL_SWEEP_LO,
                            crouch_lo=T_FULL_SWEEP_LO, crouch_hi=T_FULL_SWEEP_HI)
            # the dive: trunk height inside the window right after the sweep is released at 4.0 s
            rows = [read(f) for f in result.trace[0]]
            zs = np.array([r[0] for r in rows])
            tt = (np.arange(len(zs)) + 1) * CTRL_DT
            dive = zs[(tt > T_FULL_SWEEP_HI) & (tt <= T_FULL_DIVE_HI)]
            stats["dive_min_z"] = float(dive.min()) if len(dive) else float("nan")
            stats["dive_max_z"] = float(dive.max()) if len(dive) else float("nan")
        else:
            stats = measure(result.trace[0], read)
        means.append(stats)
        extra = (f", dive z {stats['dive_min_z']:.3f}..{stats['dive_max_z']:.3f} m" if full else "")
        print(f"seed {seed}: swept {stats['swept_deg']:6.1f} deg in the sweep phase "
              f"(quiet {stats['quiet_deg']:5.1f} deg), crouch held {stats['crouch_s']:.2f} s, "
              f"min z {stats['min_z']:.3f} m, played {stats['played_s']:.2f} s{extra}")
    dive_note = (f" | dive z {np.nanmean([m['dive_min_z'] for m in means]):.3f}"
                 f"..{np.nanmean([m['dive_max_z'] for m in means]):.3f} m" if full else "")
    print(f"MEAN: swept {np.mean([m['swept_deg'] for m in means]):.1f} deg | "
          f"quiet {np.mean([m['quiet_deg'] for m in means]):.1f} deg | "
          f"crouch {np.mean([m['crouch_s'] for m in means]):.2f} s | "
          f"min z {np.nanmean([m['min_z'] for m in means]):.3f} m{dive_note}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

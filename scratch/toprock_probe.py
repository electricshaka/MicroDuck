"""hs11 probe: does the duck actually STEP when body[0] says so?

Plays a 2.5 s stand on the Arena's stage with body[0] held at 0 then at 1, seeds 1-3, and counts
foot-contact ALTERNATIONS (a new contact by the foot that was not last down — the same rule
rewards.toprock_step pays on), plus hip_roll range and upright seconds. Flag down must barely
step (the show's opening keeps stillness); flag up must step.

    uv run --no-sync --group workshop python scratch/toprock_probe.py <policy.onnx>
"""
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

EVENT, DUR = "got-a-talent", 2.5
UPRIGHT_UPZ, STAND_Z = 0.8, 0.10
HIP_ROLL = (1, 10)  # joint indices rewards.toprock_sway uses


def _timeline(flag: float, path: Path) -> Path:
    path.write_text(f'{{"timeline_version": 1, "duration_s": {DUR}, "keyframes": ['
                    f'{{"t": 0.0, "twist": [0, 0, 0], "head": [0, 0, 0, 0], '
                    f'"body": [{flag}, 0, 0, 0, 0, 0]}}]}}')
    return path


def _read_frame(model):
    data = mujoco.MjData(model)
    floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    feet = {side: [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"{side}_foot_collision")]
            for side in ("left", "right")}
    trunk = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")
    free = int(model.jnt_qposadr[model.body_jntadr[trunk]])
    joints = [int(model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)])
              for n in JOINT_NAMES]
    hips = [joints[i] for i in HIP_ROLL]

    def read(frame):
        data.qpos[free:free + 7] = frame[0:7]
        data.qpos[joints] = frame[7:]
        mujoco.mj_forward(model, data)
        rot = data.xmat[trunk].reshape(3, 3)
        touched = {int(c.geom1 if c.geom2 == floor else c.geom2)
                   for c in data.contact[:data.ncon] if floor in (c.geom1, c.geom2)}
        return (float(rot[2, 2]), float(frame[2]),
                any(g in touched for g in feet["left"]), any(g in touched for g in feet["right"]),
                float(data.qpos[hips[0]] - data.qpos[hips[1]]))
    return read


def measure(trace, read) -> dict:
    rows = [read(f) for f in trace]
    t = (np.arange(len(rows)) + 1) * CTRL_DT
    upz, z, left, right, hip = (np.array(c) for c in zip(*rows))
    # mirrors rewards.toprock_step: pay a NEW contact (rising edge) by the foot that was not
    # last down, no sooner than every 0.5 s (the tempo cap the reward enforces).
    steps, last, since, prev = 0, -1, 10.0, (False, False)
    for i in np.where(t <= DUR)[0]:
        hits = (bool(left[i]), bool(right[i]))
        for foot in (0, 1):
            if hits[foot] and not prev[foot]:          # this foot just touched down
                if foot != last and since >= 0.5:
                    steps += 1
                    since = 0.0
                last = foot
        prev = hits
        since += CTRL_DT
    upright = float(((upz > UPRIGHT_UPZ) & (z > STAND_Z) & (t <= DUR)).sum() * CTRL_DT)
    return {"steps": steps, "upright_s": upright, "played_s": float(t[-1]),
            "hip_roll_range": float(np.ptp(hip[t <= DUR]))}


def main(argv=None) -> int:
    onnx = (argv or sys.argv[1:])[0]
    event = load_event(EVENT)
    policy = load_policy(Path(onnx))
    out = Path(tempfile.mkdtemp()) / "tl.json"
    for flag in (0.0, 1.0):
        timeline = load_troupe(_timeline(flag, out), event, allow_local=True).ducks[0].timeline
        performers = cast_of(event, solo(timeline), policy, None, {}).performers
        read = _read_frame(compile_stage(event, [(p.rig, p.start) for p in performers]))
        means = []
        for seed in (1, 2, 3):
            stats = measure(run_stage(event, performers, seed).trace[0], read)
            means.append(stats)
            print(f"body[0]={flag:.0f} seed {seed}: steps {stats['steps']}, "
                  f"hip_roll range {stats['hip_roll_range']:.3f} rad, "
                  f"upright {stats['upright_s']:.2f} of {stats['played_s']:.2f} s")
        print(f"body[0]={flag:.0f} MEAN: steps {np.mean([m['steps'] for m in means]):.1f}, "
              f"hip_roll {np.mean([m['hip_roll_range'] for m in means]):.3f} rad, "
              f"upright {np.mean([m['upright_s'] for m in means]):.2f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())

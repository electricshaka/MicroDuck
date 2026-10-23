"""How well does a MicroDuck Got a Talent policy go onto its head, hold there, and spin?

    uv run --frozen --group workshop python -m microduck_challenges.got_a_talent.headspin_check policy.onnx [--timeline FILE] [--seeds 1 2 3]

Plays a timeline on the Arena's stage with the Arena's own runner (arena.runner.run_stage, the
star alone), by default the built-in test below: stand 2 s, the headstand flag (twist[1]) from
2 s, the spin flag (twist[2]) from 5 s to the end at 15 s. Then reads each recorded frame back
into the stage's model to find what touches the floor, and reports per seed:

- whether the duck stood upright (trunk within ~37° of vertical) before the headstand flag;
- how long after the flag it was first ON ITS HEAD: trunk upside down within ~32° of vertical,
  trunk above 0.10 m, the head the only part on the floor (lying upside down on the trunk, or a
  head-and-feet tripod, does not count);
- the longest unbroken stretch on the head, and the total time on it;
- while the spin flag is up: counter-clockwise turns made on the head, the mean spin rate over
  the whole spin phase (time off the head counts as zero) and the best 1 s.

The Arena plays a performance once, on seed 1; the other seeds change the start pose's noise
and the servo delays, a check of how reliably the trick repeats. Exits 1 on a bad file.
"""

import argparse
import json
import sys
import tempfile
from pathlib import Path

import mujoco
import numpy as np
from arena.cast import cast_of
from arena.constants import CTRL_DT, JOINT_NAMES
from arena.course import compile_stage
from arena.manifest import load_event
from arena.policy import PolicyError, load_policy
from arena.runner import run_stage
from arena.timeline import TimelineError, load_troupe, solo

EVENT = "got-a-talent"
HEADSTAND_AT, SPIN_AT, DURATION = 2.0, 5.0, 15.0
UPRIGHT_UPZ = 0.8      # trunk's z axis, world z: standing above this
INVERTED_UPZ = -0.85   # on the head below this
HEAD_HIGH_Z = 0.10     # trunk height above which upside down is a headstand, not lying
HEAD_BODY = "jaw_soft"

TEST = {"timeline_version": 1, "duration_s": DURATION, "keyframes": [
    {"t": 0.0, "twist": [0, 0, 0], "head": [0, 0, 0, 0]},
    {"t": HEADSTAND_AT, "twist": [0, 1, 0], "head": [0, 0, 0, 0]},
    {"t": SPIN_AT, "twist": [0, 1, 1], "head": [0, 0, 0, 0]},
]}


def _frame_reader(model):
    """A function that puts one trace frame into a fresh MjData and returns, for that frame:
    trunk up-axis world z, trunk height, trunk heading (rad), head on the floor, anything else on it."""
    data = mujoco.MjData(model)
    trunk = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")
    head = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, HEAD_BODY)
    floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    free = int(model.jnt_qposadr[model.body_jntadr[trunk]])
    joints = [int(model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)]) for n in JOINT_NAMES]

    def read(frame):
        data.qpos[free:free + 7] = frame[0:7]
        data.qpos[joints] = frame[7:]
        mujoco.mj_forward(model, data)
        rot = data.xmat[trunk].reshape(3, 3)
        on_floor = {int(model.geom_bodyid[c.geom2 if c.geom1 == floor else c.geom1])
                    for c in data.contact[:data.ncon] if floor in (c.geom1, c.geom2)}
        return (float(rot[2, 2]), float(frame[2]), float(np.arctan2(rot[1, 0], rot[0, 0])),
                head in on_floor, bool(on_floor - {head}))
    return read


def judge(trace, read, headstand_at: float, spin_at: float | None) -> dict:
    rows = [read(f) for f in trace]
    t = (np.arange(len(rows)) + 1) * CTRL_DT
    upz, z, heading, head, other = (np.array(c) for c in zip(*rows))
    on = (t > headstand_at) & (upz < INVERTED_UPZ) & (z > HEAD_HIGH_Z) & head & ~other
    stood = bool((upz[t <= headstand_at] > UPRIGHT_UPZ).all())
    first = float(t[on.argmax()] - headstand_at) if on.any() else None
    longest = run = 0
    for o in on:
        run = run + 1 if o else 0
        longest = max(longest, run)
    out = {"stood": stood, "first": first, "longest": longest * CTRL_DT, "total": float(on.sum() * CTRL_DT)}
    if spin_at is not None:
        step = (np.diff(heading, prepend=heading[0]) + np.pi) % (2 * np.pi) - np.pi
        both = on & np.roll(on, 1)  # a turn counts between two frames both on the head
        rate = np.where(both, step / CTRL_DT, 0.0)[t > spin_at]
        window = round(1.0 / CTRL_DT)
        best = max((rate[i:i + window].mean() for i in range(max(1, len(rate) - window + 1))), default=0.0)
        out.update(turns=float(rate.sum() * CTRL_DT / (2 * np.pi)), mean_rate=float(rate.mean()) if len(rate) else 0.0,
                   best_rate=float(best))
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Judge a MicroDuck Got a Talent headstand and headspin on the Arena's stage.")
    parser.add_argument("onnx")
    parser.add_argument("--timeline", help="a timeline.json to play instead of the built-in test "
                        "(the headstand flag's first rise is the reference; spin is measured from the spin flag's)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    args = parser.parse_args(argv)
    event = load_event(EVENT)
    try:
        if args.timeline:
            path = Path(args.timeline)
        else:
            path = Path(tempfile.mkdtemp()) / "headspin_test.json"
            path.write_text(json.dumps(TEST))
        timeline = load_troupe(path, event, allow_local=True).ducks[0].timeline
        policy = load_policy(Path(args.onnx))
    except (TimelineError, PolicyError) as exc:
        print(f"{exc}", file=sys.stderr)
        return 1
    rises = lambda slot: [t for t, c in zip(timeline.times, timeline.commands) if c[slot] >= 0.5]
    headstand_at = (rises(1) or [timeline.duration_s])[0]
    spin_at = (rises(2) or [None])[0]
    performers = cast_of(event, solo(timeline), policy, None, {}).performers
    read = _frame_reader(compile_stage(event, [(p.rig, p.start) for p in performers]))
    results = []
    for seed in args.seeds:
        outcome = run_stage(event, performers, seed)
        r = judge(outcome.trace[0], read, headstand_at, spin_at)
        results.append(r)
        line = (f"seed {seed}: stood before the flag {'yes' if r['stood'] else 'NO'} | onto head after "
                f"{'never' if r['first'] is None else '%.2f s' % r['first']} | longest hold {r['longest']:.2f} s | "
                f"on head {r['total']:.2f} of {timeline.duration_s - headstand_at:.0f} s")
        if spin_at is not None:
            line += f" | spin: {r['turns']:+.2f} turns (ccw +), mean {r['mean_rate']:+.1f} rad/s, best 1 s {r['best_rate']:+.1f} rad/s"
        print(line)
    keys = ["longest", "total"] + (["turns", "mean_rate", "best_rate"] if spin_at is not None else [])
    print("mean " + ", ".join(f"{k} {np.mean([r[k] for r in results]):+.2f}" for k in keys))
    return 0


if __name__ == "__main__":
    sys.exit(main())

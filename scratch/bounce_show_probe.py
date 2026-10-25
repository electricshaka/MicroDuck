"""hs13 probe: does the duck BOUNCE to the beat when body[0]=1, and STEP when body[2]=1 too?

This is the overhaul's acceptance gate, the toprock_probe's role reborn on a moving body. It
plays, on the Arena's stage, seeds 1-3, ALL STANDING (the sit flag never goes up):
2.5 s with body all zero (the quiet stand — control), then 3.0 s with body[0]=1 (the bounce
called), then 3.0 s with body[0]=1 AND body[2]=1 (the cross-steps called on the bounce).

It counts bounce CYCLES the way rewards.bounce_hold pays them: a trough (falling to rising) that
dropped >= BOUNCE_DROP below the cycle's own reached top, with that top >= BOUNCE_Z_TOP and the
knees flexed (mean |knee - home| >= BOUNCE_KNEE at the trough) — "knee cycles" — and reports the
z-only count next to it (a bounce without knee flex is a sag, not the move). Cross-steps count
the way rewards.bounce_step does: hip_pitch peaks past +/- STEP_HIP_FWD from home (left forward
is NEGATIVE, right POSITIVE), ALTERNATION only, >= STEP_PERIOD_S apart.

The bar (see AGENTS.md hs13): >= 3 knee cycles >= 1.2 cm deep in the flagged window (control
<= 0.5), and >= 2 alternating step peaks with body[2]=1. The FLAT baseline is known: the show
timeline route measured 0 cycles / 0.003 m (scratch/bounce_probe.py) — this must read very
different, on the TRAINED regime.

    uv run --no-sync --group workshop python scratch/bounce_show_probe.py <policy.onnx>
"""
import json
import math
import re
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
from mjlab_microduck.robot.microduck_constants import HOME_FRAME

EVENT = "got-a-talent"
T_QUIET, T_BOUNCE, T_STEP = 2.5, 5.5, 8.5      # phase ends: quiet | bounce | bounce + steps

# the reward's own rules (rewards.py, hs13), for a like-for-like read
BOUNCE_DROP, BOUNCE_Z_TOP, BOUNCE_KNEE = 0.012, 0.108, 0.20
STEP_HIP_FWD, STEP_PERIOD_S = 0.45, 0.25


def _timeline(path: Path) -> Path:
    data = {
        "timeline_version": 1,
        "duration_s": 8.5,
        "keyframes": [
            {"t": 0.0, "twist": [0, 0, 0], "head": [0, 0, 0, 0], "body": [0, 0, 0, 0, 0, 0]},
            {"t": T_QUIET, "twist": [0, 0, 0], "head": [0, 0, 0, 0], "body": [1, 0, 0, 0, 0, 0]},
            {"t": T_BOUNCE, "twist": [0, 0, 0], "head": [0, 0, 0, 0], "body": [1, 0, 1, 0, 0, 0]}
        ]
    }
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def _home(name: str) -> float:
    for pat, val in HOME_FRAME.joint_pos.items():
        if re.fullmatch(pat, name):
            return float(val)
    return 0.0


_J = {n: next(i for i, x in enumerate(JOINT_NAMES) if n in x) for n in
      ("left_hip_pitch", "right_hip_pitch", "left_knee", "right_knee")}
HOME = {n: _home(next(x for x in JOINT_NAMES if n in x)) for n in _J}


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
        knee = 0.5 * (abs(frame[7 + _J["left_knee"]] - HOME["left_knee"])
                      + abs(frame[7 + _J["right_knee"]] - HOME["right_knee"]))
        dev = (frame[7 + _J["left_hip_pitch"]] - HOME["left_hip_pitch"],
               frame[7 + _J["right_hip_pitch"]] - HOME["right_hip_pitch"])
        return float(frame[2]), knee, dev[0], dev[1]
    return read


def measure(trace, read) -> dict:
    """Cycle and step counts per window, on the trace alone (no policy access)."""
    rows = [read(f) for f in trace]
    t = (np.arange(len(rows)) + 1) * CTRL_DT
    z, knee, dev_l, dev_r = (np.array(c, dtype=float) for c in zip(*rows))
    out = {}

    def cycles(lo, hi):
        """Troughs that dropped BOUNCE_DROP from their own reached top (>= BOUNCE_Z_TOP);
        knee cycles want mean |knee - home| >= BOUNCE_KNEE at the trough too."""
        idx = np.where((t > lo) & (t <= hi))[0]
        top, fell, n, nk, drops = z[idx[0]], False, 0, 0, []
        prev = z[idx[0]]
        for i in idx:
            zi = z[i]
            if zi > top:
                top = zi
            if zi < prev:
                fell = True
            elif fell and zi >= prev and top - zi >= BOUNCE_DROP:      # the trough edge
                if top >= BOUNCE_Z_TOP:
                    n += 1
                    drops.append(top - zi)
                    if knee[i] >= BOUNCE_KNEE:
                        nk += 1
                top, fell = zi, False
            elif zi >= prev:
                fell = False
            prev = zi
        return n, nk, (float(np.mean(drops)) if drops else 0.0)

    def steps(lo, hi):
        """Alternating hip_pitch peaks past STEP_HIP_FWD, >= STEP_PERIOD_S apart — bounce_step's
        rule on the trace (left forward is NEGATIVE, right POSITIVE, the downrock signs)."""
        idx = np.where((t > lo) & (t <= hi))[0]
        side, last_t, n = -1, -10.0, 0
        for i in idx:
            if dev_l[i] <= -STEP_HIP_FWD and side != 0 and t[i] - last_t >= STEP_PERIOD_S:
                side, last_t, n = 0, t[i], n + 1
            elif dev_r[i] >= STEP_HIP_FWD and side != 1 and t[i] - last_t >= STEP_PERIOD_S:
                side, last_t, n = 1, t[i], n + 1
        return n

    qc, qkc, _ = cycles(0.0, T_QUIET)
    bc, bkc, bdrop = cycles(T_QUIET, T_BOUNCE)
    sc, skc, sdrop = cycles(T_BOUNCE, T_STEP)
    n_s = steps(T_BOUNCE, T_STEP)
    sw = np.where((t > T_BOUNCE) & (t <= T_STEP))[0]
    out = {"quiet_c": qc, "bounce_c": bc, "bounce_kc": bkc, "bounce_drop": bdrop,
           "step_c": sc, "step_kc": skc, "step_drop": sdrop,
           "steps": n_s,
           "dev_l_min": float(dev_l[sw].min()), "dev_r_max": float(dev_r[sw].max()),
           "z_bounce": (float(z[np.where((t > T_QUIET) & (t <= T_BOUNCE))[0]].min()),
                        float(z[np.where((t > T_QUIET) & (t <= T_BOUNCE))[0]].max())),
           "played_s": float(t[-1])}
    return out


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
        result = run_stage(event, performers, seed)
        s = measure(result.trace[0], read)
        means.append(s)
        print(f"seed {seed}: quiet {s['quiet_c']} cycles | BOUNCE window {s['bounce_c']} cycles "
              f"({s['bounce_kc']} knee) mean drop {s['bounce_drop'] * 100:.1f} cm, "
              f"z {s['z_bounce'][0]:.3f}..{s['z_bounce'][1]:.3f} | STEP window {s['step_c']} cycles "
              f"({s['step_kc']} knee, drop {s['step_drop'] * 100:.1f} cm), "
              f"{s['steps']} alternating steps (hip dev L {s['dev_l_min']:+.2f} / R "
              f"{s['dev_r_max']:+.2f}), played {s['played_s']:.2f} s")
    m = lambda k: float(np.mean([s[k] for s in means]))
    print(f"MEAN: quiet {m('quiet_c'):.1f} | bounce {m('bounce_c'):.1f} cycles "
          f"({m('bounce_kc'):.1f} knee, drop {m('bounce_drop') * 100:.1f} cm) | "
          f"steps window {m('step_c'):.1f} cycles ({m('step_kc'):.1f} knee), "
          f"{m('steps'):.1f} alternating steps | "
          f"BAR: bounce >= 3 knee cycles >= 1.2 cm with quiet <= 0.5, steps >= 2 alternations")
    ok = (m("bounce_kc") >= 3.0 and m("bounce_drop") >= BOUNCE_DROP and m("quiet_c") <= 0.5
          and m("steps") >= 2.0)
    print("hs13 bounce acceptance:", "PASS" if ok else "FAIL")
    return 0


if __name__ == "__main__":
    sys.exit(main())

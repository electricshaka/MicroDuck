"""Phase-1 probe: does a sit flag pulsed at 108 BPM make the duck BOUNCE?

The user's toprock critique asks for an active rhythmic knee bounce — down and up on every beat.
The only standing→crouch mechanism the trained policy has is the sit flag (twist[0]), and its
training reference ramps over 2 s. This probe plays the flag ALTERNATING every half-beat
(108 BPM = 0.5556 s per beat, so a flip every 0.2778 s — the cheapest bounce rhythm: one dip and
one rise per beat) after a 1.0 s settle, for 8.0 s, seeds 1-3, and measures the trunk z per beat:
BEAT DROP = mean z over the beat's first 0.1 s minus the beat's min z, and BEAT RISE = the beat's
max z after that dip minus the dip. The bar (honest, written before running): MEDIAN beat drop
>= 0.02 m in at least 60% of the beats and a similar rise — the duck visibly springs. If the
2 s ramp cannot oscillate at 1.8 Hz, this says so with numbers and the bounce dies with them.

    uv run --no-sync --group workshop python scratch/bounce_probe.py <policy.onnx>
"""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
from arena.cast import cast_of
from arena.constants import CTRL_DT
from arena.course import compile_stage
from arena.manifest import load_event
from arena.policy import load_policy
from arena.runner import run_stage
from arena.timeline import solo, load_troupe

EVENT = "got-a-talent"
T_SETTLE = 1.0                      # standing settle before the pulse starts
T_END = 9.0                         # pulse runs [1.0, 9.0] = 14.4 beats
BEAT_S = 60.0 / 108.0               # 0.5556 s per beat at 108 BPM
BAR_DROP = 0.02                     # m: the written bar for a visible knee spring


def _timeline(path: Path) -> Path:
    kfs = [{"t": 0.0, "twist": [0, 0, 0], "head": [0, 0, 0, 0], "body": [0, 0, 0, 0, 0, 0]}]
    t, sit = T_SETTLE, 1
    while t < T_END - 1e-6:
        kfs.append({"t": round(t, 4), "twist": [sit, 0, 0], "head": [0, 0, 0, 0],
                    "body": [0, 0, 0, 0, 0, 0]})
        sit = 1 - sit
        t += BEAT_S / 2.0           # flip every half-beat: dip on sit, rise on stand
    data = {"timeline_version": 1, "duration_s": T_END, "keyframes": kfs}
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def measure(trace) -> dict:
    """Per-beat drop and rise of trunk z inside the pulse window."""
    z = np.array([f[2] for f in trace])
    t = (np.arange(len(z)) + 1) * CTRL_DT
    inside = (t >= T_SETTLE) & (t <= T_END)
    drops, rises = [], []
    n_beats = int((T_END - T_SETTLE) // BEAT_S)
    for k in range(n_beats):
        lo, hi = T_SETTLE + k * BEAT_S, T_SETTLE + (k + 1) * BEAT_S
        head = z[(t >= lo) & (t < lo + 0.1)]           # start-of-beat level
        body = z[(t >= lo) & (t < hi)]
        if len(head) == 0 or len(body) == 0:
            continue
        dip_i = int(np.argmin(body))
        drop = float(head.mean() - body.min())
        after = body[dip_i:]
        rise = float(after.max() - body.min()) if len(after) else 0.0
        drops.append(drop)
        rises.append(rise)
    return {"beats": len(drops), "drops": drops, "rises": rises,
            "z_first": float(z[inside].mean()),
            "played_s": float(t[-1])}


def main(argv=None) -> int:
    argv = list(argv or sys.argv[1:])
    onnx = argv[0]
    event = load_event(EVENT)
    policy = load_policy(Path(onnx))
    out = Path(tempfile.mkdtemp()) / "tl.json"
    timeline = load_troupe(_timeline(out), event, allow_local=True).ducks[0].timeline
    performers = cast_of(event, solo(timeline), policy, None, {}).performers
    compile_stage(event, [(p.rig, p.start) for p in performers])   # validate the stage once
    all_ok = []
    for seed in (1, 2, 3):
        stats = measure(run_stage(event, performers, seed).trace[0])
        drops, rises = np.array(stats["drops"]), np.array(stats["rises"])
        ok = int((drops >= BAR_DROP).sum())
        all_ok.append((ok, stats["beats"]))
        print(f"seed {seed}: {stats['beats']} beats | beat drop med {np.median(drops):.3f} m "
              f"(min {drops.min():.3f} max {drops.max():.3f}) | rise med {np.median(rises):.3f} m "
              f"| beats >= {BAR_DROP * 100:.0f} cm: {ok}/{stats['beats']} | mean z {stats['z_first']:.3f} m")
    ok = sum(o for o, _ in all_ok)
    tot = sum(b for _, b in all_ok)
    verdict = "BOUNCE" if ok >= 0.6 * tot else "FLAT"
    print(f"VERDICT {verdict}: {ok}/{tot} beats drop >= {BAR_DROP * 100:.0f} cm "
          f"(bar: median >= 2 cm in >= 60% of beats)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""hs12 warm-start staging: verify the body-flag dims are ALIVE in the hs11b winner, then copy it
into the load-run folder. No surgery — and that is the point.

prewarm_hs11 woke the WHOLE body block (actor dims 55-60, critic 68-73) in the hs10 base, and
hs11b trained on that copy: five iterations later body[1..5] carry normalizer mean ~0.21 / std ~0.41
(the seeded 0.25/0.1875 pulled toward the zeros the flag stayed at all run) and first-layer columns
of |w| <= ~0.07 — small Adam drift, not init noise, not the std-0 landmine. A raised sweep flag
normalizes to ~(1-0.21)/0.41 = +1.9: a clean signal. Zeroing/patching here would shift every hidden
unit's pre-activation and make the resume NOT start at hs11b's measured behaviour — so we verify
the healthy state instead of enforcing an artificial one.

    uv run --no-sync python scratch/prewarm_hs12.py [SRC.pt DST.pt]

Defaults: hs11b's winner into logs/rsl_rl/got_a_talent/hs11b_base/model_1250.pt.
"""

import shutil
import sys
from pathlib import Path

import torch

SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("checkpoints/got_a_talent/hs11b/model_1250.pt")
DST = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("logs/rsl_rl/got_a_talent/hs11b_base/model_1250.pt")
# actor obs: twist 48-50, head 51-54, body 55-60 → body[1..5] = 56-60
# critic obs adds 13 privileged dims → body block starts at 68     → body[1..5] = 69-73
BLOCKS = {"actor": slice(56, 61), "critic": slice(69, 74)}


def main() -> None:
    assert SRC.exists(), f"no checkpoint at {SRC}"
    ck = torch.load(SRC, map_location="cpu")
    for grp, sl in BLOCKS.items():
        sd = ck[f"{grp}_state_dict"]
        std = sd["obs_normalizer._std"][0, sl]
        w = sd["mlp.0.weight"][:, sl]
        # std 0 (the dead-dim landmine) would make the ONNX bake eps 1e-2 and explode the flag
        # input to ~100; std >= 0.2 keeps a raised flag near +2. Weight columns must be small or
        # the dims mean something already — trained dims here read ~0.7.
        assert float(std.min()) > 0.2, f"{grp} body[1..5] normalizer std {std.tolist()}: dead dim?"
        assert float(w.abs().max()) < 0.3, f"{grp} body[1..5] weights busy (max {float(w.abs().max()):.2f})"
        print(f"{grp}: dims {sl.start}-{sl.stop - 1} alive — normalizer std "
              f"{float(std.min()):.2f}..{float(std.max()):.2f}, first-layer |w| <= "
              f"{float(w.abs().max()):.3f} (near-neutral start, clean flag signal ~+1.9)")
    DST.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(SRC, DST)
    assert torch.load(DST, map_location="cpu")["iter"] == ck["iter"]
    print(f"staged unmodified for resume: {DST}")


if __name__ == "__main__":
    sys.exit(main())

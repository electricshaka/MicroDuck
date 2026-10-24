"""hs12b warm-start staging: verify the L-kick slot is ALIVE in the hs12 winner, then copy it into
the load-run folder. No surgery — and that is the point.

The kick flag (body[4]) never rode during hs11b or hs12 — the command zeroed slots 2-5 every step
— so its obs dims carry the state hs11's prewarm woke them to and two runs of constant zero on
top. Measured in hs12/model_2250 (2026-10-24): actor dim 59 normalizer std 0.398, first-layer
|w| <= 0.165; critic dim 72 std 0.398, |w| <= 0.260. Past the std-0 landmine (eps floor would
explode a raised flag past +100), and the small weights mean the dim is near-neutral — the resume
starts AT hs12's measured behaviour and the flag enters as a ~+2.5 signal, the same clean ride
the sweep flag had into hs12. Zeroing or re-seeding here would shift every hidden unit's
pre-activation and destroy that, so we verify the healthy state instead of enforcing one.

    uv run --no-sync python scratch/prewarm_hs12b.py [SRC.pt DST.pt]

Defaults: hs12's winner into logs/rsl_rl/got_a_talent/hs12_base/model_2250.pt.
"""

import shutil
import sys
from pathlib import Path

import torch

SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("checkpoints/got_a_talent/hs12/model_2250.pt")
DST = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("logs/rsl_rl/got_a_talent/hs12_base/model_2250.pt")
# actor obs: twist 48-50, head 51-54, body 55-60 → body[4] = 59
# critic obs adds 13 privileged dims → body block starts at 68     → body[4] = 72
BLOCKS = {"actor": slice(59, 60), "critic": slice(72, 73)}


def main() -> None:
    assert SRC.exists(), f"no checkpoint at {SRC}"
    ck = torch.load(SRC, map_location="cpu")
    for grp, sl in BLOCKS.items():
        sd = ck[f"{grp}_state_dict"]
        std = sd["obs_normalizer._std"][0, sl]
        w = sd["mlp.0.weight"][:, sl]
        # std 0 (the dead-dim landmine) would make a raised kick flag normalize past +100;
        # >= 0.2 keeps it near +2.5. Busy weight columns would mean the slot already carries a
        # move (it must not — body[4] is the freeze's new home); trained dims read ~0.7.
        assert float(std.min()) > 0.2, f"{grp} body[4] normalizer std {std.tolist()}: dead dim?"
        assert float(w.abs().max()) < 0.3, f"{grp} body[4] weights busy (max {float(w.abs().max()):.2f})"
        print(f"{grp}: body[4] (dim {sl.start}) alive — normalizer std "
              f"{float(std.min()):.3f}, first-layer |w| <= {float(w.abs().max()):.3f} "
              f"(near-neutral start, clean flag signal ~+2.5)")
    DST.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(SRC, DST)
    assert torch.load(DST, map_location="cpu")["iter"] == ck["iter"]
    print(f"staged unmodified for resume: {DST}")


if __name__ == "__main__":
    sys.exit(main())

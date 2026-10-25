"""hs11 warm-start prewarm: wake the six dead body-command dims in model_1750.

Through hs10 the body command was always zero, so in both obs normalizers the
body block holds mean 0 / var 0 / std 0 (verified: actor dims 55-60, critic
dims 68-73 are the only zero-variance runs). EmpiricalNormalization's forward
is (x - mean) / (std + eps) with eps = 1e-2, so a toprock flag of 1 would
enter the frozen network as ~100 — and the first-layer weights on those
columns never trained. With count ~713M the running stats would take many
million steps to recover on their own.

Patched copy (the original checkpoint stays untouched):
  - normalizer body dims: mean 0.25, var 0.1875 (a Bernoulli(0.5) flag raised
    only while standing: p ~ 0.25), std = sqrt(var). The running update is
    near-frozen at this count, so these stay a sane fixed normalization.
  - actor and critic mlp.0.weight columns of those dims set to 0: the resume
    then starts EXACTLY at hs10 behaviour (the columns were dead), and PPO
    learns the flag's use from zero rather than from random init noise.
"""

import math
import shutil
import sys
from pathlib import Path

import torch

SRC = Path("checkpoints/got_a_talent/hs10/model_1750.pt")
DST = Path("logs/rsl_rl/got_a_talent/hs10_base/model_1750.pt")
BLOCKS = {"actor": slice(55, 61), "critic": slice(68, 74)}
MEAN, VAR = 0.25, 0.1875

def main() -> None:
    assert torch.load(SRC, map_location="cpu"), "source checkpoint unreadable"
    DST.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(SRC, DST.parent / "model_1750_src_copy.pt")  # provenance copy
    ck = torch.load(SRC, map_location="cpu")
    for grp, sl in BLOCKS.items():
        sd = ck[f"{grp}_state_dict"]
        assert float(sd["obs_normalizer._var"][0, sl].abs().max()) == 0.0, f"{grp} body not dead"
        sd["obs_normalizer._mean"][0, sl] = MEAN
        sd["obs_normalizer._var"][0, sl] = VAR
        sd["obs_normalizer._std"][0, sl] = math.sqrt(VAR)
        w = sd["mlp.0.weight"]
        n_dead = float(w[:, sl].abs().sum())
        w[:, sl] = 0.0
        print(f"{grp}: dims {sl.start}-{sl.stop - 1} normalizer set "
              f"(mean {MEAN}, std {math.sqrt(VAR):.3f}); zeroed {w[:, sl].numel()} "
              f"first-layer weights (|w| was {n_dead:.2f}, dead weights from init)")
    torch.save(ck, DST)
    back = torch.load(DST, map_location="cpu")
    for grp, sl in BLOCKS.items():
        sd = back[f"{grp}_state_dict"]
        assert torch.allclose(sd["obs_normalizer._std"][0, sl], torch.tensor(math.sqrt(VAR)))
        assert float(sd["mlp.0.weight"][:, sl].abs().max()) == 0.0
    print(f"prewarmed copy written and verified: {DST}")

if __name__ == "__main__":
    sys.exit(main())

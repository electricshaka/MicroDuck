"""one-off: replace old commit-SHA references with post-rewrite SHAs.

Reads scratch/sha_map.txt (old9 -> new9, full hashes in scratch/old_main_chain.txt),
rewrites every occurrence of each old prefix (7 or 9 chars) in AGENTS.md and the
archived provenance.json files. Word-boundary on hex chars only."""
import re
from pathlib import Path

root = Path(__file__).resolve().parents[1]
pairs = []
for line in (root / "scratch/sha_map.txt").read_text(encoding="utf-8").splitlines():
    if "->" in line:
        old, new = line.split("->")
        pairs.append((old.strip(), new.strip()))

targets = [root / "src/microduck_challenges/got_a_talent/AGENTS.md"]
targets += sorted((root / "checkpoints/got_a_talent").glob("*/provenance.json"))
for path in targets:
    text = path.read_text(encoding="utf-8")
    hits = 0
    for old9, new9 in pairs:
        for n in (9, 7):
            old, new = old9[:n], new9[:n]
            text, k = re.subn(rf"(?<![0-9a-f]){old}(?![0-9a-f])", new, text)
            hits += k
    if hits:
        path.write_text(text, encoding="utf-8")
    print(f"{path.relative_to(root)}: {hits} replacements")

# AGENTS.md: MicroDuck Got a Talent

You are helping a user prepare a Microduck performance for the Arena's **MicroDuck Got a Talent**
(event `got-a-talent`, task `Mjlab-GotATalent-MicroDuck`, this folder). The user may be new to
reinforcement learning. Work so that it stays:

- **simple**: the user decides the performance, the budget and what gets published; you do the
  mechanics;
- **transparent**: say what you change and why before each run or edit, one commit per run, and
  report what the Arena's stage and `check.py` measure, never an impression from the curves;
- **efficient**: a better performance often needs no training at all (edit `timeline.json`); when
  it does, one change per run, a smoke test before any long run, every checkpoint judged.

Never launch a Hugging Face job, publish, enter the Arena or push without the user's explicit yes.

## The challenge

A performance, not a race. The rules, from `arena/events/got-a-talent.toml` in the installed Arena:

- **Stage**: a flat, open floor, no walls. Start from the `STAND` pose, 0.5 s settle (playing the
  first keyframe), joints perturbed by ±0.01 rad, heading by ±2°.
- **Performance**: the Arena plays the entry's `timeline.json` once, for at most **20 s**. Nothing
  is timed and nothing has to be finished. A fall does not stop it: it is part of the show (the fall
  rule, trunk tilted more than 60° for 0.3 s, only labels it). An entry without a timeline stands
  still for 20 s.
- **Judging**: visitors vote. Every performance played to the end goes to the ballots: two to four
  anonymous performances, watched one at a time on stage. The most votes over the **last seven
  days** wins; ties favour the earlier publication. The user's **latest** performance played to the
  end represents them, and votes on an older entry do not move to its replacement.

So what wins is a performance people want to watch: clear moves, timing, a little story, executed
reliably by the policy.

## The policy and the timeline

61 observations → 14 joint position targets at 50 Hz, exported to ONNX (≤ 64 MiB); `publish` checks
the shape. The observations, as the Arena builds them: gyro (3), projected gravity (3), joint
positions from the default pose (14), joint velocities one control step old (14), last action (14),
then the 13 command numbers the timeline gives: twist (3), head (4), body (6). This challenge's
policy reads:

- `twist[0]`, the **sit flag**: 1 sits, 0 stands;
- `twist[1]`, the **headstand flag**: 1 goes onto the head and holds (see "The headspin" below);
- `twist[2]`, the **spin flag**: with the headstand flag, 1 spins counter-clockwise (seen from above)
  on the head;
- `head`, **4 angles in radians from the default pose**: neck_pitch, head_pitch, head_yaw, head_roll;
- `body[0]`, the **bounce flag** (hs13): 1 knee-bounces to the beat while standing — a TRAINED
  regime (the user's 108 BPM request; `scratch/bounce_probe.py` measured the timeline route FLAT,
  0.003 m, 0 of 42 beats). It replaces hs11/hs11b's parked footwork on this slot; the show keeps
  it 0 until the regime is trained AND stage-verified, then the 0-2 s opening raises it;
- `body[1]`, the **downrock sweep flag** (hs12): 1 sweeps a leg around while the sit crouch holds
  (hs12b's live move, 419° inside the show window; hs12c's grinder reshape FAILED by measurement
  and was reverted — see the Runs row);
- `body[2]`, the **cross-step flag** (hs13): 1 alternates hip_pitch steps (Indian cross-steps /
  kick-outs) ON the bounce — the command carries it only while the bounce flag itself rides, so
  steps never stand alone (hs11's lesson applied: footwork needs a moving body);
- `body[4]`, the **L-kick freeze flag** (hs12b, trained — the live show raises it at 18.0 s): 1 holds
  the freeze on the head — one leg back to the vertical needle, the other level; 0 the ordinary
  two-level headstand;
- `body[3, 5]`, unused (zeros).

A short `timeline.json` (Pollen's, 9 keyframes over 20 s, is in this folder):

```json
{"timeline_version": 1, "duration_s": 20.0, "keyframes": [
  {"t": 0.0,  "twist": [0, 0, 0], "head": [0, 0, 0, 0]},
  {"t": 2.0,  "twist": [1, 0, 0], "head": [0, 0, 0.5, 0]},
  {"t": 7.0,  "twist": [0, 0, 0], "head": [0, 0.4, 0, 0]}
]}
```

From each keyframe's `t` until the next one, the policy reads that keyframe's numbers. A group a
keyframe leaves out is zeros, not the previous value. The first keyframe is at 0, times go up, at most
1000 keyframes, every number within ±10, `duration_s` at most 20.

A performance can also make the duck heard, with nothing for the policy to learn: `triggers` in a
keyframe (RT quacks, LT wheees), a `song` of notes beside the keyframes, and a `voice` block that
tunes the duck's voice. The Arena's
[Got a Talent guide](https://pollen-robotics-microduck-arena.hf.space/#/e/got-a-talent/guide) has the
rules. Offer it to the user when the performance wants sound; never add it without asking.

A performance can also be a **troupe** of up to 8 ducks in one shared physics. The star is this
folder's policy (`"self"`); the others are any policy, the user's or a public one pinned to a full
commit, on legs or rollers. Each duck has its own place, keyframes, voice and song, and each duck's
keyframes are commands for its own policy. The Arena's
[Got a Talent guide](https://pollen-robotics-microduck-arena.hf.space/#/e/got-a-talent/guide) has the
format and the rules in its Troupe and Placement sections. Offer it when the user wants company on
stage; never add a duck, nor borrow someone's policy, without asking.

**What the policy can do is what training showed it.** This training samples:

- the sit flag at exactly 0 or 1 (values in between were never seen);
- a posture held 3.5–6.5 s before it may flip; each sit or rise is paced by a 2 s ramp (on the
  stage the reference sits in about 0.8 s and rises in about 0.3 s);
- head angles up to ±1.10 (neck_pitch), ±1.10 (head_pitch), ±1.40 (head_yaw), ±0.31 rad (head_roll)
  once the head curriculum has widened (from iteration 2000), a new head target every 2–5 s.

A test (`test_the_reference_timeline_stays_inside_what_training_saw`) holds this folder's
`timeline.json` inside a **stage-verified** envelope: flags 0 or 1; per-dim head angles within
40 % (pitch dims), 55 % (yaw) and 70 % (roll) of the widened ranges; posture segments at least
2 s apart; head targets at least 0.5 s apart. Those last two are deliberate widenings for the
verified breakdance choreography (the 2 s knee-drop sit, the 0.6–0.7 s beat-bop nods and 0.7 rad
judge glances all played the full 20.00 s on the Arena's stage with no fall); the original
sampling envelope was 3.5 s dwells and 2 s head targets. Do not widen the test further without a
fresh stage check. A timeline outside it may still play well: check it on the stage, and if the
user wants it anyway, adjust the test with them.

## Files

| File | What it holds | Change it? |
|---|---|---|
| `timeline.json` | The performance | Yes: the user's creative part, and free to change (no training) |
| `challenge.toml` `[params]` | The three knobs below | Yes. Keep each value within its `min`/`max` (the workshop refuses others) |
| `challenge.toml` `[recipe]` | `num_envs` 4096, `max_iterations` 3000, `seed` 1 | Yes, when agreed with the user |
| `challenge.toml` `[command]` | What the policy reads in each command group; `publish` writes it into the manifest | Only with the commands' meaning |
| `env.py` | The whole environment (microduck_rl's sit & stand task): posture rewards, head command and its curriculum, domain randomization, delayed pushes, `RL_CFG` | Yes, once the knobs are not enough |
| `rewards.py` | The headstand and headspin: their command (`HeadstandCommand`), reward terms, the headstand spawn, the joint-limit penalty | Yes |
| `check.py` | Judges a policy's sit/stand and head moves against a timeline on the Arena's stage | Use it as is |
| `headspin_check.py` | Judges a policy's headstand and headspin on the Arena's stage, several seeds | Use it as is |
| `headspin_view.py` | Plays a policy's headstand and headspin live in a MuJoCo viewer window | Use it as is |
| `tasks.py`, `event`/`task`/`kind` in `challenge.toml` | Registration with mjlab and the Arena | No |

| Knob | Label | Default | Range | Sets in `env.py` |
|---|---|---|---|---|
| `head_reward` | Follow the head moves | 0.75 | 0.0–2.0 | `head_pose_tracking` weight |
| `stillness_reward` | Hold still between moves | 2.0 | 0.0–4.0 | `posture_stillness` weight |
| `upright_reward` | Stay upright | 2.5 | 0.5–5.0 | `upright_linear` weight |

Do not change, or the Arena cannot play the policy or the tests fail:

- the robot (`MICRODUCK_STANDUP_ROBOT_CFG`: the walk robot with the Arena's collision set) and the
  61-number observation layout: an unused command slot stays zero-padded, never deleted;
- the contact solver at the Arena's 10/20 iterations, and no fall termination (the stage plays on
  after a fall, so training does too);
- `experiment_name = "got_a_talent"` in `RL_CFG`: runs must land in `logs/rsl_rl/got_a_talent/`;
- every domain randomization range must contain the nominal value the Arena simulates (the Arena
  randomizes nothing but the start pose and the servos' command delay; no pushes, no noise);
- `pyproject.toml`, `uv.lock` and the other challenges' folders.

The robot model and the reward functions (`microduck_mdp`) live in the `mjlab-microduck` package
and are not edited here. Its [AGENTS.md](https://github.com/pollen-robotics/microduck_rl/blob/develop/AGENTS.md)
holds the reward-design rules learned on this robot: read its "Reward design" section before
writing a reward term.

## Working on a performance

### Once per session

```bash
uv sync --group workshop
nvidia-smi              # an NVIDIA GPU here trains locally; otherwise Hugging Face Jobs
uv run hf auth whoami   # the account a job runs and bills under, and publish publishes to
```

Pollen's reference policy lets you try a timeline before any training of the user's own:

```bash
uv run hf download pollen-robotics/microduck-got-a-talent-reference policy.onnx --local-dir logs/reference/got-a-talent
uv run --group workshop python -m microduck_challenges.got_a_talent.check logs/reference/got-a-talent/policy.onnx
```

With this folder's timeline it passes: each sit reached in 0.78 s, each stand in 0.32–0.34 s, the
head within 0.06 rad, no fall. Add `--timeline <file>` to check another timeline.

Agree with the user before the first run: the performance they want; whether their policy (at
first, the recipe trained as is) can already perform it, so that only `timeline.json` changes; and
the budget (local GPU, or Hugging Face Jobs: default hardware `l4x1`, $0.80 per hour, billed to
their account, which needs a payment method; the 3000-iteration recipe took 2 h 20).

### Editing the timeline

1. Write it with the user: what happens when, in plain words, then as keyframes.
2. Check it on the stage with the policy it will be published with:
   `uv run --group workshop python -m microduck_challenges.got_a_talent.check <policy.onnx>`
   prints a line per keyframe (posture reached and held, head error, fall) and PASS or FAIL;
   `uv run --group workshop arena score <policy.onnx> --event got-a-talent --timeline src/microduck_challenges/got_a_talent/timeline.json`
   prints the Arena's own summary (length played, fall, seconds seated and standing).
3. Show it: the workshop (`uv run --group workshop workshop got-a-talent`) edits `timeline.json` on a
   training's page, places a troupe's ducks on a plan of the stage, and plays it with the checkpoint the user picks.
4. Commit it: `git add src/microduck_challenges/got_a_talent && git commit -m "<what the performance does>"`.

### Each training

1. **State the hypothesis** in one sentence: what the policy cannot do yet, what you change, what
   you expect to see.
2. **Change one thing**, a knob first. Combine changes only on purpose, and say so.
3. **Test**: `uv run --group workshop pytest`.
4. **Smoke test** (one to two minutes, also on a machine without a GPU):

   ```bash
   uv run train Mjlab-GotATalent-MicroDuck --gpu-ids None --env.scene.num-envs 2 \
       --agent.max-iterations 2 --agent.logger tensorboard --agent.run-name smoke
   rm -rf logs/rsl_rl/got_a_talent/*_smoke
   ```

5. **Commit** the change, its message the hypothesis:
   `git add src/microduck_challenges/got_a_talent && git commit -m "<hypothesis>"`. `publish` writes
   that commit into the policy's recipe, and refuses a run started from uncommitted changes.
6. **Launch** in the background, the output in a log under `logs/` (git-ignored). `<name>` is short,
   unique, and says the change (`head1`, `still3`):

   ```bash
   # local NVIDIA GPU
   uv run train Mjlab-GotATalent-MicroDuck --env.scene.num-envs 4096 --agent.max-iterations 3000 \
       --agent.seed 1 --agent.logger tensorboard --agent.run-name <name> > logs/<name>.log 2>&1 &
   # Hugging Face Jobs: the same, plus
   #   --hf-jobs --no-wandb --namespace <user> --ckpt-repo <user>/mjlab-gotatalent-microduck-<name>
   ```

   A local run writes `logs/rsl_rl/got_a_talent/<YYYY-MM-DD_HH-MM-SS>_<name>/`: `model_<N>.pt` every
   250 iterations and the last one, TensorBoard events, `provenance.json`. A job uploads the same
   tree to its checkpoint repo. Its log prints `[job] id: <id>`. Killing the local process only
   detaches from the job: stop it with `uv run hf jobs cancel <id> --namespace <user>`.

### Following a run

- Progress: `grep -o "Learning iteration [0-9]*/[0-9]*" logs/<name>.log | tail -1`.
- A job's checkpoints and curves: `uv run hf download <user>/mjlab-gotatalent-microduck-<name> --local-dir logs/rsl_rl`
  (the repo mirrors `logs/rsl_rl/`); its state: `uv run hf jobs inspect <id> --namespace <user>`.
- Check once per checkpoint (every 250 iterations, several minutes apart), not more often.
- Curves (`Episode_Reward/<term>` is the weighted value; penalties must read ≤ 0):

  ```bash
  uv run python -c "
  import sys; from tensorboard.backend.event_processing.event_accumulator import EventAccumulator as E
  ea = E(sys.argv[1], size_guidance={'scalars': 0}).Reload()
  for tag in sys.argv[2:]: print(tag, [(e.step, round(e.value, 3)) for e in ea.Scalars(tag)][-8:])
  " logs/rsl_rl/got_a_talent/<run> Episode_Reward/posture_composite Episode_Reward/posture_stillness Episode_Reward/head_pose_tracking
  ```

  There is no fall termination, so episodes always last their 12 s: the episode length says nothing.
  `posture_composite` (height × upright × pose against the commanded posture) is the main task
  term: it must keep growing. Curriculum stages land at iterations 500, 750, 1000, 1250, 1500, 2000
  and 2500 (the head range widens up to 2000, pushes come in from 1000): a drop right at one means
  the pacing is wrong. A run that has collapsed and not recovered within about 500 iterations: tell
  the user and propose stopping it.

### Judging every checkpoint

The curves are not the performance. Judge each checkpoint on the Arena's stage with the timeline
the user will publish:

- If the user has the workshop open, it finds every run under `logs/rsl_rl/`, yours included, and
  plays the timeline with each checkpoint as it arrives. Its results:
  `logs/rsl_rl/got_a_talent/<run>/workshop/model_<N>.json` (`stage`: `fell_at`, `seated_s` and
  `standing_s` for each duck, the star first).
- Otherwise:

  ```bash
  run=logs/rsl_rl/got_a_talent/<run>
  for n in $(ls $run | sed -n 's/^model_\([0-9]*\)\.pt$/\1/p' | sort -n); do
    uv run publish --run $run --checkpoint $n --repo local/microduck-eval --dry-run --allow-dirty >/dev/null 2>&1 &&
    echo "checkpoint $n: $(uv run --group workshop python -m microduck_challenges.got_a_talent.check publish-eval/policy.onnx | tail -1)"
  done
  ```

  `--dry-run` exports exactly as the real publish (the observation normalizer baked in) into
  `publish-eval/`, and publishes nothing.

Pick a checkpoint that passes, then the one with the smallest head error and the quickest sits and
rises (the per-keyframe lines). The Arena plays one start, without training's randomization: a
checkpoint whose neighbours pass too is the safer pick. The reference is checkpoint 2000 of its
run, chosen that way: replayed under training's randomization, checkpoints 1000–1750 sat in
1.5–2.5 s instead of about 1 s. The best checkpoint is often not the last. GPU training varies a lot
from run to run: before building on a surprising result, run the same commit with another
`--agent.seed`.

### Reporting

After each run, tell the user in a few lines:

```
run head1 (commit 1a2b3c4: head_reward 0.75 → 1.0, to follow the head moves more closely)
best: checkpoint 2250, PASS, head error 0.04 rad, sits in 0.80 s, rises in 0.35 s   (reference PASS, 0.06 rad)
others: 1500 FAIL (sit reached after 1.62 s) · 2000 PASS 0.05 rad · 2999 PASS 0.07 rad
next: <your proposal and why>
```

### Publishing, on the user's yes

Publishing makes the policy and its performance public on their Hugging Face account and enters it
on the Arena, which plays it and prints the page of its result. **Pass the timeline**, or the entry
stands still for 20 s:

```bash
uv run publish --run logs/rsl_rl/got_a_talent/<run> --checkpoint <N> --repo <user>/microduck-<name> \
    --no-private --timeline src/microduck_challenges/got_a_talent/timeline.json --enter
```

`--no-private` because the Arena reads public repos only; add `--livery <colour>` for its colour.
For the policy's code link to open, push the commit to the user's fork first (`origin` must be their
fork, never `pollen-robotics`; the workshop's home sets that up).

## What is known about this challenge

- **The timeline keeps the time, not the policy.** Before this format, the challenge asked the
  policy to time a sit-and-stand routine on its own under a constant command: ten runs failed. A
  clock kept in the policy's own joints washes out under PPO's exploration noise within seconds.
  Any new move should be triggered by a command, never by the policy counting.
- **This env starts from a recipe that works**: microduck_rl's sit & stand task, whose published
  policies sit in about 1 s and rise upright in under 0.5 s. Its solver went from 30/50 to the
  Arena's 10/20 iterations, and it still sits and rises.
- **Pushes come late on purpose**: in an earlier sit task, pushes during the descent, before the
  transitions had consolidated, made the policy unlearn sitting and just stand.
- **Judge the exported, deterministic policy.** One failed attempt completed its moves only with
  exploration noise on; deterministic, it never moved. `check.py` and the workshop judge the export.
- **Reward traps met on this robot**, worth knowing before adding a move: a state that earns the
  same before and after a move (standing still) means the move is never started; a reward window
  that closes before the move completes makes the policy unlearn it; a "rise" that was really a
  lean back at 57°, just under the 60° fall threshold.
- **A new kind of move needs a command for it.** The 13 command numbers are fixed (the real robot
  feeds the same 61 observations). `twist[1]` and `twist[2]` now carry the headstand and the spin;
  the free slots left are `body` (6). A new move reads one of them, is trained on it, and
  `[command]` in `challenge.toml` says what it means.

## The headspin (work in progress)

The performance being built: the duck stands, goes onto its head on a timeline command, holds a
headstand, then spins on its head like a b-boy, all on the stage's grippy floor. Any agent can pick
this up from here: the strategy, every run and its lesson, and the open problems are below.

### Start here (picking this work up)

1. **Get the work**: it lives on the private repo `electricshaka/microduck-private`, branch `main`
   (a checkout of `pollen-robotics/microduck-challenges` on its own `main` has none of it).
2. **The trained checkpoints are committed** under `checkpoints/got_a_talent/` (training's `logs/`
   is git-ignored): **`hs12b/model_2000` is the live performance** (see Best checkpoints below),
   with `hs12/model_2250`, `hs11b/model_1250`, `hs11/model_1250`, `hs10/model_1750`,
   `hs9/model_1000`, `hs8/model_500`, `hs6/model_250`, `hs5/model_250`, `hs2/model_1250` behind it.
   Each folder has the `.pt` (to resume training), the exported `.onnx` (to judge, watch or
   publish), its `judge_<N>.txt` and `provenance.json`.
3. **Watch it**: `uv run --no-sync --group workshop python -m microduck_challenges.got_a_talent.headspin_view checkpoints/got_a_talent/hs12b/model_2000.onnx --timeline src/microduck_challenges/got_a_talent/timeline.json --subsample 2`
   — always pass `--subsample 2` (render every 2nd physics step, 25 FPS): at full rate the GLFW
   window freezes under the CUDA load of a 4096-env training run sharing the GPU.
4. **Judge it**: `headspin_check` and `check.py` on the `.onnx` (see Judging below); the downrock sweep: `scratch/sweep_probe.py <onnx>` (acceptance) and `... full` (the sweep measured inside the real show); the L-kick freeze: `scratch/kick_probe.py <onnx>` (quiet headstand window then the flag — acceptance reads the freeze window).
5. **Resume training** (hs12b is done — L-kick freeze learned and live; **hs12c, the grinder, is
   DONE — FAILED BY MEASUREMENT, fallback (c)**: all 12 checkpoints judged, the ban starved the
   sweep (frontier flat 0.002; probe 0-5° vs the 777° live), the grinder terms are REVERTED and
   hs12b's sweep shape is the live show again; judging records in `checkpoints/got_a_talent/
   hs12c/`, nothing archived — no checkpoint beat `hs12b/model_2000`; **hs13, the breakdance
   overhaul, is RUNNING, resuming from `hs12b/model_2000` per fallback (c)**). No prewarm surgery:
   the `hs10_base` prewarm woke the WHOLE body block and every run since trained on it, so all body
   dims are alive in `hs12b/model_2000` (hs13 opens body[2] at its prewarm values — verify-then-copy
   says stage byte-for-byte, which is exactly what `hs12b_base/` holds). The verify-then-copy rule
   (VERIFY alive dims before staging; patching alive dims shifts the hidden units and breaks the
   "resume starts at the measured behaviour" promise — the one run that loaded a patched copy, hs12,
   only did so because the patch beat the rewrite to the load) is what `scratch/prewarm_hs12b.py`
   enforced before hs12b (measured: body[4] normalizer std 0.398, first-layer |w| ≤ 0.26 — a raised
   flag lands at ~+2.5, near-neutral). The pattern, as hs13 stages (run 2026-10-25):

   ```bash
   # base already staged byte-for-byte: logs/rsl_rl/got_a_talent/hs12b_base/model_2000.pt
   PYTHONUTF8=1 MICRODUCK_WARM_START=1 uv run --no-sync train Mjlab-GotATalent-MicroDuck --env.scene.num-envs 4096 \
       --agent.max-iterations 3000 --agent.seed 1 --agent.logger tensorboard --agent.run-name hs13 \
       --agent.resume True --agent.load-run hs12b_base --agent.load-checkpoint model_2000.pt > logs/hs13.log 2>&1 &
   ```

   Commit the change first (one change per run), and add any checkpoint worth keeping to
   `checkpoints/got_a_talent/` the same way. Push only to the user's repos (never `pollen-robotics`).

### Why a headspin

On the board (2026-10-22) the best vote rates went to single, physical stunts: dashi's trained
backward roll 46 %, juicenv's variety show 37 %; troupes with songs 18–30 %. Nobody has a
headspin.

### What the robot and the Arena allow (measured in the Arena's physics)

- **The floor grips**: friction 1.0 against every robot geom, the Arena's, and a test holds training
  to it. MuJoCo takes the larger friction of a pair, so nothing on the robot side lowers it. Each
  contact point has no torsional friction (condim 3); the braking of a spin comes from the head
  touching at several points. A free spin on the head slowed from 7.9 to 3.7 rad/s in 0.2 s.
- **No arms**, so no hand taps. What can add spin is any joint that twists the body against the
  head's grip: the neck's head_yaw (±170°) whipped the body up to 10–13 rad/s in one swing with the
  head planted; leg swings work the same way. Opening and shutting the legs (the skater trick)
  changes the speed by only ~5% on this robot (light legs) and adds no spin.
- **Spin did not stabilise an open-loop headstand**: dropped spinning on the head with no control,
  20 rad/s fell off sooner than 8 rad/s. Balance has to be active; a trained policy may differ.
- **The geometry**: hips open only ±22° sideways (no side split), ±90° front and back (a scissor
  split). Headstand pose: neck_pitch +0.5, head_pitch +1.0 rad (absolute), legs at HOME; the trunk
  then sits at z 0.15 m. Pollen's forward roll passes through the headstand 0.1–0.25 s in and
  overshoots it: getting up is easy, stopping there is the skill.
- **The stage labels a headstand a fall** (trunk past 60° for 0.3 s) but plays on: harmless.

### What b-boys do, and the duck's version

From [The Conversation](https://theconversation.com/how-do-breakdancers-avoid-breaking-their-necks-231579),
[Scientific American](https://www.scientificamerican.com/article/the-physics-of-breakdancing-a-new-olympic-sport/),
[bboydojo](https://bboydojo.com/breakdance/how-to-do-a-headspin/) and
[Flashmavi](https://www.flashmavi.com/learn/learn-the-headspin):

| B-boy technique | Duck version | Where |
|---|---|---|
| Torso stacked vertically over the head, neck stiff | Hold: upside down within ~25°, only the head down, trunk at 0.15 m, low pitch/roll rates | `headstand_hold` |
| Hand taps / arm pumps keep the spin going | Neck whips and leg swings twisting the body against the head's grip | Free to the policy; `headspin` pays the turn |
| Legs together = faster ("drills"), out = slower and steadier | Small effect here; the policy may use it | Free to the policy |
| Hips swing to start the turn; learn quarter, then half turns | Twisting entry, spawns already spinning; any new turn pays | `spawn_headstand`, `headspin` |
| Low-friction cap on the crown | Not available; a head pose that touches at one rounded point brakes less | Free to the policy |

### Commands and rewards (rewards.py, env.py)

- `twist[1]` **headstand flag**, latched in training (once up it stays up to the episode's end);
  raised with `HEADSTAND_PROB` 0.15 per resample, and `HEADSTAND_SPAWN_PROB` 0.2 of episodes start
  on the head (some spinning up to ±6 rad/s). `twist[2]` **spin flag**, drawn with `SPIN_PROB` 0.5
  at every resample while the headstand flag is up.
- `body[0]` **bounce flag** (hs13): drawn with `BOUNCE_PROB` 0.5 per resample, rides only a
  standing posture. While it rides, `bounce_hold` (6.0) pays a DENSE window (0.75 s) that only a
  VALID cycle opens — drop ≥ 0.012 m from the cycle's own top, top ≥ 0.108 (it came back UP),
  mean |knee−home| ≥ 0.20, a foot planted, ≥ 0.30 s since the last (cycle state lives on the
  command term). `spawn_mid_bounce` (30%) starts episodes at the measured trough, one hip shown
  forward, a small sink, marking the bounce AND step flags. The hs11/hs11b toprock terms
  (`toprock_step`, `toprock_sway`) and `spawn_toprock` are RETIRED registrations (hs13: the
  functions and their tests stay in `rewards.py`, unregistered). `body[2]` **cross-step flag**
  (hs13): drawn with `STEP_PROB` 0.5, carried ONLY while the bounce flag itself rides —
  `bounce_step` (4.0) pays the dense window opened by ALTERNATING hip_pitch peaks 0.45 rad past
  home (left negative/right positive), ≥ 0.25 s apart. `posture_stillness` stays paused under
  body[0] (`unless_toprock`, now the bounce flag); the sit composite KEEPS paying: the bounce's
  top IS the stand, this regime replaces no rest target.
- `body[4]` **L-kick freeze flag** (hs12b): drawn with `KICK_PROB` 0.5 per resample and forced down
  unless the headstand flag is up (`ToprockCommand` masks it, as the sweep rides the sit). While it
  rides, `l_kick_hold` (6.0) pays DENSE: exp on the leg pose — one leg level (hip 1.42 the measured
  level side, knee straight), the other the HOME needle (measured: HOME hips put the foot 0.117 m
  ABOVE the inverted trunk; ±1.42 brings the foot level with it — `scratch/kick_env_probe.py`) —
  gated on the head planted and trunk z ≥ 0.13, either leg may carry the kick (min of the two
  variants). `spawn_l_kick` (30%) starts episodes already IN the freeze, at rest, marking the
  headstand flag too; it skips episodes spawn_headstand/spawn_seated_sweep already claimed.
- While the headstand flag is up, the sit/stand terms in `REST_STACK` are off (`unless_headstand`),
  and these pay: `headstand_progress` (2, upside down AND trunk off the floor), `headstand_support`
  (2, head-only contact, trunk off the floor), `headstand_hold` (8, the multiplicative goal),
  `headstand_body_contact` (-2, trunk or a leg on the floor), `headspin` (6, new counter-clockwise
  turn on the head past the furthest so far, `rate_max` curriculum 10 → 20 rad/s at the run's
  stages; back-and-forth pays nothing), `head_stillness_on_head` (3, exp quiet-pivot on head joints),
  and since hs10 `head_spindle_penalty` (-1, non-saturating quadratic on joints 5-8 off the spindle
  pose [0.5, 1.0, 0.0, 0.0] rad — see "The next phase").
- Always: `head_contact_unless_headstand` (-2, the head on the floor outside a headstand) and
  `action_beyond_limits` (-0.5, squared distance of each joint target past its soft limit).

### Runs (local RTX 4090 / DGX Spark GB10; each continues from the last good checkpoint)

| Run | Commit | Change | Result | Lesson |
|---|---|---|---|---|
| hs1 | fb9241b | headstand flag, rewards, spawns | Lay upside down rocking head/trunk at z 0.04 m; hold 0.2 s | Shaping that only checks "inverted" pays lying down: gate it on trunk height |
| hs2 | a1e8f2c | shaping needs the trunk off the floor; -2 body contact | Tripod at 250, one foot at 500; **checkpoint 1000-1250: onto the head in 0.40 s, held 12.6 s, 3/3 seeds** | Works. But sit/stand FAIL from 250: neck folded to its limit |
| hs3 | 3d42daa | HEADSTAND_PROB 0.35 → 0.15 | Neck still pinned; target asked -9.5 rad | Not a share problem |
| hs4 | 35cea14 | penalise targets past joint limits | Target back to -1.7 rad, but stood as a tripod (feet + head down), 36° lean | A target past a limit gives no gradient; fix that first |
| hs5 | d710797 | -2 head on the floor outside a headstand | Checkpoint 250: head error 1.92 → 0.57 rad, stands 3/3, headstand 12.5 s | The tripod was the trap; still above check.py's 0.25 rad |
| hs6 | 3c688fa | spin flag + `headspin` reward | Paused at iteration ~330. Checkpoint 250: headstand holds 12.5 s with spin flag up, but no spin: 0.01 turns; `Episode_Reward/headspin` 0.007 | Spin not discovered: a slow first turn (1 rad/s) earns 0.4/step against hold's 8 |
| hs7 | d715e29 | "quarter turns first" curriculum (`rate_max` 3.0 → 15.0 rad/s) | **Checkpoint 1000: +12.60 CCW turns @ 7.9 rad/s**, 100% continuous hold on head (12.65 s) | Broke reward trap: early turns pay 2.0/step instead of 0.4. But policy whips neck against floor |
| hs8 | 4a66303 | `head_stillness_on_head` quiet-pivot regularizer (weight 3.0) | **Checkpoint 500: +13.86 CCW turns @ 8.7 rad/s**, head roll wobble -95%, hold reward peak 2.78 | Stabilized head roll to 0.012 rad; neck yaw still oscillated ±31° because Gaussian tails vanish |
| hs9 | 09d00fb | scale to 8192 envs on DGX Spark (GB10), `rate_max` 10 → 20 rad/s | **Checkpoint 1000: +23.52 CCW turns @ 14.8 rad/s** (peak 18.0 rad/s), +24.59 turns on full timeline, 18.16 s continuous hold | Unlocked Stage 3 (20 rad/s). Massive power stroke, unbroken hold across seeds |
| hs10 | ba5e993 | `head_spindle_penalty`: −1.0 quadratic on joints 5-8 off [0.5, 1.0, 0, 0] rad while the flag is up, to kill the ±31° neck-yaw whip and force leg-driven spin | **Checkpoint 1750: +23.95 CCW turns @ mean 15.1 rad/s, 3/3 seeds, hold 12.68 s, sit 0.64 s** — spindle penalty −0.07 at its floor: the neck is quiet and the legs carry the spin | Spindle lock holds turns AND kills the whip (penalty ~0 vs hs9's ±31° oscillation); but the deterministic-stage **sit flickers checkpoint to checkpoint** (250, 500, 1000, 2000, 2500, 2750, 2999 sat late/never; 750, 1250, 1500, 1750, 2250 sat in 0.6–1.2 s): judge the sit line per checkpoint, and pick one whose neighbours pass too |
| hs11 | ca08bcd | toprock on `body[0]`: standing-only 0/1 flag, `toprock_step` paying each alternating landing (2 Hz cap), hip_roll sway, stillness paused under the flag; dead body dims prewarmed in the normalizer (mean 0.25 / var 0.1875, first-layer cols zeroed — `scratch/prewarm_hs11.py`) | **Footwork never discovered: 0 of 12 checkpoints ever stepped** on stage (probe: 1 touchdown, hip_roll ≤ 0.02 rad, flag up or down; `toprock_step` flat 0.004–0.008 for 3000 iters). Spin meanwhile hit record: **1750 +25.88 @ 16.26** (never sat), and **2999 collapsed (+0.01 turns)** | A reward that pays ONE frame per landing amortizes a full step to ~0.004/step — under the posture cost of lifting a foot, so no gradient; the same trap hs6's spin had, which only a DENSE pay fixed. And the final checkpoint of a run can collapse: never archive the last one without judging it |
| hs11b | ff66846 | dense rhythm window (0.75 s) + mid-step spawn (prob 0.25) to unlock toprock footwork | **Footwork failed again: the probe reads 1 touchdown at all 12 checkpoints, flag up or down** — the step curve grew 40× (0.004 → flat ~0.16 from iter ~50 on) but that is the *spawned* landing harvesting the window, not learned steps. The run's other lines improved: **1250 +24.25 turns @ 15.24 rad/s, hold 12.68 s, sit 0.46 s (fastest on record), dive 0.28 s**; 1500 peaked 18.17 rad/s; no collapse anywhere | Dense pay amplifies whatever the state already contains — a seeded landing included; harvesting the seed is free, so producing a step was never needed. Two mechanisms, two runs, one verdict: footwork on this robot is not reward-shapeable — `body[0]` parked. Sit and spin improved on the way |
| hs12 | 1419963 | downrock sweep on `body[1]`: `downrock_sweep` (6.0, frontier CCW trunk yaw/sec dense in the crouch band, flag-gated), `downrock_leg` (2.0 straight-leg shape), `spawn_seated_sweep` (35% mid-sweep), stillness pause under the flag, curriculum 2.0→3.5→5.0 rad/s | **Downrock LEARNED: 2250 sweeps 837.2° per crouch (probe), crouch 3.00/3.00 s, quiet crouch 0.8°; +24.01 turns @ 15.09, sit 0.50 s, dive 0.26 s — best show checkpoint yet; on the full timeline 464° inside the 2 s knee drop, committed as the live show.** Neighbours 2000 (837°, sit 0.56) and 2500 (804°) pass; 2750/2999 decayed (+19.55/+18.60, sit missed) at the 2500 stage | The frontier pay + seeded start works where footwork failed because the crouch is a state the robot HOLDS — the seed is a starting position, not the whole move, so producing more of it is the only way to earn. Seasoning died: `downrock_leg` flat 0.004 all run — the duck grinds the sweep without the straight-leg shape; drop dead seasoning terms next run. Late-run decay at the final curriculum stage struck a second run: judge the last stage's checkpoints hardest |
| hs12b | 7b551e4 | L-kick freeze on `body[4]`: `l_kick_hold` (6.0, DENSE exp on the measured leg pose — one leg level at hip ±1.42 knee straight, the other the HOME-hip needle — gated on the kick flag riding a headstand with the head planted and trunk z ≥ 0.13), `spawn_l_kick` (30% IN the freeze, at rest, marking the headstand flag too), kick flag rides only a commanded headstand; `downrock_leg` registration RETIRED (hs12's recorded lesson: flat 0.004). Resumed byte-for-byte from `hs12/model_2250` (`scratch/prewarm_hs12b.py` verified body[4] std 0.398 / \|w\| ≤ 0.26) | **L-KICK FREEZE LEARNED: 2000 freeze kick err 0.01 rad, NEEDLE 0.10 rad (baseline ~1.7 — the flag pulls one leg back to vertical), up 3.00 s, drift 11°, quiet window keeps the two-level pencil (needle 1.74); +21.67 turns @ 13.62, hold 12.69, dive 0.30, sit 0.34; head 0.42/0.37/0.37 the only checkpoint UNDER the hs12 baseline. Sweep survived (show 419° inside the window, probe 777°/crouch), arena 20.00 s — committed as the live show; timeline now raises `body[4]=1` at 18.0 s (guard test widened citing the verification).** Ladder: 1000–2000 all gates (5 consecutive; 1000 spun +23.47 but head 0.62 at 4.5 s), 2250/2500/2750 freeze passes with spin dips (+19.4/+19.1/+17.8), 2999 decayed (+13.66) | The seeded-hold recipe worked a third time (hold this time, not a turn): 500 already had the needle (0.10) — the seed + dense pay find the shape fast, the long run then refines balance around it. Cost measured honestly: the freeze pay trades ~2–4 spin turns (24.0 → 21.7) for the leg shape; head error IMPROVED (kicking one leg stabilises the spindle?). And the final-stage decay struck a THIRD run — the 2500-stage curriculum is a proven hazard, judge those checkpoints hardest |
| hs12c | fb19d43 | **Grinder sweep — the butt-spin ban** (user's video review of hs12): `downrock_sweep` frontier now REQUIRES a kneel regime (trunk z ≥ 0.085 — flat crouch rests 0.061 — planted knee/foot via body+feet sensors) and pays MULTIPLIED by blade shape (hip ±1.2 knee 0: foot radius 0.089 m vs 0.067 tucked, `scratch/sweep_env_probe.py`); `sweep_progress` (2.0) funds the regime before the turn exists; `sweep_pelvis_contact` (−3.0 STRICT, new `pelvis_ground` sensor: pelvis/trunk geoms ONLY — a planted knee is support, a planted pelvis is the cheat); the sit composite PAUSES under the sweep flag (`unless_sit_rest`); seed shows the shape (blade offset, z 0.095). Resumed byte-for-byte from `hs12b/model_2000` | **Grinder FAILED BY MEASUREMENT — fallback (c). All 12 checkpoints judged** (`checkpoints/got_a_talent/hs12c/judge_*.txt`): frontier flat 0.002 for all 3000 iters; probe sweeps 2.8°/5.4°/3.7° at 1750/2000/2250, and from the 2500 stage the duck **never crouches at all** (0.0°/4.6°/4.5°, min z 0.115; bar ≥ 400°). Fallback (a) precondition never held (blade radius 0.042 med, 0% ≥ 0.08), nor (b) (pelvis contacts 0 all run — the cheat never appeared, the ban was never tested). Spin best at **1500: +22.82 @ 14.34** (best since hs11, kick freeze intact: needle 0.09, 3.00 s) then the 2500-stage decay struck a FOURTH run: 2750 +15.93 @ 10.01, 2999 +14.37 @ 9.03, holds collapsing to 4–7 s. No checkpoint beats hs12b/model_2000 — nothing archived, grinder terms REVERTED the verdict commit (hs12b shape restored; functions/sensor/tests stay, unregistered) | A gate that demands a posture the robot has never visited STARVES the move it funds: the kneel-and-blade was unreachable from the sit crouch, so the frontier paid nothing anywhere and by the last stage the crouch itself died — pay must be reachable from where the policy already stands, and withholding it from a working move to force a new one trades a measured 777° for zero. Fourth consecutive run decaying at the 2500-stage curriculum: treat it as law — judge those checkpoints hardest and expect to cut back to ≤ 2250 |
| hs13 | (this commit) | **Breakdance overhaul — the user's Phase-1 fix** (their review: the opening "just looks side to top with head"; bounce_probe.py: the timeline bounce is FLAT): body[0] re-purposed to the **knee BOUNCE** as a trained regime — `bounce_hold` (6.0) pays a DENSE window that only VALID cycles open (drop ≥ 1.2 cm from the cycle's own top, REACHED top ≥ 0.108, knees flexed ≥ 0.20 rad, foot planted, ≥ 0.30 s apart; `scratch/bounce_env_probe.py`: knees ±0.35-0.50 + mirrored hip bend drop the trunk 1.5-2.2 cm, stay VERTICAL, feet planted, trough z 0.093-0.105 clear of the crouch); body[2] **cross-steps** ride ONLY the bounce flag — `bounce_step` (4.0) pays alternating hip_pitch peaks (0.45 rad forward, left negative/right positive, 0.25 s cap; probe: reaches the foot 2.3-3.2 cm forward, a shuffle not a hop); `spawn_mid_bounce` (30%) seeds the measured trough with one hip shown forward and a small sink; the hs11/hs11b toprock terms + seed are RETIRED registrations (functions/tests stay); stillness pause kept under body[0]; the sit composite KEEPS paying (the bounce's top IS the stand — this regime replaces no rest target). Resumes byte-for-byte from `hs12b/model_2000` — hs12c fell to fallback (c), whose written base for the overhaul is hs12b again (body dims alive; body[2] at prewarm values — zero first-layer weight, normalizer std ≈ 0.433, near-neutral, weights learn to grow) | **RUNNING** — launched 2026-10-24 21:39 (run folder `2026-10-24_21-39-45_hs13`, ~3 s/iter). Warm start verified in the log (weights/normalizer/optimizer kept, curricula restart); reward table reads `bounce_hold` 6.0, `bounce_step` 4.0, `downrock_sweep` 6.0, `l_kick_hold` 6.0 as committed; at iter ~50 sweep 0.090 and kick 0.012 (carryover holds), `bounce_hold` 0.0000 (expected: only the 30% spawn yet) | — |

Best checkpoints committed in `checkpoints/got_a_talent/`:
- **`hs12b/model_2000`** (live): the first checkpoint with the **L-kick freeze** — the flag pulls
  one leg back to the vertical (freeze kick err **0.01 rad, needle 0.10 rad**, baseline needle err
  ~1.7) while the other stays level, up 3.00 s, drift 11°, and the quiet headstand keeps its
  two-level pencil shape (flag moves the leg, the show does not) — plus **+21.67 CCW turns @ mean
  13.62 rad/s**, hold 12.69 s, sit 0.34 s, dive 0.30 s, and the sweep SURVIVED (419° inside the
  show window, probe 777°/crouch); head 0.42/0.37/0.37 the only checkpoint under the hs12 baseline;
  arena 20.00 s played. Neighbours 1000 (+23.47, head 0.62 at 4.5 s) through 1750 (+22.34) and 2250
  (+19.43) pass; 2999 decayed (+13.66) — do not resume from it. **This is the resume base for
  hs12c** (stage byte-for-byte; body dims all alive).
- **`hs12/model_2250`**: the first checkpoint with the **downrock sweep**: probe MEAN
  **837.2° per crouch** (quiet crouch 0.8°, held 3.00/3.00 s), and on the full show timeline
  **464° inside the 2 s knee drop** — plus **+24.01 CCW turns @ mean 15.09 rad/s** (best 1 s 17.64),
  3/3 seeds, hold 12.68 s, **sit 0.50 s**, dive 0.26 s. Neighbours 2000 (837°, sit 0.56 s) and 2500
  (804°) pass; the same run's 2750/2999 decayed — do not resume from them. Was the live show until
  hs12b; the resume base hs12b grew from (byte-for-byte, md5-verified).
- **`hs11b/model_1250`**: **+24.25 CCW turns @ mean 15.24 rad/s** (best 1 s 17.75 rad/s), 3/3 seeds,
  continuous hold 12.68 s, knee-drop sit reached in **0.46 s**, dive onto the head
  in 0.28 s. The resume base hs12 grew from (see the Runs row; its resume copy was the lightly
  patched one, documented in hs12's provenance).
- **`hs11/model_1250`**: **+25.22 CCW turns @ mean 15.9 rad/s** (best 1 s 18.4 rad/s), 3/3 seeds,
  hold 12.68 s, knee-drop sit reached in **0.52 s** — the show before hs12. Same run's 1750 spins
  more (+25.88 @ 16.26, longest 12.37 s) but **never sits** — a spin
  specialist, not the show; kept in the run folder, not archived.
- **`hs10/model_1750`**: **+23.95 CCW turns @ mean 15.1 rad/s** (best 1 s 17.7 rad/s), 3/3 seeds,
  continuous hold 12.68 s, spindle penalty at its floor (neck quiet through spin, brake and freeze),
  knee-drop sit reached in 0.64 s on the stage. Neighbours 1500 (+23.34, sit 0.74 s) and 1250
  (+22.79, sit 0.68 s) pass too.
- **`hs9/model_1000`**: **+23.52 CCW turns @ 14.8 rad/s** (peak 18.0 rad/s / 172 RPM on standard test; +24.59 turns on full timeline), 18.16 s continuous hold (Seeds 2 & 3), Stage 3 unlocked (20 rad/s).
- **`hs9/model_750`**: +20.17 CCW turns @ 12.7 rad/s (peak 15.7 rad/s), 100% continuous hold, 150° leg drive.
- **`hs8/model_500`**: +13.86 CCW turns @ 8.7 rad/s, head roll stabilized.
- **`hs6/model_250`**: Solid upright headstand hold before spin discovery.

**The live choreography (`timeline.json`, verified 20.00 s on the Arena's stage, hs12b/model_2000)**:
- 0.0–2.0 s Toprock opening: upright stance, beat-bop head nods (±0.35 rad pitch) and sharp
  judge call-out glances (±0.7 rad head_yaw, 0.2 rad roll). Feet static — standing footwork failed
  twice (hs11, hs11b) and is parked; the body stays at zeros here.
- 2.0–4.0 s B-Boy Knee Drop **with the downrock sweep**: sit flag up AND `body[1]=1`, snaps from
  z 0.115 m down in ~0.35 s, then **sweeps ~419° inside the window** (probe 777° when given 3 s),
  crouch held, trunk z ~0.062 m. (hs12c's kneel-and-blade grinder FAILED by measurement and was
  reverted — this low sweep stays the live move, the accepted flat-platform cost included.)
- 4.0–4.5 s Inversion Dive: headstand flag up (sweep released), pops hips and lands on the crown in 0.30 s.
- 4.5–11.5 s Power Headspin: spin flag up, +21.7 CCW turns @ mean 13.6 rad/s (best 1 s
  15.8 rad/s ≈ 151 RPM).
- 11.5–18.0 s Deceleration: spin flag drops, legs flare, momentum brakes to a complete stop at
  18.0 s.
- 18.0–20.0 s **L-Kick Freeze**: `body[4]=1` — on the buzzer beat the duck pulls one leg back to
  the vertical needle (needle err 0.10 rad) and holds the other level, dead still on the head until
  the buzzer (kick_probe-verified 3/3 seeds, drift 11°). `check.py` segments `headstand`
  (`command[1] >= 0.5`) beside `sit` and `stand`.

## The next phase: the full b-boy set, training on MARS

The roadmap turns the headspin into a complete, authentic breakdance set inside 20 s:
**[1 Toprock] → [2 Downrock] → [3 Dynamic Entry] → [4 Power Headspin] → [5 Freeze & Exit]**,
one hypothesis per training sprint, every checkpoint judged on the Arena's stage.

**Where the code must run**: training continues on **MARS (RTX 4090 workstation, 24 GB)**:
use `--env.scene.num-envs 4096` (host RAM tight → 2048). Everything needed is committed on
`main`: code, `checkpoints/`, timeline. On MARS: clone `electricshaka/microduck-private`,
`uv sync --group workshop`, then `uv run --no-sync …` for every command. (The DGX Spark box this
was developed on runs a vLLM container holding ~95 GB VRAM: `cd /home/cheetor/GitHub/qwen3.8-Flash-DGX
&& ./flash stop` before GPU training there and `./flash start && ./flash wait` after. Not needed on MARS.)

### hs10 — Spindle Lock & pure leg-driven headspin (DONE — model_1750 committed, live)

Done: the bar was met (3/3 seeds ≥ +20 @ ≥ 12 rad/s, penalty at its floor, hold continuous) —
see the Runs table for the numbers. The `hs10b` fallback was not needed. Results below kept as the record.

*Hypothesis*: hs9 still pumps the spin by oscillating neck yaw ±31° against the floor — the
exp-shaped `head_stillness_on_head` grants nothing and costs nothing beyond its ~0.3 rad basin,
so big neck swings were free. A strict non-saturating quadratic penalty, `rewards.head_spindle_penalty`
(weight −1.0, paid only while the headstand flag is up) on the 4 head/neck joints (5–8) measuring
deviation from the spindle pose **[0.5, 1.0, 0.0, 0.0] rad** (absolute, = the spawn pose), will
kill the yaw whip, lock the head/neck spindle (good for the freeze too) and force 100 % leg-driven
rotation. The term is registered in `env.py`; tests cover its gate and its quadratic shape.

```bash
mkdir -p logs/rsl_rl/got_a_talent/hs9_base
cp checkpoints/got_a_talent/hs9/model_1000.pt logs/rsl_rl/got_a_talent/hs9_base/
PYTHONUTF8=1 MICRODUCK_WARM_START=1 uv run --no-sync train Mjlab-GotATalent-MicroDuck \
    --env.scene.num-envs 4096 --agent.max-iterations 3000 --agent.seed 1 \
    --agent.logger tensorboard --agent.run-name hs10 \
    --agent.resume True --agent.load-run hs9_base --agent.load-checkpoint model_1000.pt \
    > logs/hs10.log 2>&1 &
```

*Follow it*: `Episode_Reward/head_spindle_penalty` must read ≤ 0 and trend toward ~0 while
`Episode_Reward/headspin` holds hs9 levels; a collapse not recovered in ~500 iterations: stop
and tell the user. *Judge every checkpoint* (see Judging) with `headspin_check` (seeds 1-3),
`check.py` and `arena score` against this folder's `timeline.json`.
*Success bar*: **≥ +20 CCW turns @ ≥ 12 rad/s mean on 3/3 seeds**, continuous hold, and the
spindle penalty within ~0.1 of its floor (neck quiet through spin, brake and freeze).
*If the turns collapse below +12*: the penalty is too strict — rerun with weight −0.5
(run name `hs10b`, change nothing else). *Finish*: commit the best checkpoint to
`checkpoints/got_a_talent/hs10/` (`.pt`, `.onnx`, `judge_<N>.txt`, `provenance.json`), fill the
Runs table row, push both remotes.

### Backlog — the remaining skills (one commit + one run each, in order, after hs10)

**hs11 — Active Toprock footwork** (`body[0]` = toprock flag) — **DONE, footwork FAILED to emerge**
(see the Runs table: 0 of 12 checkpoints stepped; the full implementation — `ToprockCommand`,
per-landing `toprock_step`, sway, `unless_toprock` — is on `main` and in hs11's run). What the
stage probe `scratch/toprock_probe.py` (2.5 s stand, flag 0 then 1, seeds 1-3, counts landings by
`toprock_step`'s own rules) says is the acceptance gate for ANY footwork attempt: today it must
read **≥ 2 alternating steps with body[0]=1** where hs11 read 1 touchdown.

**hs11b — Dense footwork + mid-step spawn** — **DONE, footwork FAILED again → parked.** The dense
window + mid-step spawn (see the Runs table) grew the curve 40× but the stage probe still read
1 touchdown at all 12 checkpoints: the payout harvested the seeded landing, so producing a step
was never needed. Per the written fallback: `body[0]` stays untrained-used (flag exists, show keeps
it 0, timeline foot-static, guard test unchanged), **no more footwork runs until a new mechanism
idea**. The `toprock_probe` gate (≥ 2 alternating steps) still stands for any future attempt.

**hs12 — Downrock floorwork** (`body[1]` = sweep flag, scissor sweep) — **DONE, downrock LEARNED**
(run `hs12`, cwd the hs12 worktree, run folder junction-linked under main's `logs/`; resumed from
a lightly PATCHED copy of `hs11b/model_1250` — the pre-rewrite prewarm set body[1..5] normalizer
dims to mean 0.25/var 0.1875 and zeroed their first-layer columns, a whisper for these
alive-but-small dims; full hashes and notes in `checkpoints/got_a_talent/hs12/provenance.json`).
The implementation as planned: `rewards.downrock_sweep` (6.0, crouch band z 0.045–0.095, flag-gated,
frontier CCW trunk yaw, dense, back-and-forth nothing) + `downrock_leg` seasoning +
`spawn_seated_sweep` 35% + stillness pause under the flag + curriculum 2.0→3.5→5.0 rad/s.
The bar was **≥ 30° swept**; delivered: probe MEAN 837° at 2250 (see the Runs row), quiet-crouch
control ≤ 1.4° throughout, and `headspin`/sit/dive held hs11b levels the whole run. The
`body[1]=1` knee drop was stage-verified (464° inside the 2 s window) and is the committed live
timeline. What to carry to the next move: the frontier pay works on held states; `downrock_leg`
earned nothing (flat 0.004) — cut dead seasoning; the final curriculum stage decayed the run's
last two checkpoints again — treat 2500-stage checkpoints with suspicion.

**hs12b — Inverted "L-Kick" Freeze** (`body[4]` = L-kick flag) — **DONE, FREEZE LEARNED**
(model_2000 archived and live; see the Runs row for the numbers and the ladder; resumed
byte-for-byte from `hs12/model_2250` — md5 of the staged copy 7DF330EEC0AA1562B40B684DE1D4FFED,
`scratch/prewarm_hs12b.py` verified body[4] alive first: normalizer std 0.398, first-layer |w| ≤ 0.26).
Delivered: freeze kick 0.01 / needle 0.10 rad, up 3.00 s, drift 11°, quiet window keeps the pencil;
+21.67 turns @ 13.62, sit 0.34, dive 0.30, arena 20.00 s; sweep survived (show 419°, probe 777°).
The `body[4]=1` upgrade at 18.0 s is stage-verified and in the committed live timeline. What the
entry below predicted, kept as the record:

*Hypothesis*: the show's last 2 s are a dead-still headstand; a freeze POP — one leg staying the
needle, the other swinging level — makes the ending a move, not a wait. PPO's noise alone will
never visit a one-hip 1.9 rad swing from the needle headstand, however dense the pay; the
`spawn_l_kick` seed (30% of resets, IN the freeze, at rest) puts the policy there and
`l_kick_hold` (6.0) pays every step it holds — the hs12 lesson that a dense pay on a SEEDED hold
works, applied to a hold instead of a turn.

*Measured, not guessed* (`scratch/kick_env_probe.py`, live CPU env, duck verified inverted
projected-gravity +1.0): the HOME hips (∓0.458) put both feet **0.117 m ABOVE** the inverted
trunk — the ordinary headstand's legs are already two vertical needles; hips toward ±1.42 (soft
limit 1.414, hard 1.571) with the knee straight bring a foot **level** with the trunk (dz −0.012).
The two sides are NOT mirror-symmetric in the world: the right kick reaches 0.03 m FORWARD, the
left the same way swings 0.07 m BACK (a donkey-kick) — both read as a frozen leg off the vertical,
so `l_kick_hold` takes the min of the two per-side variants. Reward: exp(−pose err / 0.55²) × the
kick flag riding a commanded headstand × the head planted × trunk z ≥ 0.13; no competitor wrapper
(the rest stack is already off on the head, headspin isn't paying during the freeze). The dead
`downrock_leg` registration is RETIRED here, hs12's recorded lesson (function and tests stay).

*Baseline measured on hs12/model_2250 itself* (`scratch/kick_probe.py`, the acceptance probe): the
trained headstand already holds BOTH legs level — quiet-window kick err 0.14 rad, needle err
~1.7, up 2.17 s of 2.5; freeze window (flag up, untrained) 0.04 rad, drift 3°. So the flag's
visible signal is the NEEDLE recovery: the freeze pulls one leg back to the vertical while the
other stays level. **Success bar** (per checkpoint, seeds 1-3): freeze-window kick err ≤ 0.5 rad
AND needle err ≤ 0.5 rad, up ≥ 2.0 s of 3, drift ≤ 60°, quiet window keeps the baseline shape
(needle ≥ 1.2 rad) — plus `headspin_check`/`check.py` holding hs12 levels (+24 turns, sit ~0.5 s)
and the kick flag NOT raising the legs without the flag. *If the needle never returns* (< 0.8 err
stuck at 3000): the hold shape is too far from the trained two-level balance — re-scope the entry
honestly as a two-level pencil freeze vs a legs-tucked hold, or raise `spawn_l_kick` to 0.5 and
re-run once. *Finish*: same ritual — judge all 12 checkpoints, archive the winner to
`checkpoints/got_a_talent/hs12b/`, fill the Runs row, push, and ONLY after a stage-verified
checkpoint raise `body[4]=1` at 18.0 s in `timeline.json` (widening the guard test then, with the
measured verification cited).

**hs12c — Grinder sweep (the butt-spin ban)** (`body[1]` unchanged; **DONE — FAILED BY MEASUREMENT,
fallback (c), REVERTED** — implemented in fb19d43, ran 2026-10-24 resumed byte-for-byte from
`hs12b/model_2000`; all 12 checkpoints judged, records in `checkpoints/got_a_talent/hs12c/`; the
verdict and the numbers are in the Runs row above; the grinder terms' functions, sensor and tests
stay in the code UNREGISTERED — the measured record — and the live sweep is hs12b's shape again).

*Hypothesis* (the user's video review, confirmed by measurement): hs12's sweep pays its frontier
inside z < 0.095 with no shape requirement — so the cheapest earn is the pelvis FLAT at 0.061
spinning on the butt, which is what the video shows. If the frontier REQUIRES a kneel-height
regime (trunk z ≥ 0.085, weight on a planted knee/foot) and pays MULTIPLIED by blade shape
(swept hip ±1.2, knee 0), the duck must grind on its knee with the leg out — and the regime-
replacement pattern (pause the sit composite under the flag, fund the regime with its own terms)
is the one this robot has already learned twice (headstand, and hs12's crouch itself).

*Measured* (`scratch/sweep_env_probe.py` — kinematics from static forward(); contact flags read by
live physics in the terms): the blade foot (hip ±1.2, knee 0) sits at **radius 0.089 m** about the
trunk axis vs **0.067 m tucked**, and at z 0.085–0.095 it trails the floor by 0.2–1.2 cm — the
perimeter it should sweep. The flat crouch rests at 0.061 (hs12's measured platform); 0.085 is the
user's floor and a 2.4 cm lift from it. `pelvis_ground` sensor pattern (trunk_base|hip_l|hip_l_2)
verified against the model's body names.

*Acceptance* (`scratch/sweep_probe.py`, extended to report the blade foot's world radius, trunk z
and a pelvis-contact count for the crouch window; seeds 1-3): **≥ 400° per kneel-drop with foot
radius ≥ 0.08 m for most of the window and pelvis contacts ≈ 0** (the ban), plus `headspin_check`/
`check.py` holding hs12b levels (+21…+24 turns, sit ~0.5 s, dive ≤ 0.3 s) and the L-kick freeze
still answering `body[4]` (it rides the resumed winner — verify, don't assume). The live timeline's
knee-drop window needs NO flag change (body[1]=1 already); its z story line updates after stage
verification.
*Fallbacks*: (a) if degrees collapse (< ~300) while the shape holds — the kneel is unreachable
from the sit pose: relax `SWEEP_Z_MIN` to 0.08 once, nothing else; (b) if the pelvis-contact
penalty saturates while the frontier still pays — the −3.0 lost the sum against 6.0+2.0: raise to
−5.0, one rerun; (c) if neither: keep hs12b's sweep as the live show (777°/crouch on the probe,
419° inside the window), revert the grinder terms, and record the ban as failed-by-measurement.
*Finish*: same ritual — 12
checkpoints judged, winner archived, Runs row + timeline story updated, pushed.

**hs13 — Breakdance Overhaul** (`body[0]` = BOUNCE flag, `body[2]` = CROSS-STEP flag;
**IMPLEMENTED, RUNNING** — launched 2026-10-24 21:39 from `hs12b_base` with the warm start verified;
the user's direct order: the show's Phase-1 fix; hs12c fell to fallback (c), so the base is
`hs12b/model_2000` (staged byte-for-byte, md5 7566EB15257368DA231DBD7F8C6FB488); see the Runs row).

*Hypothesis* (the user's review, and two measurements): the opening "just looks side to top with
head" because the body is static — and it CANNOT be fixed from the timeline (`bounce_probe.py`:
the sit flag flipped every half-beat at 108 BPM moves the trunk 0.003 m, 0 of 42 beats reach
2 cm — the 2 s-ramped reference cannot oscillate at 1.8 Hz) nor by standing footwork (hs11/hs11b:
a stand earns the same before and after a step). Both walls fall to ONE new state: a rhythmic
knee BOUNCE the robot HOLDS and cycles, the way the crouch holds a sweep (the lesson that made
hs12 work) — `bounce_hold` (6.0) pays a dense window that only a VALID completed cycle opens,
and the cross-steps ride ON it (body[2], masked to the bounce flag itself): the steps the static
flag never bought are steps on a moving body this time.

*Measured* (`scratch/bounce_env_probe.py`, static forward() on the training model; contact and
velocity read live in the terms): symmetric knee flexion (left +/right −, the lift signs) with a
small mirrored hip bend keeps the trunk VERTICAL (projected gravity −1.00 in every pose tried)
and takes the feet off the floor — knees ±0.35 give a 1 cm trough at z 0.105, knees ±0.50 a
2.2 cm trough at z 0.093-0.095: a real cycle runs **0.115 → ~0.095 → 0.115, 15% of body height,
clear of the crouch floor (0.061) and the grinder floor (0.085)**. A cycle must therefore drop
≥ 0.012 m from its own top, HAVE REACHED the top (≥ 0.108 — a sag into a squat is not a bounce),
flex the knees (mean |knee−home| ≥ 0.20 rad — not a hip hinge), keep a foot planted (a knee
bounce, not a hop) and arrive ≥ 0.30 s after the last (108 BPM = 0.555 s; spam cannot earn).
The cross-step: from the trough, a hip 0.40-0.55 rad forward (left NEGATIVE/right POSITIVE, the
downrock signs) reaches the foot 2.3-3.2 cm forward and shuffles it ≤ 1.4 cm up, the other foot
planted — a peak is 0.45 rad past home, ALTERNATION only (hs11's rule), 0.25 s apart minimum.
The seed (`spawn_mid_bounce`, 30%, registered LAST): the measured trough pose (test-verified on
the model: planted foot at floor level, shown foot lifted) with one hip shown forward (the
sweep-seed lesson: SHOW the shape) and a small sink so the first cycle completes at once.
Counter-sway, honest: this robot has no torso joints — the sway the eye reads is the bounce, the
beat nods the head command already carries, and the steps' mirrored hip swing rocking the
pelvis; nothing pretends at a chest sway.

*Acceptance* (new `scratch/bounce_show_probe.py` once trained: 3 s standing window, body[0]=1,
then body[0]=1 AND body[2]=1, seeds 1-3; reads trunk-z cycles, drop depth, cycle rate, step
alternations; quiet-flag control window first): **≥ 3 counted bounce cycles ≥ 1.2 cm deep in the
3 s flagged window (control ≤ 0.5 cycles), and ≥ 2 alternating cross-step peaks with body[2]=1**
— plus `headspin_check`/`check.py` holding the hs12c winner's levels (turns, sit ~0.5 s, dive
≤ 0.3 s, sweep and freeze still answering). The 0-2 s timeline flags (body[0]=1, body[2]=1) come
ONLY after a stage-verified checkpoint, guard test widened then with the verification cited.
*Fallbacks*: (a) if cycles never start from standing (< 1 in the window) — the rise requirement
is too strict for a resumed stand: relax BOUNCE_Z_TOP to 0.104 once, nothing else; (b) if the
bounce works but steps never alternate — hs11's verdict for THIS state too: deliver the bounce
alone honestly (body[2] parked at 0, steps recorded failed-by-measurement — two mechanisms,
three tries) ; (c) if neither emerges in 3000 iters: the show's opening stays head-nods only and
the overhaul is recorded failed-by-measurement, as toprock was (the show keeps hs12c's live
show either way).
*Finish*: same ritual — 12 checkpoints judged, winner archived, Runs row + timeline opening
updated after stage verification, pushed.

**hs14 — Headstand Exit & upright freeze** (unlatch `twist[1]`) — was hs13; the overhaul took
the slot. Problem: the headstand flag is latched for the whole episode; the show ends upside
down. Implementation: in `HeadstandCommand._resample_command`, let the flag go back down (a
small probability per resample, and always down when the sit/stand draw says so); add recovery
rewards gated on "flag just went down": hip extension, trunk righting (the REST_STACK powers
back on), landing on feet/knees within ~1.5 s, then upright stillness for the final freeze pop
and b-boy bow. Trap to avoid (hit before): a "rise" that is really a lean back at 57° — require
the upright stack, not just the angle. Then the timeline drops `twist[1]` at 17.0 s (move the
brake start earlier so the spin stops by ~16.5 s) and the last 3 s are an upright freeze pop.
Judge: `check.py` must PASS with the upright finish, headstand hold unchanged while the flag is
up.

**hs15 — "Block Rockin' Beats" Breakbeat Cypher Troupe** (`timeline_version: 2`, no policy
training; `timeline.json` only; STAGING ONLY — no publishing without a new explicit yes) —
was hs14. Transforms the solo performance into a full 6-duck B-Boy Battle Cypher on
the 2.6 m × 3.8 m stage:
- **Spatial Cypher**: Star Breaker Duck (`policy: "self"`, `(0, 0, 0°)`) surrounded by 5 Crew Ducks
  in a circle ($r \approx 0.8$ m) facing inward, executing synchronized head nods (108 BPM,
  `neck_pitch = ±0.35` rad) to watch and hype the breaker.
- **"Block Rockin' Beats" by The Chemical Brothers (108 BPM Big Beat score)**:
  - *Duck 4 (808 Dirty Acid Bass)*: `voice: {pitch: 0.05, buzz: 0.95, brightness: 0.85}`. Sings the
    syncopated D-minor riff (D2 ➔ D2 ➔ F2 ➔ G2 ➔ G#2 ➔ A2) with vowel `"oo"`/`"eh"` on dynamic `"f"`.
  - *Duck 5 (Beakbox Snare)*: `voice: {pitch: 0.70, brightness: 0.98, buzz: 0.60}`. Sings sharp G4 `"ah"`
    notes on Beats 2 and 4 while pulling `triggers: [1.0, 0]` (mechanical beak snap producing a fat snare quack!).
  - *Duck 3 (Laser Siren FX)*: Chromatic rising sweep (D5 ➔ A5) with `vibrato: 0.8` pulling `triggers: [0, 1.0]`
    (screaming resonant wheee siren sweep) right at 3.5–4.5 s as the dancer dives into the headspin!
  - *Duck 2 (Vocal Hook)*: Chants the iconic hook *"Back with another one of those BLOCK ROCKIN' BEATS!"*
    (`"ah"`, `"ee"`, `"oh"`, `"ff"` on "BLOCK" / "BEATS").
  - The drop lands at $t = 2.0$ s as Duck 1 snaps into the 0.52 s Knee Drop, followed by the 172 RPM headspin!

### Repo duties for every agent (MARS included)

- Every commit authored `electricshaka <953027+electricshaka@users.noreply.github.com>`.
  **Item 3 of the user's commit ritual (2026-10-25): ZERO mentions of AI assistants, LLMs, bots,
  Claude, or Qwen in commit messages (subject AND body), in code comments, or in any prose an
  agent authors into the repo.** (Word-level: "robot" is a machine, not a hit.) Audit at this
  writing, word-boundary and case-insensitive across all of `main`: clean everywhere except the
  inherited upstream commit `c38a36f`, whose message names the file `CLAUDE.md` — upstream, not
  editable, left untouched. This AGENTS.md documents the rule, so the words appear here once;
  they must never appear anywhere else an agent writes.
- Scratch scripts are named `*_probe.py`, **never** `*_test.py`: bare `pytest` collects `*_test.py`
  under the repo root as well, and the probes insert training-checkout paths into `sys.path` at
  import — a bare run then resolved `microduck_challenges` from a stale worktree (7 phantom hs12
  failures against hs11b's code, while `pytest tests` stayed green). Keep probes out of collection.
- Never edit Pollen's upstream files (`README.md`, `CLAUDE.md`, `tests/test_challenges.py`,
  `pyproject.toml`, `uv.lock`, other challenges' folders); challenge logic lives in this folder
  plus `tests/test_got_a_talent.py`.
- `uv run --no-sync --group workshop pytest` before every launch: **92 green at this writing**
  (81 through hs12; +5 the hs12b kick tests; +2 the hs12c grinder tests; +4 the hs13 bounce
  tests — window, steps, seed marks, seed geometry — and −1 the retired toprock wiring test).
- **Continuously commit and push ALL progress to the private repo (user's rule, restated
  2026-10-25)**: `origin/main` (`electricshaka/microduck-private`) must never lag — every
  commit — launch, judge-record, archive, timeline change — is pushed the moment it is committed,
  never batched to the end of a sprint, and judge records (the per-checkpoint `judge_<N>` files)
  are committed as they are produced, not only at archive time. Publishing to the Arena still
  needs the explicit yes. (On the workshop box this folder had a second remote, the public fork
  `electricshaka/MicroDuck`, kept identical; on this checkout `origin` is the private repo and
  the only one.) The user dictated the exact ritual (2026-10-25, item 1):
  `git commit --author="electricshaka <953027+electricshaka@users.noreply.github.com>" -m "<concise technical summary>"`
  — this checkout's git config already resolves author AND committer to that identity bare
  (`git var` verified), and commit messages take a concise technical summary first (a needed
  detail goes after it, not instead of it). Item 2: **push immediately to `origin/main`** after
  every commit (`git push origin main`), same breath, never deferred. Item 4: **never leave local
  commits unpushed** — the invariant, checked any time the repo is touched:
  `git fetch origin; git rev-list --count origin/main..main` must print **0**; a non-zero is an
  emergency — push again before anything else commits, and never amend or force-move a commit
  that is already pushed. **User mandates (2026-10-25), logged:** (1) the private
  `electricshaka/microduck-private` receives continuous commits+pushes of ALL progress, with
  commit + push as the two-command ritual above and zero forbidden mentions; (2) the PUBLIC repo
  `electricshaka/MicroDuck` (local checkout `C:\GitHub\MicroDuck`, remote added here as
  `public`) is kept in sync — latest clean commits and checkpoints pushed to public `main` —
  and the Arena entry `ElectricShaka/quackstreet-boy-the-tailspin` (got-a-talent) is updated by
  publishing the best stage-verified checkpoint with the live timeline and entering
  (`--no-private --enter --code-url https://github.com/electricshaka/MicroDuck/tree/<public
  commit>`); the 2026-10-25 publish/enter is explicitly ordered; further Arena publications
  still wait for a fresh explicit yes.

### Open problems and next ideas

1. **Leg-driven spin under hs10**: confirm the legs alone can hold ≥ +20 turns; if not, spawn
   more episodes already spinning (they seed leg-driven states).
2. **Coming down** from the headstand on command — now scheduled as hs14 (the flag is latched
   in training).
3. **Show**: footwork parked after two failed runs (hs11, hs11b — do not retry without a new
   mechanism); the downrock sweep is LEARNED and live (hs12); the L-kick freeze is LEARNED and live
   at 18.0 s (hs12b, model_2000); the grinder re-shape of the sweep (hs12c) runs now; the bounce +
   cross-steps overhaul (hs13) is queued at the hs12c handoff — if it lands, its flags enter the
   0-2 s opening after stage verification (the standing footwork question is folded into it:
   steps on a moving body, the one route hs11's verdict did not cover); exit + bow
   (hs14); sound + cypher troupe (hs15, user's yes given for staging, not publishing).
4. **The rhythmic bounce the user asked for in the Toprock phase is NOT reachable by timeline**:
   measured (`scratch/bounce_probe.py`, sit flag flipped every half-beat at 108 BPM, seeds 1-3) the
   trunk moves **0.003 m** — 0 of 42 beats reach even 2 cm; the 2 s-ramped sit reference cannot
   oscillate at 1.8 Hz and the policy was only ever trained on dwells ≥ 2 s. The user's answer
   (2026-10-24): train it — the bounce became hs13's trained regime on `body[0]` (dense pay on
   VALID cycles, cycle clock kept by the COMMAND term, not the policy counting; the probe proved
   the command signal reaches the policy, it had only never learned to answer fast). If the
   trained regime also fails by measurement, the bounce is genuinely unreachable on this robot
   and the opening stays head-nods — record it and stop chasing it.

### Judging

```bash
run=logs/rsl_rl/got_a_talent/<run>; n=<checkpoint>
uv run --no-sync publish --run $run --checkpoint $n --repo local/microduck-eval --dry-run --allow-dirty
uv run --no-sync --group workshop python -m microduck_challenges.got_a_talent.headspin_check publish-eval/policy.onnx
uv run --no-sync --group workshop python -m microduck_challenges.got_a_talent.check publish-eval/policy.onnx
```

`headspin_check` plays stand 2 s → headstand flag → spin flag at 5 s, on seeds 1-3, and reports the
time onto the head, the longest hold, time on the head, turns and spin rates. A policy trained
without the spin flag falls when it rises: judge those with `--timeline` on a file without it.
`check.py` judges the sit/stand routine; it must still pass for a show that also sits and looks
around.

### Running on Windows (what this machine needed)

- The lock installs CPU-only torch on Windows. Training on the GPU here used torch 2.9.1+cu128
  installed into the venv (`uv pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128`,
  no cu129 Windows wheel exists), then every command with `uv run --no-sync` so uv does not put the
  CPU build back. `pyproject.toml` and `uv.lock` stay untouched.
- Resuming a run: `MICRODUCK_WARM_START=1` (else every curriculum jumps to its last stage) and
  `PYTHONUTF8=1` (the warm-start message crashes a cp1252 console), then
  `--agent.resume True --agent.load-run <run folder> --agent.load-checkpoint model_<N>.pt`.
- 2048 envs, not the recipe's 4096: 4096 ran out of host memory with other jobs running.

## The competition

- The Arena is the Hugging Face Space
  [pollen-robotics/microduck-arena](https://huggingface.co/spaces/pollen-robotics/microduck-arena).
  The board ranks performances by votes over the last seven days. Read it live:
  `curl -s https://pollen-robotics-microduck-arena.hf.space/api/events/got-a-talent/leaderboard`
  (each row: `user`, `policy_name`, `votes`, `shown`).
- Player points: +100 for a first publication in a challenge, +10 per publication (the first 3 per
  UTC day, all challenges together), +250/+150/+100 while holding first/second/third place (a
  voted entry needs an active vote). Receiving votes earns no personal-best bonus. Voters earn 2
  points per accepted vote, 20 votes per UTC day.
- The challenge's page on the Arena has the same rules in plain words, and the **Train my own robot**
  button that opens the workshop.

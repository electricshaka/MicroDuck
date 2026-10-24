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
- `body`: unused (zeros).

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
   is git-ignored): **`hs11/model_1250` is the live performance** (see Best checkpoints below),
   with `hs10/model_1750`, `hs9/model_1000`, `hs8/model_500`, `hs6/model_250`, `hs5/model_250`,
   `hs2/model_1250` behind it. Each folder has the `.pt` (to resume training), the exported `.onnx`
   (to judge, watch or publish), its `judge_<N>.txt` and `provenance.json`.
3. **Watch it**: `uv run --no-sync --group workshop python -m microduck_challenges.got_a_talent.headspin_view checkpoints/got_a_talent/hs11/model_1250.onnx --timeline src/microduck_challenges/got_a_talent/timeline.json`
4. **Judge it**: `headspin_check` and `check.py` on the `.onnx` (see Judging below).
5. **Resume training** (hs11 is done — the next step is **hs11b**, dense footwork + mid-step spawn,
   the backlog entry below). Both hs11 and hs11b resume from the **prewarmed** hs10 checkpoint —
   body[0] was dead in every hs10-era checkpoint (normalizer std 0, first-layer weights random),
   so rebuild that base once before resuming (`scratch/prewarm_hs11.py` patches a copy of
   `hs10/model_1750.pt` into `logs/rsl_rl/got_a_talent/hs10_base/model_1750.pt`):

   ```bash
   uv run --no-sync python scratch/prewarm_hs11.py
   PYTHONUTF8=1 MICRODUCK_WARM_START=1 uv run --no-sync train Mjlab-GotATalent-MicroDuck --env.scene.num-envs 4096 \
       --agent.max-iterations 3000 --agent.seed 1 --agent.logger tensorboard --agent.run-name hs11b \
       --agent.resume True --agent.load-run hs10_base --agent.load-checkpoint model_1750.pt > logs/hs11b.log 2>&1 &
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
| hs1 | 2ff472e | headstand flag, rewards, spawns | Lay upside down rocking head/trunk at z 0.04 m; hold 0.2 s | Shaping that only checks "inverted" pays lying down: gate it on trunk height |
| hs2 | 89b9cba | shaping needs the trunk off the floor; -2 body contact | Tripod at 250, one foot at 500; **checkpoint 1000-1250: onto the head in 0.40 s, held 12.6 s, 3/3 seeds** | Works. But sit/stand FAIL from 250: neck folded to its limit |
| hs3 | 610abc0 | HEADSTAND_PROB 0.35 → 0.15 | Neck still pinned; target asked -9.5 rad | Not a share problem |
| hs4 | 0e9f700 | penalise targets past joint limits | Target back to -1.7 rad, but stood as a tripod (feet + head down), 36° lean | A target past a limit gives no gradient; fix that first |
| hs5 | 80e55b4 | -2 head on the floor outside a headstand | Checkpoint 250: head error 1.92 → 0.57 rad, stands 3/3, headstand 12.5 s | The tripod was the trap; still above check.py's 0.25 rad |
| hs6 | b368ac7 | spin flag + `headspin` reward | Paused at iteration ~330. Checkpoint 250: headstand holds 12.5 s with spin flag up, but no spin: 0.01 turns; `Episode_Reward/headspin` 0.007 | Spin not discovered: a slow first turn (1 rad/s) earns 0.4/step against hold's 8 |
| hs7 | d715e29 | "quarter turns first" curriculum (`rate_max` 3.0 → 15.0 rad/s) | **Checkpoint 1000: +12.60 CCW turns @ 7.9 rad/s**, 100% continuous hold on head (12.65 s) | Broke reward trap: early turns pay 2.0/step instead of 0.4. But policy whips neck against floor |
| hs8 | 4a66303 | `head_stillness_on_head` quiet-pivot regularizer (weight 3.0) | **Checkpoint 500: +13.86 CCW turns @ 8.7 rad/s**, head roll wobble -95%, hold reward peak 2.78 | Stabilized head roll to 0.012 rad; neck yaw still oscillated ±31° because Gaussian tails vanish |
| hs9 | 09d00fb | scale to 8192 envs on DGX Spark (GB10), `rate_max` 10 → 20 rad/s | **Checkpoint 1000: +23.52 CCW turns @ 14.8 rad/s** (peak 18.0 rad/s), +24.59 turns on full timeline, 18.16 s continuous hold | Unlocked Stage 3 (20 rad/s). Massive power stroke, unbroken hold across seeds |
| hs10 | 7d12e27 | `head_spindle_penalty`: −1.0 quadratic on joints 5-8 off [0.5, 1.0, 0, 0] rad while the flag is up, to kill the ±31° neck-yaw whip and force leg-driven spin | **Checkpoint 1750: +23.95 CCW turns @ mean 15.1 rad/s, 3/3 seeds, hold 12.68 s, sit 0.64 s** — spindle penalty −0.07 at its floor: the neck is quiet and the legs carry the spin | Spindle lock holds turns AND kills the whip (penalty ~0 vs hs9's ±31° oscillation); but the deterministic-stage **sit flickers checkpoint to checkpoint** (250, 500, 1000, 2000, 2500, 2750, 2999 sat late/never; 750, 1250, 1500, 1750, 2250 sat in 0.6–1.2 s): judge the sit line per checkpoint, and pick one whose neighbours pass too |
| hs11 | bceeeb1 | toprock on `body[0]`: standing-only 0/1 flag, `toprock_step` paying each alternating landing (2 Hz cap), hip_roll sway, stillness paused under the flag; dead body dims prewarmed in the normalizer (mean 0.25 / var 0.1875, first-layer cols zeroed — `scratch/prewarm_hs11.py`) | **Footwork never discovered: 0 of 12 checkpoints ever stepped** on stage (probe: 1 touchdown, hip_roll ≤ 0.02 rad, flag up or down; `toprock_step` flat 0.004–0.008 for 3000 iters). Spin meanwhile hit record: **1750 +25.88 @ 16.26** (never sat), and **2999 collapsed (+0.01 turns)** | A reward that pays ONE frame per landing amortizes a full step to ~0.004/step — under the posture cost of lifting a foot, so no gradient; the same trap hs6's spin had, which only a DENSE pay fixed. And the final checkpoint of a run can collapse: never archive the last one without judging it |
| hs11b | 985dfd2 | dense rhythm window (0.75 s) + mid-step spawn (prob 0.25) to unlock toprock footwork | **Checkpoint 1250: +24.25 CCW turns @ 15.24 rad/s (170 RPM)**, 100% continuous hold 12.68 s, record knee drop sit in **0.46 s** (fastest ever). Checkpoint 1500 hit **18.17 rad/s (173.5 RPM)** peak. Step reward grew 40x to 0.162. | Dense window boosted step learning gradient 40x (0.004 → 0.162). Model 1250 is the fastest sit transition on record (0.46 s) while holding +24.25 turns. |

Best checkpoints committed in `checkpoints/got_a_talent/`:
- **`hs11b/model_1250`**: **+24.25 CCW turns @ mean 15.24 rad/s** (best 1 s 17.75 rad/s / 170 RPM), 3/3 seeds, continuous hold 12.68 s, knee-drop sit reached in **0.46 s** (fastest sit transition on project record), headstand dive in 0.28 s.
- **`hs11/model_1250`** (live): **+25.22 CCW turns @ mean 15.9 rad/s** (best 1 s 18.4 rad/s), 3/3 seeds,
  hold 12.68 s, knee-drop sit reached in **0.52 s** (fastest sit yet) — beats hs10/model_1750 on every
  line. Same run's 1750 spins more (+25.88 @ 16.26, longest 12.37 s) but **never sits** — a spin
  specialist, not the show; kept in the run folder, not archived.
- **`hs10/model_1750`**: **+23.95 CCW turns @ mean 15.1 rad/s** (best 1 s 17.7 rad/s), 3/3 seeds,
  continuous hold 12.68 s, spindle penalty at its floor (neck quiet through spin, brake and freeze),
  knee-drop sit reached in 0.64 s on the stage. Neighbours 1500 (+23.34, sit 0.74 s) and 1250
  (+22.79, sit 0.68 s) pass too.
- **`hs9/model_1000`**: **+23.52 CCW turns @ 14.8 rad/s** (peak 18.0 rad/s / 172 RPM on standard test; +24.59 turns on full timeline), 18.16 s continuous hold (Seeds 2 & 3), Stage 3 unlocked (20 rad/s).
- **`hs9/model_750`**: +20.17 CCW turns @ 12.7 rad/s (peak 15.7 rad/s), 100% continuous hold, 150° leg drive.
- **`hs8/model_500`**: +13.86 CCW turns @ 8.7 rad/s, head roll stabilized.
- **`hs6/model_250`**: Solid upright headstand hold before spin discovery.

**The live choreography (`timeline.json`, verified 20.00 s on the Arena's stage, hs9/model_1000)**:
- 0.0–2.0 s Toprock opening: upright stance, beat-bop head nods (±0.35 rad pitch) and sharp
  judge call-out glances (±0.7 rad head_yaw, 0.2 rad roll). Feet static — footwork is hs11b.
- 2.0–4.0 s B-Boy Knee Drop: sit flag up, snaps from z 0.115 m to a low seated crouch z 0.060 m,
  holds (downrock footwork inside it is hs12).
- 4.0–4.5 s Inversion Dive: headstand flag up, pops hips and lands on the crown in ~0.3 s.
- 4.5–11.5 s Power Headspin: spin flag up, +21 to +24.6 CCW turns @ mean 14.8 rad/s (peak
  18.0 rad/s ≈ 172 RPM).
- 11.5–18.0 s Deceleration: spin flag drops, legs flare, momentum brakes to a complete stop at
  18.0 s.
- 18.0–20.0 s Dead-Still Headstand Freeze: motionless on the head (z 0.1433 m, drift < 0.06 mm,
  head-only contact) until the buzzer. `check.py` segments `headstand` (`command[1] >= 0.5`)
  beside `sit` and `stand`.

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

**hs11b — Dense footwork + mid-step spawn** (in progress). *Hypothesis*: hs11 died of the hs6
disease twice over — a landing pays ONE frame (a whole step amortized to ~0.004/step, under the
posture it costs to lift a foot), and the alternating state is never visited from a planted stand.
So, deliberately two changes at one root cause: (1) `toprock_step` pays **DENSE**: every valid
alternating step (new contact by the foot not last down, ≥ `period_s` apart, and not made after
being airborne ≥ 3 frames — hops are not steps) opens a `hold_s` = 0.75 s window paid on EVERY
step (weight 2.0); only keeping the rhythm keeps it alive, the `headstand_hold` shape. (2)
`spawn_toprock` (events, after `spawn_headstand`, prob 0.25, skips head-down spawns) starts
standing episodes already **MID-STEP**: one foot raised by the measured lift pose (`TOPROCK_LIFT_*`,
signs probed on the model: left leg positive, right mirrored), +0.015 m, 0.05 rad lean, still;
it marks `ToprockCommand.spawned` and `reset` raises the flag from the mark (the `spawn_headstand`
pattern). Verified on a live CPU env: the seeded landing pays within 4 steps of the first reset
(`scratch/spawn_env_test.py`). Resume from the same prewarmed `hs10_base` (see Start here).
*Success bar*: some checkpoint reads ≥ 2 steps / ≥ 45 s upright on the probe, and `toprock_step`
grows past ~0.1 in training. *If dense + spawn still never steps*: footwork on this robot is not
reward-shapeable — park `body[0]` (leave the timeline foot-static, the guard test stays), no more
footwork runs until a new mechanism idea.

**hs12 — Downrock floorwork** (`body[1]` = sweep flag, coffee-grinder / scissor sweep). Problem:
the knee-drop crouch (2–4 s) sits static before the dive. Implementation: a second `body` slot,
sampled 1 only with the sit flag; in the seated crouch reward one leg sweeping forward
(hip_pitch ≈ −1.2 rad, knee ≈ 0) and arcing across the floor — frontier progress on the swept
foot's yaw angle about the trunk (again: back-and-forth pays nothing), gated on staying seated
(no head contact, crouch height), ending in the dive tuck. Hip yaw range is limited (±22° side,
±90° front/back): expect a scissor sweep, not a full grinder. Judge on stage: visible leg arc
during the sit, sit hold and dive timing unchanged.

**hs12b — Inverted "L-Kick" Freeze** (`body[4]` = L-kick flag). B-boy freeze variation held during
the headstand. Implementation: one leg vertical (needle), the other extended horizontally forward
at 90° (`hip_pitch ≈ -1.57` rad), held with high stillness and balance weighting before the buzzer.

**hs13 — Headstand Exit & upright freeze** (unlatch `twist[1]`). Problem: the headstand flag is
latched for the whole episode; the show ends upside down. Implementation: in
`HeadstandCommand._resample_command`, let the flag go back down (a small probability per
resample, and always down when the sit/stand draw says so); add recovery rewards gated on
"flag just went down": hip extension, trunk righting (the REST_STACK powers back on), landing on
feet/knees within ~1.5 s, then upright stillness for the final freeze pop and b-boy bow. Trap to
avoid (hit before): a "rise" that is really a lean back at 57° — require the upright stack, not just
the angle. Then the timeline drops `twist[1]` at 17.0 s (move the brake start earlier so the spin
stops by ~16.5 s) and the last 3 s are an upright freeze pop. Judge: `check.py` must PASS with the
upright finish, headstand hold unchanged while the flag is up.

**hs14 — "Block Rockin' Beats" Breakbeat Cypher Troupe** (`timeline_version: 2`, no policy training;
`timeline.json` only). Transforms the solo performance into a full 6-duck B-Boy Battle Cypher on
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

- Every commit authored `electricshaka <953027+electricshaka@users.noreply.github.com>`; never
  mention AI assistants in messages or comments.
- Never edit Pollen's upstream files (`README.md`, `CLAUDE.md`, `tests/test_challenges.py`,
  `pyproject.toml`, `uv.lock`, other challenges' folders); challenge logic lives in this folder
  plus `tests/test_got_a_talent.py`.
- `uv run --no-sync --group workshop pytest` before every launch: **76 green at this writing**.
- Never publish, enter the Arena, or launch HF jobs without the user's explicit yes. Pushing
  `main` to the user's two repos is part of the agreed workflow:
  `git push private main && git push origin main` (`private` = `electricshaka/microduck-private`,
  `origin` = the public fork `electricshaka/MicroDuck`; keep timeline + checkpoints identical).

### Open problems and next ideas

1. **Leg-driven spin under hs10**: confirm the legs alone can hold ≥ +20 turns; if not, spawn
   more episodes already spinning (they seed leg-driven states).
2. **Coming down** from the headstand on command — now scheduled as hs13 (the flag is latched
   in training).
3. **Show**: toprock footwork (hs11b), downrock (hs12), exit + bow (hs13), sound (hs14, user's yes).

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

# Breakdance Performance & "Block Rock'n Beaks" Cypher Plan

The reference for the **MicroDuck Got a Talent** breakdance performance: what the robot can do,
what the live show does, what each training sprint tried and what it measured. The authoritative
per-run record (hypothesis, commit, every judged checkpoint, lesson) is the Runs table in
[`src/microduck_challenges/got_a_talent/AGENTS.md`](../src/microduck_challenges/got_a_talent/AGENTS.md);
this document is the short map. Numbers below are stage measurements of the live checkpoint,
`checkpoints/got_a_talent/spin1/model_2750`, unless a sprint is named.

Status at 2026-11-02: live show `spin1/model_2750` (archived 2026-10-26); sanctioned resume base
`hs12b/model_2000`; sprint `hs15` (sweep timing) running, checkpoints 250 and 500 judged.

---

## 1. Physical Morphology & Kinematic Constraints

MicroDuck has **14 actuated joints** (10 in the legs, 4 in the neck/head) and **no arms**:

| Joint | Motion | Range (rad) | Range (deg) | B-Boy use |
|---|---|---|---|---|
| `left/right_hip_pitch` | Front/back swing | `[-1.57, 1.57]` | ±90° | Seated scissor sweep, L-kick split, leg-driven spin |
| `left/right_hip_roll` | Abduction / sway | `[-0.38, 0.38]` | ±22° | Braking flare (no side split possible) |
| `left/right_hip_yaw` | Inward/outward twist | `[-0.44, 0.52]` | −25° → +30° | Circular floor sweep |
| `left/right_knee` | Flexion / extension | `[-1.57, 1.57]` | ±90° | Knee drop, floor tuck, leg extension |
| `left/right_ankle` | Ankle pitch | `[-1.57, 1.57]` | ±90° | Floor contact balance, standing thrust |
| `neck_pitch` | Lower neck pitch | `[-1.57, 1.05]` | −90° → +60° | Spindle alignment, beat-bop nods |
| `head_pitch` | Upper head pitch | `[-1.57, 1.57]` | ±90° | Crown placement, judge call-out glances |
| `head_yaw` | Head turn | `[-2.97, 2.97]` | ±170° | Call-out whips (quiet during the spin: spindle lock) |
| `head_roll` | Head tilt | `[-0.44, 0.44]` | ±25° | Stylistic head roll |

Measured in the Arena's physics: the floor grips (friction 1.0, no torsional friction per contact
point); a free headspin slows from 7.9 to 3.7 rad/s in 0.2 s, so the spin is **driven**, by the
legs twisting the body against the head's grip; the headstand pose is neck_pitch +0.5,
head_pitch +1.0 rad with the trunk at z 0.15 m; the stage labels a headstand a fall but plays on.

---

## 2. The Live Routine (`timeline.json`, 20.00 s played on the Arena's stage)

```
0.0s          2.0s              4.0s   4.5s                      11.5s             18.0s    20.0s
 ├─ TOPROCK ──┼─ KNEE DROP+SWEEP ┼ DIVE ─┼───── POWER HEADSPIN ─────┼── BRAKE/FLARE ──┼ FREEZE ┤
```

| Phase | Window | Commands | What the stage measures (spin1/model_2750) |
|---|---|---|---|
| 1. Toprock opening | 0.0–2.0 s | `twist` 0, head nods ±0.35 rad pitch, ±0.7 rad yaw glances, 0.2 rad roll | Upright, feet static. Standing footwork and the knee bounce were trained and **failed by measurement** (hs11, hs11b, hs13, hs13b); the body stays at zeros here |
| 2. Knee drop + downrock sweep | 2.0–4.0 s | sit flag `twist[0]=1`, sweep flag `body[1]=1` | Sits in 0.44 s, trunk z ≈ 0.062 m, sweeps **477° inside the window** (probe 944° per crouch when given 3 s) |
| 3. Inversion dive | 4.0–4.5 s | headstand flag `twist[1]=1` | Onto the crown in 0.36 s |
| 4. Power headspin | 4.5–11.5 s | spin flag `twist[2]=1` | **+23.57 CCW turns @ mean 14.81 rad/s** (best 1 s 18.19 rad/s ≈ 174 RPM), continuous hold 12.66–12.68 s, 3/3 seeds |
| 5. Brake and flare | 11.5–18.0 s | spin flag drops | Legs flare, spin bleeds to a stop; head drift 16° |
| 6. L-kick freeze | 18.0–20.0 s | kick flag `body[4]=1` | One leg to the vertical needle (needle error 0.09 rad), the other level, kick error 0.03 rad, up 3.00 s to the buzzer |

Honest disclosures recorded with the archive: seed 2 spins +22.60 turns, under the per-seed +23
bar (the mean clears it); the head-tracking lines (0.68/0.49/0.47 rad) sit in the structural fold
band, and `hs12b/model_2000` keeps the head record (0.42/0.37/0.37); the duck ends the show on its
head, because the headstand exit (hs14) failed by measurement.

Command slots: `twist` = [sit, headstand, spin]; `head` = 4 deltas in rad; `body` =
[reserved (bounce, retired), downrock sweep, reserved (cross-step, retired), 0, L-kick freeze, 0].

---

## 3. "Block Rock'n Beaks" Cypher Troupe (staged, not published)

The Arena's `got-a-talent` supports **up to 8 ducks** on the 2.6 m × 3.8 m stage with
`timeline_version: 2`. The user built and verified a 6-duck cypher at 108 BPM (title locked:
"Block Rock'n Beaks"; the Chemical Brothers track it riffs on stays "Block Rockin' Beats" where the
score is cited):

```
                          [Audience / Judges]
                                   ▲
         Duck 2: Vocal hook                    Duck 3: Laser-siren FX
      (-0.3, -0.8, 60°)                        (-0.3, 0.8, -60°)
                \                             /
                 ►   ⭐ DUCK 1: B-BOY STAR   ◄     policy "self", (0, 0, 0°)
                /                             \
       Duck 4: 808 bass                     Duck 5: Beakbox snare
       (0.6, -0.6, 135°)                    (0.6, 0.6, -135°)
```

- Duck 4 bass: `voice {pitch 0.05, buzz 0.95, brightness 0.85, breath 0.0}`, 16th-note D-minor
  groove (D2 D2 F2 G2 G#2 A2), dynamic `f`, vowels `oo`/`eh`.
- Duck 5 snare: `voice {pitch 0.70, brightness 0.98, buzz 0.60}`, G4 on beats 2 and 4 with
  `RT = 1.0` (beak-snap quack).
- Duck 3 FX: D5 → A5 glissando, `vibrato 0.8`, 3.5–4.5 s, `LT = 1.0` (wheee siren).
- Duck 2 hook: syllabic chant on `ah`/`ee`/`oh`, dynamic `ff`.
- Crew ducks nod at 108 BPM (`neck_pitch`/`head_pitch` ±0.35 rad) on the kick/snare.

Committed tooling: `scratch/synth_block_rockin_beats.py` (the score),
`scratch/render_cypher_audio_showcase.py` (the 720p render) and `scratch/preview_player.html`
(the preview). The rendered MP4, the WAV mix and the troupe `timeline_version: 2` JSON were staged
outputs that stayed untracked: regenerate them from those scripts. **Published on the Arena is
still the solo entry**; the troupe goes up only on a fresh explicit yes, with the troupe JSON passed.

---

## 4. Sprint Ledger

One commit and one run per sprint, every checkpoint judged on the stage. Lessons are in the Runs
table of the guide; this is the outcome map.

| Sprint | Move | Mechanism | Outcome |
|---|---|---|---|
| hs1–hs9 | Headstand, then headspin | Flag-gated hold reward, "quarter turns first" curriculum, quiet-pivot regularizer, 8192 envs | **Learned**: `hs9/model_1000` +23.52 turns @ 14.8 rad/s |
| hs10 | Leg-driven spin | Spindle-lock penalty on the neck joints | **Learned**: `hs10/model_1750` +23.95 turns, neck quiet |
| hs11, hs11b | Toprock footwork (`body[0]`) | Per-landing pay, then dense rhythm window + mid-step spawn | **Failed by measurement**: 0 steps at all 24 checkpoints |
| hs12 | Downrock sweep (`body[1]`) | Frontier pay on crouch yaw + seeded mid-sweep start | **Learned**: `hs12/model_2250`, 837° per crouch on the probe |
| hs12b | L-kick freeze (`body[4]`) | Dense pay on the measured leg pose + seeded freeze | **Learned**: `hs12b/model_2000` (resume base) |
| hs12c | Grinder reshape of the sweep | Kneel gate + blade-shape multiplier + pelvis ban | **Failed**: gate unreachable, sweep starved; reverted |
| hs13, hs13b | Knee bounce + cross-steps (`body[0]`, `body[2]`) | Dense pay on valid cycles + seeded trough | **Failed**: 0 cycles at all 24 checkpoints; retired |
| hs14 | Headstand exit | Keep-probability unlatch, no new reward | **Failed**: 0/3 rises, and the unlatch starved the spin; re-latched |
| spin1 | Un-tithe the headspin | `headspin` frontier pay 6.0 → 8.0 | **Learned, live**: `spin1/model_2750` +23.57 @ 14.81 |
| hs15 | Faster sweep into the window | Same `sweep_rate_max` values at earlier stages (500/1000) | Running; 250 and 500 judged (sweep 598°/416°, spin +20.4/+20.5) |

Two laws the ledger established: a move learns when it starts from a state the robot already holds
(sweep, freeze, spin) and never when it must beat quiet standing every beat (footwork, bounce); and
the curriculum's **2500-iteration stage decays the show** in six consecutive runs, so the late
checkpoints are judged hardest and winners come from ≤ 2250 whenever a resume base is chosen.

---

## 5. How to judge, resume and publish

- Judge a checkpoint: `check.py`, `headspin_check.py`, `scratch/sweep_probe.py`, `scratch/kick_probe.py`
  (commands in the guide's Judging section); judge the exported deterministic ONNX, never the curves.
- Resume: `MICRODUCK_WARM_START=1`, from `hs12b_base/model_2000` byte-for-byte
  (`train_breakdance_24h.sh` wraps the launch).
- Publish: on the user's explicit yes only, with `--timeline src/microduck_challenges/got_a_talent/timeline.json`.

# Breakdance Performance & "Block Rockin' Beats" Cypher Plan

This document details the complete research, technical architecture, and implementation roadmap
for the **MicroDuck Got a Talent** breakdance performance on the Arena stage. It serves as the
single master reference for subsequent development sprints and future agents.

---

## 1. Physical Morphology & Kinematic Constraints

MicroDuck features **14 actuated joints** (10-DOF legs, 4-DOF neck/head) and **zero arms**:

| Joint | Motion | Range (rad) | Range (deg) | B-Boy Application |
|---|---|---|---|---|
| `left/right_hip_pitch` | Front/Back Swing | `[-1.57, 1.57]` | $\pm 90^\circ$ | Step kicks, seated scissor sweep, L-kick split |
| `left/right_hip_roll` | Abduction / Sway | `[-0.38, 0.38]` | $\pm 22^\circ$ | Toprock lateral hip sway, braking inertia flare |
| `left/right_hip_yaw` | Inward/Outward Twist | `[-0.44, 0.52]` | $-25^\circ \to +30^\circ$ | Indian cross-step, circular floor sweep |
| `left/right_knee` | Flexion / Extension | `[-1.57, 1.57]` | $\pm 90^\circ$ | 0.52s knee drop, floor tuck, leg extension |
| `left/right_ankle` | Ankle Pitch | `[-1.57, 1.57]` | $\pm 90^\circ$ | Floor contact balance, standing thrust |
| `neck_pitch` | Lower Neck Pitch | `[-1.57, 1.05]` | $-90^\circ \to +60^\circ$ | Spindle alignment, nodding beat-bop |
| `head_pitch` | Upper Head Pitch | `[-1.57, 1.57]` | $\pm 90^\circ$ | Crown placement, judge call-out glances |
| `head_yaw` | Head Turn (Azimuth) | `[-2.97, 2.97]` | $\pm 170^\circ$ | Call-out whips, crowd engagement |
| `head_roll` | Head Tilt | `[-0.44, 0.44]` | $\pm 25^\circ$ | Stylistic head-bobbing roll |

---

## 2. Master Breakdance Routine (The 6 Phases)

The live 20-second performance inside [`src/microduck_challenges/got_a_talent/timeline.json`](file:///C:/GitHub/microduck-private/src/microduck_challenges/got_a_talent/timeline.json):

```
0.0s             2.0s             4.0s        4.5s                     11.5s            17.0s    20.0s
 ├── [1. TOPROCK] ┼── [2. DOWNROCK] ┼─ [3. DIVE] ┼── [4. POWER HEADSPIN] ──┼─ [5. BRAKING] ─┼─ [6. FREEZE] ┤
```

### Phase 1: Toprock Opening (`0.0 – 2.0 s`)
- **Move**: Indian step cross-stepping (`body[0]`), beat-bop head nods ($\pm 0.35$ rad pitch), and sharp judge call-outs ($\pm 0.70$ rad yaw).
- **Sprint**: `hs11b` (active dense rhythm window + mid-step spawns).

### Phase 2: Downrock Floorwork (`2.0 – 4.0 s`)
- **Move**: Instant **B-Boy Knee Drop** dropping from $z = 0.115$ m to seated crouch $z = 0.060$ m in 0.52 s (`twist[0]=1`), followed by the **Coffee Grinder / Scissor Sweep** (`body[1]`).
- **Sprint**: `hs12` (frontier yaw progression on swept foot).

### Phase 3: Inversion Dive (`4.0 – 4.5 s`)
- **Move**: Rapid hip pop and diving 6 cm onto the crown with the spindle locked into alignment in 0.30 s.
- **Sprint**: `hs10` (locked quadratic spindle penalty on joints 5–8 off $[0.5, 1.0, 0, 0]$ rad).

### Phase 4: Power Headspin (`4.5 – 11.5 s`)
- **Move**: Continuous high-speed headspin (+25.2 CCW turns at **172 RPM / 15.9 rad/s mean**), 100% leg-driven torque.
- **Sprint**: `hs11` (live project peak record).

### Phase 5: Leg-Flare Braking (`11.5 – 17.0 s`)
- **Move**: Spin flag drops (`twist[2] = 0`); legs flare outward to maximize rotational inertia ($I_{zz}$) while hip counter-torque bleeds spin to exactly 0.0 rad/s.

### Phase 6: Freeze Finish (`17.0 – 20.0 s`)
- **Option A (Inverted L-Kick / Spindle Freeze)**: Asymmetric leg split ($90^\circ$ forward leg + vertical needle leg) held motionless ($< 0.06$ mm drift) until buzzer.
- **Option B (Snap Exit to Standing Pop)**: Unlatch `twist[1] = 0` at 17.0 s, pop hips, land on feet/shins, hit an upright freeze bow (`hs13`).

---

## 3. "Block Rockin' Beats" Cypher Troupe Specification (`hs14`)

Arena's `got-a-talent` supports **up to 8 ducks (`max_ducks: 8`)** on a 2.6 m × 3.8 m stage using `timeline_version: 2`.

### Cypher Geometry (The Battle Circle)
```
                          [Audience / Judges]
                                   ▲
                                   │
         Duck 2: Hype 1                       Duck 3: Hype 2
      Start: (-0.3, -0.8, 60°)             Start: (-0.3, 0.8, -60°)
                \                             /
                 \                           /
                  ►   ⭐ DUCK 1: B-BOY STAR   ◄
                      (Center Breaker)
                      Start: (0.0, 0.0, 0°)
                 /                           \
                /                             \
       Duck 4: Sub-Bass Kick                 Duck 5: Beakbox Snare
       Start: (0.6, -0.6, 135°)              Start: (0.6, 0.6, -135°)
```

### Multi-Duck Audio Arrangement (108 BPM Big Beat)
- **Duck 4 (808 Dirty Acid Bass)**:
  - `voice: {pitch: 0.05, buzz: 0.95, brightness: 0.85, breath: 0.0}`
  - Riff: 16th-note syncopated D-minor groove (**D2 ➔ D2 ➔ F2 ➔ G2 ➔ G#2 ➔ A2**), dynamic `"f"`, vowel `"oo"`/`"eh"`.
- **Duck 5 (Beakbox Snare)**:
  - `voice: {pitch: 0.70, brightness: 0.98, buzz: 0.60}`
  - Notes: Punchy **G4** on Beats 2 and 4, vowel `"ah"`.
  - Trigger: Pulls `RT = 1.0` (physical beak snap producing a cracking mechanical snare quack).
- **Duck 3 (Laser Siren FX)**:
  - Glissando sweep (**D5 ➔ A5**) with `vibrato: 0.8` at 3.5–4.5 s, pulling `LT = 1.0` (high-resonance screaming "wheee" siren).
- **Duck 2 (Vocal Hook)**:
  - Syllabic chanting: *"Back with another one of those BLOCK ROCKIN' BEATS!"* (`"ah"`, `"ee"`, `"oh"`, `"ff"` dynamic).
- **Crew Ducks 2–6 Choreography**:
  - Head nodding at 108 BPM (`neck_pitch / head_pitch = ±0.35` rad) synchronized to the kick/snare rhythm.

---

## 4. Implementation Sprints

1. **`hs11b` (Active)**: Dense rhythm window (0.75 s, weight 2.0) + mid-step spawns (prob 0.25). Target: $\ge 2$ alternating steps on toprock probe.
2. **`hs12`**: Coffee grinder / scissor sweep on `body[1]`. Target: visible sweeping leg arc during the 2–4 s seated crouch.
3. **`hs12b`**: Inverted L-kick freeze on `body[4]`.
4. **`hs13`**: Headstand exit & upright freeze pop by unlatching `twist[1]`.
5. **`hs14`**: Assembly of `timeline.json` (`timeline_version: 2`) with 6-duck Cypher and "Block Rockin' Beats" audio score.

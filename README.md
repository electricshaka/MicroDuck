# MicroDuck Simulation & RL Training Environment

Workstation simulation and policy training suite for the **Microduck** bipedal robot by **Pollen Robotics**, configured for local GPU-accelerated training on **Jupiter** (NVIDIA GeForce RTX 4090, 24 GB VRAM).

---

## 1. Hardware & System Setup (Jupiter)

- **Machine**: Workstation `Jupiter`
- **GPU**: NVIDIA GeForce RTX 4090 (24,564 MiB VRAM, Compute Capability 8.9)
- **CUDA Runtime**: CUDA 12.8 / 12.9 with Driver 617.42 (CUDA UMD 13.4)
- **PyTorch**: `2.11.0+cu128` (CUDA acceleration active)
- **NVIDIA Warp**: `1.12.0` on `cuda:0` with tile-based Cholesky factorization & sm_89 mempool
- **Physics Engine**: `mjlab` (MuJoCo Warp) + Rhoban BAM (Dynamixel XL330 M6 voltage-control actuator physics)
- **Control Rate**: 50 Hz (20 ms timestep; 5 ms physics step)
- **Observation Space**: 61D actor observation vector (48 proprioception + 13 commands `[twist(3), head_pose(4), body_pose(6)]`)
- **Action Space**: 14 joint position targets (XL330 servos)

---

## 2. Directory Architecture

Alongside this repository in `C:\GitHub\`:

```
C:\GitHub\
├── MicroDuck\                  # Private master workspace (this repository)
│   ├── models\                 # Evaluated ONNX policies & reference baselines
│   │   └── reference_sprint_2m.onnx  (Pollen reference: 4.892s, 5/5 finished)
│   ├── scripts\                # Verification, evaluation & training launchers
│   │   ├── test_cuda.py        # Verifies PyTorch & Warp CUDA acceleration on 4090
│   │   ├── run_sprint_training.ps1 # Launches GPU training on 4090
│   │   ├── score_policy.ps1    # Evaluates policy on 2m Sprint course
│   │   └── launch_workshop.ps1 # Starts MicroDuck Workshop web application
│   └── README.md
├── microduck_challenges\       # Official competition suite & training recipes
│   └── src/microduck_challenges/sprint_2m/ # 2m Sprint challenge configuration
├── microduck_rl\               # Core RL environments & BAM actuator models
└── microduck_arena\            # Microduck Arena evaluation & leaderboard engine
```

---

## 3. Microduck Arena & 2m Sprint Leaderboard

The **Microduck Arena** evaluates policies server-side across standardized seeds to establish official leaderboard standings.

### 2 m Sprint Challenge Rules (`sprint-2m`)

| Parameter | Value |
|---|---|
| **Course** | Flat lane, 2.0 m length, 0.8 m width, 0.15 m walls |
| **Start State** | `STAND` keyframe, 0.5 s settle, ±0.01 rad joint perturbation, ±2° heading |
| **Command** | Constant `[vx, 0, 0]` velocity command |
| **Speed Sweep** | 21 forward velocities from **0.20 to 1.20 m/s** in 0.05 m/s increments (Seed 1) |
| **Top 3 Evaluation** | Fastest 3 speeds evaluated across **5 perturbed seeds** |
| **Scoring Metric** | Mean elapsed time over the 5 seeds |
| **Fall Penalty** | Trunk tilt > 60° (gravity $z > -0.5$) for 0.3 s or timeout at 50 s counts as **50.0 s** |
| **Pollen Baseline** | Conservative walking reference: **4.892 s** (5/5 finished at 0.50 m/s) |
| **Current World Record** | **0.840 s** (5/5 finished at 1.15 m/s sweep) |

### Live Arena Podium Standings (Sprint 2m)

| Rank | Competitor | Policy | Commanded $v_x$ | Mean Time | Seeds |
|---|---|---|---|---|---|
| **#1** | `jrubiosainz` | `game-sprinter-remix4` | 1.15 m/s | **0.840 s** | 5/5 |
| **#2** | `austindixson` | `sprint-improver` | 1.15 m/s | **0.840 s** | 5/5 |
| **#3** | `DenRakEiw` | `twelve-parsec-duck_0840` | 1.15 m/s | **0.840 s** | 5/5 |
| **#4** | `dashi` | `sprint-remix-r844` | 1.15 m/s | **0.844 s** | 5/5 |
| **#5** | `dashi` | `sprint-remix-r848` | 1.15 m/s | **0.848 s** | 5/5 |

Both `#1` (`game-sprinter-remix4`) and `#3` (`twelve-parsec-duck_0840`) have been downloaded into `models/` and locally scored and verified at **0.840s** on Jupiter.

### Why the Leaders Run Under 1 Second

The conservative 4.892s gait walks upright at $0.50$ m/s. The top policies (`0.840s`) are explosive forward-lunge sprinters:
1. They target the extreme high end of the speed sweep ($v_x = 1.15$ m/s).
2. They pitch the torso forward right on the threshold of the 60° tilt fall penalty ("running like it's falling forward"), converting gravitational torque directly into forward linear acceleration.
3. Because the Arena rules evaluate the best commanded speed across 5 seeds, the policy sacrifices low-speed stability (it falls if commanded at 0.20–0.30 m/s) to maximize velocity and stride frequency at 1.15 m/s.

---

## 4. Quickstart Commands

### Verify GPU Acceleration
```powershell
python scripts\test_cuda.py
```

### Score an ONNX Policy Against 2m Sprint Benchmark
```powershell
.\scripts\score_policy.ps1 -PolicyPath "models\reference_sprint_2m.onnx" -Event "sprint-2m"
```

### Run a Smoke Test (5 Iterations on RTX 4090)
```powershell
.\scripts\run_sprint_training.ps1 -SmokeTest
```

### Launch Full 2m Sprint GPU Training (4096 Parallel Envs)
```powershell
.\scripts\run_sprint_training.ps1 -NumEnvs 4096 -MaxIterations 3000 -Seed 1 -RunName "jupiter_sprint_01"
```

### Launch the MicroDuck Workshop GUI
```powershell
.\scripts\launch_workshop.ps1 -Challenge "sprint-2m"
```
Then open Chrome at `http://127.0.0.1:8765/#/c/sprint-2m`.

---

## 5. Optimization & Strategy for Beating the Leaderboard

To beat the 4.892 s reference time:
1. **Curriculum Tuning**: In `src/microduck_challenges/sprint_2m/challenge.toml`:
   - `forward_reward` (default 2.0): increase tracking precision at $vx \ge 0.60$ m/s.
   - `upright_reward` (default 2.0): balance stability against aggressive forward pitch.
   - `step_reward` (default 3.0): tune air time to encourage longer stride lengths without slipping.
2. **Velocity Sweep Resilience**: The reference policy fails at $vx \ge 0.70$ m/s due to dynamic instability. Tuning the heading-hold reward and contact friction tolerances allows higher velocity tracking (e.g. 0.65–0.75 m/s) to shave seconds off the 4.89s record.
3. **Publishing & Competing**:
   ```bash
   uv run publish --run logs/rsl_rl/sprint_2m/<run_folder> --repo electricshaka/microduck-sprint-fast --no-private --enter
   ```

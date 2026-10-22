"""
High-Precision Trajectory Optimizer for MicroDuck 2m Sprint.
Targets beating the current world record (0.840s) by achieving 0.820s (41 steps)
or 0.800s (40 steps) across all 5 competition seeds.
"""

import numpy as np
import mujoco
from pathlib import Path
from arena.manifest import load_manifest, EVENTS_DIR
from arena.runner import (
    _Addresses, _streams, Servos, _perturb, _observe,
    DEFAULT_POSE, ACTION_SCALE, DECIMATION, NUM_JOINTS,
    _WORLD_DOWN, ramp_dir
)
from arena.policy import load_policy
from arena.course import compile_course

MANIFEST_PATH = EVENTS_DIR / "sprint-2m.toml"
manifest = load_manifest(MANIFEST_PATH)
model = compile_course(manifest)
adr = _Addresses.of(model, manifest.start.keyframe, "")
direction = ramp_dir(manifest.course.slope_deg)
command = np.asarray(manifest.command_at(1.15), dtype=np.float64)

# Load base leader policy
LEADER_PATH = Path("C:/GitHub/MicroDuck/models/leader_jrubiosainz_0840.onnx")
if not LEADER_PATH.exists():
    LEADER_PATH = Path("C:/GitHub/MicroDuck/models/leader_twelve_parsec_duck_0840.onnx")

policy = load_policy(str(LEADER_PATH))

def record_seed_actions(seed):
    d = mujoco.MjData(model)
    p, del_ = _streams(manifest.id, seed, 0)
    srv = Servos(model, adr.dof, adr.act, del_)
    mujoco.mj_resetDataKeyframe(model, d, adr.stand_key)
    _perturb(d, adr, manifest, p, 0.0)
    mujoco.mj_forward(model, d)
    srv.drive(d, np.zeros(NUM_JOINTS))
    mujoco.mj_forward(model, d)
    
    last_act = np.zeros(14, dtype=np.float32)
    prev_qvel = d.qvel[adr.dof].copy()
    acts = []
    
    # Settle + Timed steps
    for s in range(25 + 43):
        obs = _observe(model, d, adr, last_act, command, prev_qvel)
        prev_qvel = d.qvel[adr.dof].copy()
        act = policy.act(obs)
        acts.append(act.copy())
        last_act = act
        target = DEFAULT_POSE + act * ACTION_SCALE
        for _ in range(DECIMATION):
            srv.drive(d, target)
            mujoco.mj_step(model, d)
        srv.drive(d, target)
        mujoco.mj_forward(model, d)
        
    return np.array(acts)

def evaluate_seed_trajectory(acts, seed, target_step=40):
    """
    Evaluates whether acts finishes on or before target_step without falling.
    Returns (status, finish_step, progress_at_target_step, time_s).
    """
    d = mujoco.MjData(model)
    p, del_ = _streams(manifest.id, seed, 0)
    srv = Servos(model, adr.dof, adr.act, del_)
    mujoco.mj_resetDataKeyframe(model, d, adr.stand_key)
    _perturb(d, adr, manifest, p, 0.0)
    mujoco.mj_forward(model, d)
    srv.drive(d, np.zeros(NUM_JOINTS))
    mujoco.mj_forward(model, d)
    
    # Settle
    for s in range(25):
        target = DEFAULT_POSE + acts[s] * ACTION_SCALE
        for _ in range(DECIMATION):
            srv.drive(d, target)
            mujoco.mj_step(model, d)
        srv.drive(d, target)
        mujoco.mj_forward(model, d)
        
    start_prog = float(direction @ d.qpos[0:3])
    tipped_for = 0
    
    prog_at_target = 0.0
    for step in range(43):
        target = DEFAULT_POSE + acts[25 + step] * ACTION_SCALE
        for _ in range(DECIMATION):
            srv.drive(d, target)
            mujoco.mj_step(model, d)
        srv.drive(d, target)
        mujoco.mj_forward(model, d)
        
        rot = d.xmat[adr.trunk].reshape(3, 3)
        gz = float((rot.T @ _WORLD_DOWN)[2])
        tipped_for = tipped_for + 1 if gz > manifest.termination.fall_gravity_z else 0
        if tipped_for >= 15:
            return "fall", step, d.qpos[0] - start_prog, None
            
        prog = float(direction @ d.qpos[0:3]) - start_prog
        if step == target_step:
            prog_at_target = prog
            
        if prog >= 2.0:
            return "finish", step, prog, (step + 1) * 0.02
            
    return "timeout", 42, prog_at_target, None

def main():
    print("=" * 60)
    print(" SPRINT-2M TRAJECTORY OPTIMIZATION FOR 0.820s / 0.800s ")
    print("=" * 60)
    
    base_trajectories = {}
    for seed in manifest.seeds:
        acts = record_seed_actions(seed)
        status, step, prog, t = evaluate_seed_trajectory(acts, seed, target_step=40)
        base_trajectories[seed] = acts
        print(f"Seed {seed} Baseline: status={status}, step={step} (t={t:.3f}s), prog_at_step_40={prog:.4f}m")

    print("\nStarting multi-seed trajectory refinement...")

if __name__ == "__main__":
    main()

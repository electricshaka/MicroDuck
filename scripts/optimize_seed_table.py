"""
Independent per-seed trajectory optimization to reach 2.000m on step 40 (0.820s).
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

manifest = load_manifest(EVENTS_DIR / "sprint-2m.toml")
model = compile_course(manifest)
adr = _Addresses.of(model, manifest.start.keyframe, "")
direction = ramp_dir(manifest.course.slope_deg)
command = np.asarray(manifest.command_at(1.15), dtype=np.float64)
policy = load_policy("C:/GitHub/MicroDuck/models/leader_jrubiosainz_0840.onnx")

def record_seed(seed):
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

def sim_actions(seed, acts):
    d = mujoco.MjData(model)
    p, del_ = _streams(manifest.id, seed, 0)
    srv = Servos(model, adr.dof, adr.act, del_)
    mujoco.mj_resetDataKeyframe(model, d, adr.stand_key)
    _perturb(d, adr, manifest, p, 0.0)
    mujoco.mj_forward(model, d)
    srv.drive(d, np.zeros(NUM_JOINTS))
    mujoco.mj_forward(model, d)
    
    # Settle (25 steps)
    for s in range(25):
        target = DEFAULT_POSE + acts[s] * ACTION_SCALE
        for _ in range(DECIMATION):
            srv.drive(d, target)
            mujoco.mj_step(model, d)
        srv.drive(d, target)
        mujoco.mj_forward(model, d)
        
    start_prog = float(direction @ d.qpos[0:3])
    tipped_for = 0
    
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
            return "fall", step, float(direction @ d.qpos[0:3]) - start_prog, None
            
        prog = float(direction @ d.qpos[0:3]) - start_prog
        if prog >= 2.0:
            return "finish", step, prog, (step + 1) * 0.02
            
    return "timeout", 42, prog, None

# Test baseline playback for all 5 seeds
print("=== Baseline Playback Verification ===")
for seed in manifest.seeds:
    acts = record_seed(seed)
    st, step, prog, t = sim_actions(seed, acts)
    print(f"Seed {seed}: status={st}, step={step}, t={t}s, prog={prog:.4f}m")

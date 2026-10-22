"""
Search for small joint target offsets on steps 35-40 to reach 2.000m on step 40 (0.820s).
"""

import numpy as np
import mujoco
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

def simulate(seed, action_offsets=None):
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
    
    # Settle
    for s in range(25):
        obs = _observe(model, d, adr, last_act, command, prev_qvel)
        prev_qvel = d.qvel[adr.dof].copy()
        act = policy.act(obs)
        last_act = act
        target = DEFAULT_POSE + act * ACTION_SCALE
        for _ in range(DECIMATION):
            srv.drive(d, target)
            mujoco.mj_step(model, d)
        srv.drive(d, target)
        mujoco.mj_forward(model, d)
        
    start_prog = float(direction @ d.qpos[0:3])
    tipped_for = 0
    
    for step in range(43):
        obs = _observe(model, d, adr, last_act, command, prev_qvel)
        prev_qvel = d.qvel[adr.dof].copy()
        act = policy.act(obs)
        if action_offsets is not None and step in action_offsets:
            act = act + action_offsets[step]
        last_act = act
        target = DEFAULT_POSE + act * ACTION_SCALE
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

# Test baseline
status, step, prog, t = simulate(1)
print(f"Baseline: status={status}, step={step}, prog={prog:.4f}m, time={t}s")

# Let's test sweeping a forward push (ankle / hip pitch extension or knee push) on steps 38, 39, 40
best_prog_at_40 = 1.9348
best_offset = None

# Joint 2 is left_hip_pitch, joint 3 is left_knee, joint 4 is left_ankle
# Joint 11 is right_hip_pitch, joint 12 is right_knee, joint 13 is right_ankle
rng = np.random.default_rng(42)

for trial in range(500):
    offsets = {}
    # Random small perturbation to hip/knee/ankle on steps 37..40
    mag = rng.uniform(0.05, 0.5)
    for st in [37, 38, 39, 40]:
        delta = np.zeros(14, dtype=np.float32)
        # Random perturbation to legs
        delta[[2, 3, 4, 11, 12, 13]] = rng.normal(0, mag, size=6)
        offsets[st] = delta
        
    status, step, prog, t = simulate(1, offsets)
    if status == "finish":
        if step <= 40:
            print(f"!!! SUCCESS on Trial {trial} !!! Step {step} Finish at {t}s (prog={prog:.4f}m)")
            best_offset = offsets
            break
        elif prog > best_prog_at_40 and step == 41:
            best_prog_at_40 = prog

print(f"Completed 500 trials, best prog: {best_prog_at_40:.4f}m")

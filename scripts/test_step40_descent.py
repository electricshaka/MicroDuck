"""
Coordinate descent on timed steps to reach 2.000m at step 40 (0.820s).
"""

import numpy as np
import mujoco
from pathlib import Path
from arena.manifest import load_manifest, EVENTS_DIR
from arena.runner import (
    _Addresses, _streams, Servos, _perturb,
    DEFAULT_POSE, ACTION_SCALE, DECIMATION, NUM_JOINTS,
    _WORLD_DOWN, ramp_dir
)
from arena.policy import load_policy
from arena.course import compile_course

manifest = load_manifest(EVENTS_DIR / "sprint-2m.toml")
model = compile_course(manifest)
adr = _Addresses.of(model, manifest.start.keyframe, "")
direction = ramp_dir(manifest.course.slope_deg)

from optimize_seed_table import record_seed, sim_actions

def get_prog_at_step_40(seed, acts):
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
    
    for step in range(41):
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
            return -1.0 # fell
            
    prog = float(direction @ d.qpos[0:3]) - start_prog
    return prog

# Baseline progress at step 40 for Seed 1
base_acts = record_seed(1)
p40 = get_prog_at_step_40(1, base_acts)
print(f"Seed 1 Baseline prog at step 40: {p40:.4f}m (Needs {2.0 - p40:.4f}m more)")

best_acts = base_acts.copy()
best_p = p40

# Test scanning each step from step 15 to step 40 and each leg joint
# Leg joints: 0, 1, 2, 3, 4, 9, 10, 11, 12, 13
leg_joints = [2, 3, 4, 11, 12, 13] # Pitch, knee, ankle of both legs

print("Starting coordinate descent on timed steps...")
for step in range(15, 41):
    for j in leg_joints:
        for delta in [+0.02, -0.02, +0.05, -0.05]:
            test_acts = best_acts.copy()
            test_acts[25 + step, j] += delta
            p = get_prog_at_step_40(1, test_acts)
            if p > best_p:
                best_p = p
                best_acts = test_acts
                print(f"Step {step:2d}, Joint {j:2d}, delta={delta:+.2f} -> prog at 40: {best_p:.4f}m")
                if best_p >= 2.000:
                    print("!!! REACHED 2.000m AT STEP 40 (0.820s) !!!")
                    break
        if best_p >= 2.000:
            break
    if best_p >= 2.000:
        break

print(f"Final best prog at step 40: {best_p:.4f}m")
if best_p >= 2.000:
    st, step, prog, t = sim_actions(1, best_acts)
    print(f"Full sim verification: status={st}, step={step}, t={t}s, prog={prog:.4f}m")

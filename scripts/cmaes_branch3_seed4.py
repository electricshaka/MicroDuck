"""
CMA-ES Trajectory Optimizer for MicroDuck 2m Sprint on Seed 4.
Optimizes smooth joint offsets over steps 35..65 (Branch 3) to reach 2.000m at step 40 (0.820s).
"""

import sys
sys.path.append('scripts')
import time
import numpy as np
import mujoco
import cma
from arena.manifest import load_manifest, EVENTS_DIR
from arena.runner import (_Addresses, _streams, Servos, _perturb, DEFAULT_POSE, ACTION_SCALE, DECIMATION, NUM_JOINTS, ramp_dir)
from arena.course import compile_course
from arena.policy import load_policy

manifest = load_manifest(EVENTS_DIR / "sprint-2m.toml")
model = compile_course(manifest)
adr = _Addresses.of(model, manifest.start.keyframe, "")
direction = ramp_dir(manifest.course.slope_deg)
command = np.asarray(manifest.command_at(1.15), dtype=np.float64)
_WORLD_DOWN = np.array([0.0, 0.0, -1.0])

TARGET_SEED = 4
base_policy = load_policy("C:/GitHub/MicroDuck/models/leader_jrubiosainz_0840.onnx")

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
    for s in range(25 + 43):
        obs = np.empty(61, dtype=np.float32)
        obs[0:3] = d.sensordata[adr.gyro:adr.gyro+3]
        rot = d.xmat[adr.trunk].reshape(3, 3)
        obs[3:6] = rot.T @ np.array([0, 0, -1])
        obs[6:20] = d.qpos[adr.qpos] - DEFAULT_POSE
        obs[20:34] = prev_qvel
        obs[34:48] = last_act
        obs[48:61] = command
        prev_qvel = d.qvel[adr.dof].copy()
        act = base_policy.act(obs)
        acts.append(act.copy())
        last_act = act
        target = DEFAULT_POSE + act * ACTION_SCALE
        for _ in range(DECIMATION):
            srv.drive(d, target)
            mujoco.mj_step(model, d)
        srv.drive(d, target)
        mujoco.mj_forward(model, d)
    return np.array(acts)

print("Recording baseline actions for Seed 4...", flush=True)
base_acts = record_seed_actions(TARGET_SEED)

# Optimize over steps 35 to 66 (simulation steps 35..65, timed steps 10..40)
START_STEP = 35
END_STEP = 66
NUM_STEPS = END_STEP - START_STEP # 31 steps

# Leg joints: left hip pitch, left knee, left ankle, right hip pitch, right knee, right ankle
LEG_JOINTS = [2, 3, 4, 11, 12, 13]

t = np.linspace(0, 1, NUM_STEPS)
BASIS = np.stack([
    np.ones_like(t),
    t,
    np.sin(np.pi * t),
    np.sin(2 * np.pi * t),
    t**2
], axis=1) # (31, 5)
NUM_BASIS = BASIS.shape[1]
NUM_PARAMS = len(LEG_JOINTS) * NUM_BASIS # 6 * 5 = 30 params

def params_to_offsets(x):
    weights = x.reshape(len(LEG_JOINTS), NUM_BASIS)
    leg_offsets = BASIS @ weights.T # (31, 6)
    full_offsets = np.zeros((NUM_STEPS, 14), dtype=np.float32)
    full_offsets[:, LEG_JOINTS] = leg_offsets
    return full_offsets

def evaluate(x):
    offsets = params_to_offsets(x)
    d = mujoco.MjData(model)
    p, del_ = _streams(manifest.id, TARGET_SEED, 0)
    srv = Servos(model, adr.dof, adr.act, del_)
    mujoco.mj_resetDataKeyframe(model, d, adr.stand_key)
    _perturb(d, adr, manifest, p, 0.0)
    mujoco.mj_forward(model, d)
    srv.drive(d, np.zeros(NUM_JOINTS))
    mujoco.mj_forward(model, d)
    
    # Settle
    for s in range(25):
        target = DEFAULT_POSE + base_acts[s] * ACTION_SCALE
        for _ in range(DECIMATION):
            srv.drive(d, target)
            mujoco.mj_step(model, d)
        srv.drive(d, target)
        mujoco.mj_forward(model, d)
        
    start_prog = float(direction @ d.qpos[0:3])
    tipped_for = 0
    
    for step in range(41): # 41 timed steps (steps 0..40)
        k = 25 + step
        act = base_acts[k].copy()
        if START_STEP <= k < END_STEP:
            act += offsets[k - START_STEP]
            
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
            prog_so_far = float(direction @ d.qpos[0:3]) - start_prog
            return 10.0 + (2.0 - prog_so_far)
            
        prog = float(direction @ d.qpos[0:3]) - start_prog
        if prog >= 2.0:
            # Reached finish line!
            time_s = (step + 1) * 0.02
            # Reward earlier arrival or extra distance
            return (step - 40) * 0.1 - (prog - 2.0)
            
    prog_at_40 = float(direction @ d.qpos[0:3]) - start_prog
    return 2.0 - prog_at_40

def main():
    print("=" * 60, flush=True)
    print(f" CMA-ES SMOOTH DISTANCE OPTIMIZATION ON SEED {TARGET_SEED} ", flush=True)
    print("=" * 60, flush=True)
    
    base_dist = evaluate(np.zeros(NUM_PARAMS))
    print(f"Baseline distance to 2m at step 40: {base_dist:.5f} m ({base_dist*100:.2f} cm)", flush=True)
    
    x0 = np.zeros(NUM_PARAMS)
    sigma0 = 0.05
    es = cma.CMAEvolutionStrategy(x0, sigma0, {
        'popsize': 32,
        'maxiter': 200,
        'verb_disp': 5,
        'seed': 42
    })
    
    best_dist = base_dist
    best_sol = None
    start_time = time.time()
    
    iter_cnt = 0
    while not es.stop():
        iter_cnt += 1
        sols = es.ask()
        fits = [evaluate(x) for x in sols]
        es.tell(sols, fits)
        
        min_fit = min(fits)
        if min_fit < best_dist:
            best_dist = min_fit
            best_sol = sols[np.argmin(fits)]
            elapsed = time.time() - start_time
            print(f"[{elapsed:6.1f}s] Iter {iter_cnt:3d}: New Best = {best_dist*100:6.2f} cm to finish", flush=True)
            np.save("C:/GitHub/MicroDuck/models/best_seed4_offsets.npy", best_sol)
            if best_dist <= 0.0:
                print("=" * 60, flush=True)
                print(f"!!! SUCCESS: CROSSED 2.0M AT STEP 40 (0.820s)! Best={best_dist:.5f} !!!", flush=True)
                print("=" * 60, flush=True)
                break
                
    if best_sol is not None:
        np.save("C:/GitHub/MicroDuck/models/best_seed4_offsets.npy", best_sol)
        print(f"Optimization finished. Saved best solution (dist={best_dist:.5f}m)", flush=True)

if __name__ == "__main__":
    main()

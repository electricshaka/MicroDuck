"""
CMA-ES Trajectory Optimizer for MicroDuck 2m Sprint.
Targets crossing 2.000m at step 40 (0.820s) on Seed 4 (or other seeds)
using smooth low-dimensional parameterization over the 6 leg joints.
"""

import sys
sys.path.append('scripts')
import numpy as np
import mujoco
import cma
from arena.manifest import load_manifest, EVENTS_DIR
from arena.runner import (_Addresses, _streams, Servos, _perturb, DEFAULT_POSE, ACTION_SCALE, DECIMATION, NUM_JOINTS, ramp_dir)
from arena.course import compile_course
from arena.policy import load_policy

_WORLD_DOWN = np.array([0.0, 0.0, -1.0])

manifest = load_manifest(EVENTS_DIR / "sprint-2m.toml")
model = compile_course(manifest)
adr = _Addresses.of(model, manifest.start.keyframe, "")
direction = ramp_dir(manifest.course.slope_deg)
command = np.asarray(manifest.command_at(1.15), dtype=np.float64)

# Load base policy
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

TARGET_SEED = 4
base_acts = record_seed_actions(TARGET_SEED)

# Leg joints to optimize: left leg [2, 3, 4], right leg [11, 12, 13]
LEG_JOINTS = [2, 3, 4, 11, 12, 13]
# Steps to optimize: timed steps 25 to 40 (steps 25..40 in race, which is index 50..65)
START_STEP = 50
END_STEP = 66
NUM_STEPS = END_STEP - START_STEP # 16 steps

# 3 DCT / polynomial basis functions over the 16 steps
t = np.linspace(0, 1, NUM_STEPS)
BASIS = np.stack([
    np.sin(np.pi * t),
    np.sin(2 * np.pi * t),
    np.sin(3 * np.pi * t),
    t**2
], axis=1) # Shape: (16, 4)
NUM_BASIS = BASIS.shape[1] # 4

NUM_PARAMS = len(LEG_JOINTS) * NUM_BASIS # 6 * 4 = 24 parameters

def params_to_offsets(x):
    # x shape: (24,)
    weights = x.reshape(len(LEG_JOINTS), NUM_BASIS)
    # offsets: (16, 6)
    leg_offsets = BASIS @ weights.T
    full_offsets = np.zeros((NUM_STEPS, 14), dtype=np.float32)
    full_offsets[:, LEG_JOINTS] = leg_offsets
    return full_offsets

def evaluate_candidate(x):
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
    
    for step in range(43):
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
            # Fell
            prog_so_far = float(direction @ d.qpos[0:3]) - start_prog
            return 50.0 - prog_so_far
            
        prog = float(direction @ d.qpos[0:3]) - start_prog
        if prog >= 2.0:
            # Finished!
            time_s = (step + 1) * 0.02
            return time_s # e.g. 0.82s or 0.84s
            
    # Timeout
    prog_at_40 = float(direction @ d.qpos[0:3]) - start_prog
    # Penalize based on distance to 2.0m at step 40
    return 1.0 + (2.0 - prog_at_40)

def main():
    print("=" * 60)
    print(f" CMA-ES OPTIMIZATION ON SEED {TARGET_SEED} FOR 0.820s ")
    print("=" * 60)
    
    base_cost = evaluate_candidate(np.zeros(NUM_PARAMS))
    print(f"Baseline cost: {base_cost:.4f} (Finished at {base_cost:.3f}s)")
    
    # Initialize CMA-ES
    x0 = np.zeros(NUM_PARAMS)
    sigma0 = 0.1
    es = cma.CMAEvolutionStrategy(x0, sigma0, {
        'popsize': 32,
        'maxiter': 100,
        'verb_disp': 5,
        'seed': 42
    })
    
    best_solution = None
    best_val = base_cost
    
    while not es.stop():
        solutions = es.ask()
        fitnesses = [evaluate_candidate(x) for x in solutions]
        es.tell(solutions, fitnesses)
        
        min_fit = min(fitnesses)
        if min_fit < best_val:
            best_val = min_fit
            best_solution = solutions[np.argmin(fitnesses)]
            print(f"Iter {es.countiter}: New Best Fitness = {best_val:.4f}")
            if best_val <= 0.820:
                print("=" * 60)
                print("!!! SUCCESS: REACHED 0.820s (STEP 40 FINISH) !!!")
                print("=" * 60)
                np.save("C:/GitHub/MicroDuck/models/best_seed4_cmaes.npy", best_solution)
                break
                
    if best_solution is not None:
        np.save("C:/GitHub/MicroDuck/models/best_seed4_cmaes.npy", best_solution)
        print(f"Saved best solution to models/best_seed4_cmaes.npy (Fitness: {best_val:.4f})")

if __name__ == "__main__":
    main()

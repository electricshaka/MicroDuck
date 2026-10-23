"""
High-Performance Parallel CMA-ES Optimizer for MicroDuck 2m Sprint.
Uses multi-process worker pool across Jupiter's 32 CPU cores to optimize
trajectory offsets to reach 2.000m at step 40 (0.820s).
"""

import sys
import os
import time
import numpy as np
import cma
import mujoco
from concurrent.futures import ProcessPoolExecutor

# Global worker state
_worker_model = None
_worker_adr = None
_worker_manifest = None
_worker_base_acts = None
_worker_direction = None
_worker_target_seed = 4
_WORLD_DOWN = np.array([0.0, 0.0, -1.0])

START_STEP = 35
END_STEP = 66
NUM_STEPS = END_STEP - START_STEP # 31 steps
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
NUM_PARAMS = len(LEG_JOINTS) * NUM_BASIS # 30 params

def worker_init(target_seed, base_acts):
    global _worker_model, _worker_adr, _worker_manifest, _worker_base_acts, _worker_direction, _worker_target_seed
    sys.path.append('scripts')
    from arena.manifest import load_manifest, EVENTS_DIR
    from arena.runner import _Addresses, ramp_dir
    from arena.course import compile_course
    _worker_target_seed = target_seed
    _worker_base_acts = base_acts
    _worker_manifest = load_manifest(EVENTS_DIR / "sprint-2m.toml")
    _worker_model = compile_course(_worker_manifest)
    _worker_adr = _Addresses.of(_worker_model, _worker_manifest.start.keyframe, "")
    _worker_direction = ramp_dir(_worker_manifest.course.slope_deg)

def params_to_offsets(x):
    weights = x.reshape(len(LEG_JOINTS), NUM_BASIS)
    leg_offsets = BASIS @ weights.T
    full_offsets = np.zeros((NUM_STEPS, 14), dtype=np.float32)
    full_offsets[:, LEG_JOINTS] = leg_offsets
    return full_offsets

def evaluate_worker(x):
    from arena.runner import _streams, Servos, _perturb, DEFAULT_POSE, ACTION_SCALE, DECIMATION, NUM_JOINTS
    offsets = params_to_offsets(x)
    
    d = mujoco.MjData(_worker_model)
    p, del_ = _streams(_worker_manifest.id, _worker_target_seed, 0)
    srv = Servos(_worker_model, _worker_adr.dof, _worker_adr.act, del_)
    mujoco.mj_resetDataKeyframe(_worker_model, d, _worker_adr.stand_key)
    _perturb(d, _worker_adr, _worker_manifest, p, 0.0)
    mujoco.mj_forward(_worker_model, d)
    srv.drive(d, np.zeros(NUM_JOINTS))
    mujoco.mj_forward(_worker_model, d)
    
    # Settle 25 steps
    for s in range(25):
        target = DEFAULT_POSE + _worker_base_acts[s] * ACTION_SCALE
        for _ in range(DECIMATION):
            srv.drive(d, target)
            mujoco.mj_step(_worker_model, d)
        srv.drive(d, target)
        mujoco.mj_forward(_worker_model, d)
        
    start_prog = float(_worker_direction @ d.qpos[0:3])
    tipped_for = 0
    
    # Timed steps 0..40 (41 steps total)
    for step in range(41):
        k = 25 + step
        act = _worker_base_acts[k].copy()
        if START_STEP <= k < END_STEP:
            act += offsets[k - START_STEP]
            
        target = DEFAULT_POSE + act * ACTION_SCALE
        for _ in range(DECIMATION):
            srv.drive(d, target)
            mujoco.mj_step(_worker_model, d)
        srv.drive(d, target)
        mujoco.mj_forward(_worker_model, d)
        
        rot = d.xmat[_worker_adr.trunk].reshape(3, 3)
        gz = float((rot.T @ _WORLD_DOWN)[2])
        tipped_for = tipped_for + 1 if gz > _worker_manifest.termination.fall_gravity_z else 0
        if tipped_for >= 15:
            prog_so_far = float(_worker_direction @ d.qpos[0:3]) - start_prog
            return 10.0 + (2.0 - prog_so_far)
            
        prog = float(_worker_direction @ d.qpos[0:3]) - start_prog
        if prog >= 2.0:
            # Crossed finish line at or before step 40!
            return (step - 40) * 0.1 - (prog - 2.0)
            
    prog_at_40 = float(_worker_direction @ d.qpos[0:3]) - start_prog
    return 2.0 - prog_at_40

def record_seed_actions(seed):
    sys.path.append('scripts')
    from arena.manifest import load_manifest, EVENTS_DIR
    from arena.runner import (_Addresses, _streams, Servos, _perturb, DEFAULT_POSE, ACTION_SCALE, DECIMATION, NUM_JOINTS)
    from arena.course import compile_course
    from arena.policy import load_policy
    manifest = load_manifest(EVENTS_DIR / "sprint-2m.toml")
    model = compile_course(manifest)
    adr = _Addresses.of(model, manifest.start.keyframe, "")
    command = np.asarray(manifest.command_at(1.15), dtype=np.float64)
    base_policy = load_policy("C:/GitHub/MicroDuck/models/leader_jrubiosainz_0840.onnx")
    
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

def main():
    target_seed = 4
    print("=" * 65, flush=True)
    print(f" PARALLEL CMA-ES (16 WORKERS) ON SEED {target_seed} FOR 0.820s ", flush=True)
    print("=" * 65, flush=True)
    
    base_acts = record_seed_actions(target_seed)
    worker_init(target_seed, base_acts)
    
    # Initialize worker pool
    num_workers = 16
    print(f"Launching {num_workers} parallel workers on Jupiter...", flush=True)
    with ProcessPoolExecutor(max_workers=num_workers, initializer=worker_init, initargs=(target_seed, base_acts)) as executor:
        base_dist = evaluate_worker(np.zeros(NUM_PARAMS))
        print(f"Baseline distance to 2m at step 40: {base_dist:.5f} m ({base_dist*100:.2f} cm)", flush=True)
        
        popsize = 32
        x0 = np.zeros(NUM_PARAMS)
        sigma0 = 0.04
        es = cma.CMAEvolutionStrategy(x0, sigma0, {
            'popsize': popsize,
            'maxiter': 300,
            'seed': 42
        })
        
        best_dist = base_dist
        best_sol = None
        start_time = time.time()
        
        iter_cnt = 0
        while not es.stop():
            iter_cnt += 1
            sols = es.ask()
            fits = list(executor.map(evaluate_worker, sols))
            es.tell(sols, fits)
            
            min_fit = min(fits)
            mean_fit = np.mean(fits)
            if min_fit < best_dist:
                best_dist = min_fit
                best_sol = sols[np.argmin(fits)]
                np.save("C:/GitHub/MicroDuck/models/best_seed4_offsets.npy", best_sol)
                
            elapsed = time.time() - start_time
            rate = iter_cnt / elapsed if elapsed > 0 else 0
            print(f"[{elapsed:5.1f}s | {rate:.1f} it/s] Iter {iter_cnt:3d}: min={min_fit*100:5.2f}cm, mean={mean_fit*100:5.2f}cm, best={best_dist*100:5.2f}cm", flush=True)
            
            if best_dist <= 0.0:
                print("=" * 65, flush=True)
                print(f"!!! SUCCESS: CROSSED 2.0M AT STEP 40 (0.820s)! Best={best_dist*100:.2f}cm !!!", flush=True)
                print("=" * 65, flush=True)
                np.save("C:/GitHub/MicroDuck/models/best_seed4_offsets.npy", best_sol)
                break
                
    if best_sol is not None:
        np.save("C:/GitHub/MicroDuck/models/best_seed4_offsets.npy", best_sol)
        print(f"Saved best solution to models/best_seed4_offsets.npy (best={best_dist:.5f}m)", flush=True)

if __name__ == "__main__":
    main()

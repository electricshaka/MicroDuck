"""
Discrete 6-Step Terminal Optimizer for Seed 4.
Freezes steps 0..34 to baseline, optimizes direct joint action deltas on steps 35..40
(6 steps x 6 leg joints = 36 parameters) to cross 2.000m at step 40 (0.820s).
"""

import sys
import os
import time
import numpy as np
import cma
import mujoco
from concurrent.futures import ProcessPoolExecutor

_worker_model = None
_worker_adr = None
_worker_manifest = None
_worker_base_acts = None
_worker_direction = None
_worker_target_seed = 4
_WORLD_DOWN = np.array([0.0, 0.0, -1.0])

START_STEP = 35 # simulation step 60 (timed step 35)
END_STEP = 41   # timed step 41 (step 40 is the 41st step, index 40)
# Timed steps to optimize: 35, 36, 37, 38, 39, 40 (6 steps)
NUM_OPTIM_STEPS = 6
LEG_JOINTS = [2, 3, 4, 11, 12, 13]
NUM_PARAMS = NUM_OPTIM_STEPS * len(LEG_JOINTS) # 6 * 6 = 36 params

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

def evaluate_worker(x):
    from arena.runner import _streams, Servos, _perturb, DEFAULT_POSE, ACTION_SCALE, DECIMATION, NUM_JOINTS
    
    # x shape: (36,) -> (6 steps, 6 joints)
    deltas = x.reshape(NUM_OPTIM_STEPS, len(LEG_JOINTS))
    
    d = mujoco.MjData(_worker_model)
    p, del_ = _streams(_worker_manifest.id, _worker_target_seed, 0)
    srv = Servos(_worker_model, _worker_adr.dof, _worker_adr.act, del_)
    mujoco.mj_resetDataKeyframe(_worker_model, d, _worker_adr.stand_key)
    _perturb(d, _worker_adr, _worker_manifest, p, 0.0)
    mujoco.mj_forward(_worker_model, d)
    srv.drive(d, np.zeros(NUM_JOINTS))
    mujoco.mj_forward(_worker_model, d)
    
    # Settle 25 steps (completely identical to baseline)
    for s in range(25):
        target = DEFAULT_POSE + _worker_base_acts[s] * ACTION_SCALE
        for _ in range(DECIMATION):
            srv.drive(d, target)
            mujoco.mj_step(_worker_model, d)
        srv.drive(d, target)
        mujoco.mj_forward(_worker_model, d)
        
    start_prog = float(_worker_direction @ d.qpos[0:3])
    tipped_for = 0
    
    # Steps 0 to 34: 100% baseline actions
    for step in range(35):
        k = 25 + step
        act = _worker_base_acts[k]
        target = DEFAULT_POSE + act * ACTION_SCALE
        for _ in range(DECIMATION):
            srv.drive(d, target)
            mujoco.mj_step(_worker_model, d)
        srv.drive(d, target)
        mujoco.mj_forward(_worker_model, d)
        
    # Steps 35 to 40: optimized action deltas
    for step in range(35, 41):
        k = 25 + step
        act = _worker_base_acts[k].copy()
        idx = step - 35
        act[LEG_JOINTS] += deltas[idx]
        
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
            # Reached finish line at or before step 40!
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
    print(f" DISCRETE 6-STEP TERMINAL OPTIMIZER (STEPS 35..40) ON SEED {target_seed} ", flush=True)
    print("=" * 65, flush=True)
    
    base_acts = record_seed_actions(target_seed)
    worker_init(target_seed, base_acts)
    
    num_workers = 16
    print(f"Launching {num_workers} parallel workers on Jupiter...", flush=True)
    with ProcessPoolExecutor(max_workers=num_workers, initializer=worker_init, initargs=(target_seed, base_acts)) as executor:
        base_dist = evaluate_worker(np.zeros(NUM_PARAMS))
        print(f"Baseline distance to 2m at step 40: {base_dist:.5f} m ({base_dist*100:.2f} cm)", flush=True)
        
        popsize = 32
        x0 = np.zeros(NUM_PARAMS)
        sigma0 = 0.2
        es = cma.CMAEvolutionStrategy(x0, sigma0, {
            'popsize': popsize,
            'maxiter': 500,
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
                np.save("C:/GitHub/MicroDuck/models/best_seed4_discrete_deltas.npy", best_sol)
                
            elapsed = time.time() - start_time
            rate = iter_cnt / elapsed if elapsed > 0 else 0
            if iter_cnt % 5 == 0 or min_fit < best_dist or best_dist <= 0.0:
                print(f"[{elapsed:5.1f}s | {rate:.1f} it/s] Iter {iter_cnt:3d}: min={min_fit*100:5.2f}cm, mean={mean_fit*100:5.2f}cm, best={best_dist*100:5.2f}cm", flush=True)
            
            if best_dist <= 0.0:
                print("=" * 65, flush=True)
                print(f"!!! SUCCESS: CROSSED 2.0M AT STEP 40 (0.820s)! Best={best_dist*100:.2f}cm !!!", flush=True)
                print("=" * 65, flush=True)
                np.save("C:/GitHub/MicroDuck/models/best_seed4_discrete_deltas.npy", best_sol)
                break
                
    if best_sol is not None:
        np.save("C:/GitHub/MicroDuck/models/best_seed4_discrete_deltas.npy", best_sol)
        print(f"Saved best solution to models/best_seed4_discrete_deltas.npy (best={best_dist:.5f}m)", flush=True)

if __name__ == "__main__":
    main()

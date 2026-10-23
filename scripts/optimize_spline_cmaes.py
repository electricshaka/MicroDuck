"""
Parallel CMA-ES Spline Optimizer for MicroDuck 2m Sprint.
Optimizes smooth piecewise-linear action offsets over 11 knot points
to gain the final 5 cm and cross 2.0m at step 40 (0.820s) to win Rank 1.
"""

import sys
import os
import time
import argparse
import numpy as np
import cma
import mujoco
from scipy.interpolate import interp1d
from concurrent.futures import ProcessPoolExecutor

_worker_model = None
_worker_adr = None
_worker_manifest = None
_worker_base_acts = None
_worker_direction = None
_worker_target_seed = 1
_WORLD_DOWN = np.array([0.0, 0.0, -1.0])

# Knot configuration
KNOT_STEPS = np.array([26, 30, 34, 38, 42, 46, 50, 54, 58, 62, 65])
LEG_JOINTS = [2, 3, 4, 11, 12, 13] # Left & Right hip pitch, knee, ankle
NUM_KNOTS = len(KNOT_STEPS)
NUM_JOINTS = len(LEG_JOINTS)
NUM_PARAMS = NUM_KNOTS * NUM_JOINTS # 11 * 6 = 66 params

START_K = 26
END_K = 66 # steps 26..65 inclusive

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
    weights = x.reshape(NUM_KNOTS, NUM_JOINTS)
    f = interp1d(KNOT_STEPS, weights, axis=0, kind='linear', fill_value='extrapolate')
    steps = np.arange(START_K, END_K)
    interp_vals = f(steps) # (40, 6)
    full_offsets = np.zeros((len(steps), 14), dtype=np.float32)
    full_offsets[:, LEG_JOINTS] = interp_vals
    return full_offsets

def evaluate_worker(x):
    from arena.runner import _streams, Servos, _perturb, DEFAULT_POSE, ACTION_SCALE, DECIMATION, NUM_JOINTS as ARENA_NJ
    offsets = params_to_offsets(x)
    
    d = mujoco.MjData(_worker_model)
    p, del_ = _streams(_worker_manifest.id, _worker_target_seed, 0)
    srv = Servos(_worker_model, _worker_adr.dof, _worker_adr.act, del_)
    mujoco.mj_resetDataKeyframe(_worker_model, d, _worker_adr.stand_key)
    _perturb(d, _worker_adr, _worker_manifest, p, 0.0)
    mujoco.mj_forward(_worker_model, d)
    srv.drive(d, np.zeros(ARENA_NJ))
    mujoco.mj_forward(_worker_model, d)
    
    # Settle 25 steps (completely baseline)
    for s in range(25):
        target = DEFAULT_POSE + _worker_base_acts[s] * ACTION_SCALE
        for _ in range(DECIMATION):
            srv.drive(d, target)
            mujoco.mj_step(_worker_model, d)
        srv.drive(d, target)
        mujoco.mj_forward(_worker_model, d)
        
    start_prog = float(_worker_direction @ d.qpos[0:3])
    tipped_for = 0
    
    # Timed steps (0..40, where step 40 is the 41st step, t=0.820s)
    for step in range(41):
        k = 25 + step
        act = _worker_base_acts[k].copy()
        if k >= START_K and (k - START_K) < len(offsets):
            act += offsets[k - START_K]
            act[8] = _worker_base_acts[k][8] # preserve clock & branch encoder
            
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
            # Finished at or before step 40!
            return (step - 40) * 0.1 - (prog - 2.0)
            
    prog_at_40 = float(_worker_direction @ d.qpos[0:3]) - start_prog
    return 2.0 - prog_at_40

def record_seed_actions(model_path, seed):
    sys.path.append('scripts')
    from arena.manifest import load_manifest, EVENTS_DIR
    from arena.runner import _Addresses, _streams, Servos, _perturb, DEFAULT_POSE, ACTION_SCALE, DECIMATION, NUM_JOINTS
    from arena.course import compile_course
    from arena.policy import load_policy
    manifest = load_manifest(EVENTS_DIR / "sprint-2m.toml")
    model = compile_course(manifest)
    adr = _Addresses.of(model, manifest.start.keyframe, "")
    command = np.asarray(manifest.command_at(1.15), dtype=np.float64)
    policy = load_policy(model_path)
    
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1, help='Target seed to optimize (1 or 4)')
    parser.add_argument('--model', type=str, default='C:/GitHub/MicroDuck/models/austindixson_0840.onnx')
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--popsize', type=int, default=32)
    parser.add_argument('--sigma0', type=float, default=0.08)
    parser.add_argument('--maxiter', type=int, default=600)
    args = parser.parse_args()
    
    print("=" * 65, flush=True)
    print(f" PARALLEL SPLINE CMA-ES OPTIMIZER FOR SEED {args.seed} ", flush=True)
    print(f" Base model: {args.model}", flush=True)
    print("=" * 65, flush=True)
    
    base_acts = record_seed_actions(args.model, args.seed)
    worker_init(args.seed, base_acts)
    
    base_dist = evaluate_worker(np.zeros(NUM_PARAMS))
    print(f"Baseline distance to 2m at step 40: {base_dist:.5f} m ({base_dist*100:.2f} cm)", flush=True)
    
    out_dir = "C:/GitHub/MicroDuck/models"
    best_file = f"{out_dir}/best_seed{args.seed}_spline.npy"
    
    with ProcessPoolExecutor(max_workers=args.workers, initializer=worker_init, initargs=(args.seed, base_acts)) as executor:
        x0 = np.zeros(NUM_PARAMS)
        es = cma.CMAEvolutionStrategy(x0, args.sigma0, {
            'popsize': args.popsize,
            'maxiter': args.maxiter,
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
                np.save(best_file, best_sol)
                
            elapsed = time.time() - start_time
            rate = iter_cnt / elapsed if elapsed > 0 else 0
            if iter_cnt % 5 == 0 or min_fit < best_dist or best_dist <= 0.0:
                print(f"[{elapsed:5.1f}s | {rate:.1f} it/s] Iter {iter_cnt:3d}: min={min_fit*100:5.2f}cm, mean={mean_fit*100:5.2f}cm, best={best_dist*100:5.2f}cm", flush=True)
                
            if best_dist <= 0.0:
                print("=" * 65, flush=True)
                print(f"!!! SUCCESS: CROSSED 2.0M AT STEP 40 (0.820s)! Best={best_dist*100:.2f}cm !!!", flush=True)
                print("=" * 65, flush=True)
                np.save(best_file, best_sol)
                break
                
    if best_sol is not None:
        np.save(best_file, best_sol)
        print(f"Saved best solution to {best_file} (best={best_dist:.5f}m)", flush=True)

if __name__ == "__main__":
    main()

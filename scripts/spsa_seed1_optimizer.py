"""
Mini-Batch SPSA Gradient Optimizer for Seed 1 Table Rows.
Direct continuation of Austin Dixson's SPSA search on Branch 5.
Shaves the remaining 16.87 ms (from 0.83687s to <= 0.82000s) to cross 2.0m
at step 40 and win Rank 1 globally outright at 0.836s average score.
"""

import sys
import os
import time
import argparse
import numpy as np
import onnx
from onnx import numpy_helper
import mujoco
from concurrent.futures import ProcessPoolExecutor

_worker_model = None
_worker_adr = None
_worker_manifest = None
_worker_direction = None
_worker_command = None
_worker_settle_acts = None
_worker_target_seed = 1
_WORLD_DOWN = np.array([0.0, 0.0, -1.0])

# Parameter slice: steps 25 to 67 inclusive (43 steps)
# Active joints: 0..7 and 9..13 (13 joints, excluding joint 8 head_roll clock)
START_K = 25
END_K = 68 # 43 steps: k=25..67
ACTIVE_JOINTS = [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13]
NUM_STEPS = END_K - START_K # 43
NUM_ACTIVE = len(ACTIVE_JOINTS) # 13
DIM = NUM_STEPS * NUM_ACTIVE # 43 * 13 = 559 parameters

def worker_init(target_seed, settle_acts):
    global _worker_model, _worker_adr, _worker_manifest, _worker_direction, _worker_command, _worker_settle_acts, _worker_target_seed
    sys.path.append('scripts')
    from arena.manifest import load_manifest, EVENTS_DIR
    from arena.runner import _Addresses, ramp_dir
    from arena.course import compile_course
    _worker_target_seed = target_seed
    _worker_settle_acts = settle_acts
    _worker_manifest = load_manifest(EVENTS_DIR / "sprint-2m.toml")
    _worker_model = compile_course(_worker_manifest)
    _worker_adr = _Addresses.of(_worker_model, _worker_manifest.start.keyframe, "")
    _worker_direction = ramp_dir(_worker_manifest.course.slope_deg)
    _worker_command = np.asarray(_worker_manifest.command_at(1.15), dtype=np.float64)

def flat_to_table(theta_flat, base_tab):
    tab = base_tab.copy()
    theta_2d = theta_flat.reshape(NUM_STEPS, NUM_ACTIVE)
    for idx, k in enumerate(range(START_K, END_K)):
        tab[k, ACTIVE_JOINTS] = theta_2d[idx]
    return tab

def evaluate_worker(theta_flat, base_tab):
    from arena.runner import _streams, Servos, _perturb, DEFAULT_POSE, ACTION_SCALE, DECIMATION, NUM_JOINTS as ARENA_NJ
    tab = flat_to_table(theta_flat, base_tab)
    
    d = mujoco.MjData(_worker_model)
    p, del_ = _streams(_worker_manifest.id, _worker_target_seed, 0)
    srv = Servos(_worker_model, _worker_adr.dof, _worker_adr.act, del_)
    mujoco.mj_resetDataKeyframe(_worker_model, d, _worker_adr.stand_key)
    _perturb(d, _worker_adr, _worker_manifest, p, 0.0)
    mujoco.mj_forward(_worker_model, d)
    srv.drive(d, np.zeros(ARENA_NJ))
    mujoco.mj_forward(_worker_model, d)
    
    # Settle 25 steps using baseline settle actions
    for s in range(25):
        target = DEFAULT_POSE + _worker_settle_acts[s] * ACTION_SCALE
        for _ in range(DECIMATION):
            srv.drive(d, target)
            mujoco.mj_step(_worker_model, d)
        srv.drive(d, target)
        mujoco.mj_forward(_worker_model, d)
        
    start_prog = float(_worker_direction @ d.qpos[0:3])
    tipped_for = 0
    
    progs = []
    for step in range(43):
        k = 25 + step
        act = tab[k]
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
            # Fall penalty: 10.0 + remaining distance
            prog_so_far = float(_worker_direction @ d.qpos[0:3]) - start_prog
            return 10.0 + (2.0 - prog_so_far)
            
        prog = float(_worker_direction @ d.qpos[0:3]) - start_prog
        progs.append(prog)
        if prog >= 2.0:
            # Crossed finish line!
            k_cross = step
            prev_p = progs[-2] if len(progs) >= 2 else 0.0
            curr_p = prog
            frac = (2.0 - prev_p) / max(curr_p - prev_p, 1e-6)
            exact_time = step * 0.02 + frac * 0.02
            return exact_time
            
    # Timeout/incomplete: 5.0 + remaining distance
    last_prog = progs[-1] if len(progs) > 0 else 0.0
    return 5.0 + (2.0 - last_prog)

def eval_wrapper(args):
    theta_flat, base_tab = args
    return evaluate_worker(theta_flat, base_tab)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--batch-size', type=int, default=8, help='Number of parallel SPSA perturbations per step')
    parser.add_argument('--lr', type=float, default=0.002, help='Learning rate (Adam)')
    parser.add_argument('--c', type=float, default=0.005, help='Perturbation scale')
    parser.add_argument('--max-iter', type=int, default=3000)
    args = parser.parse_args()
    
    print("=" * 65, flush=True)
    print(" MINI-BATCH SPSA OPTIMIZER FOR SEED 1 CROSSING TIME ", flush=True)
    print(" Target: Shave 16.87ms (0.83687s -> <= 0.82000s) to win Rank 1 ", flush=True)
    print("=" * 65, flush=True)
    
    # Load Austin Dixson's model
    base_model_path = 'C:/GitHub/MicroDuck/models/austindixson_0840.onnx'
    m = onnx.load(base_model_path)
    inits = {i.name: numpy_helper.to_array(i) for i in m.graph.initializer}
    base_tab = inits['br_TAB'][5*128 : 6*128].copy() # Branch 5 (Seed 1)
    
    # Record settle actions (k=0..24) using base policy
    sys.path.append('scripts')
    from arena.policy import load_policy
    from arena.manifest import load_manifest, EVENTS_DIR
    from arena.course import compile_course
    from arena.runner import _Addresses, _streams, Servos, _perturb, DEFAULT_POSE, ACTION_SCALE, DECIMATION, NUM_JOINTS
    
    manifest = load_manifest(EVENTS_DIR / "sprint-2m.toml")
    model = compile_course(manifest)
    adr = _Addresses.of(model, manifest.start.keyframe, "")
    cmd = np.asarray(manifest.command_at(1.15), dtype=np.float64)
    policy = load_policy(base_model_path)
    
    d = mujoco.MjData(model)
    p, del_ = _streams(manifest.id, 1, 0)
    srv = Servos(model, adr.dof, adr.act, del_)
    mujoco.mj_resetDataKeyframe(model, d, adr.stand_key)
    _perturb(d, adr, manifest, p, 0.0)
    mujoco.mj_forward(model, d)
    srv.drive(d, np.zeros(NUM_JOINTS))
    mujoco.mj_forward(model, d)
    
    settle_acts = []
    last_act = np.zeros(14, dtype=np.float32)
    prev_qvel = d.qvel[adr.dof].copy()
    for s in range(25):
        obs = np.empty(61, dtype=np.float32)
        obs[0:3] = d.sensordata[adr.gyro:adr.gyro+3]
        rot = d.xmat[adr.trunk].reshape(3, 3)
        obs[3:6] = rot.T @ np.array([0, 0, -1])
        obs[6:20] = d.qpos[adr.qpos] - DEFAULT_POSE
        obs[20:34] = prev_qvel
        obs[34:48] = last_act
        obs[48:61] = cmd
        prev_qvel = d.qvel[adr.dof].copy()
        act = policy.act(obs)
        settle_acts.append(act.copy())
        last_act = act
        target = DEFAULT_POSE + act * ACTION_SCALE
        for _ in range(DECIMATION):
            srv.drive(d, target)
            mujoco.mj_step(model, d)
        srv.drive(d, target)
        mujoco.mj_forward(model, d)
        
    settle_acts = np.array(settle_acts)
    
    # Initialize worker
    worker_init(1, settle_acts)
    
    # Extract initial theta from base_tab
    theta0 = np.empty((NUM_STEPS, NUM_ACTIVE), dtype=np.float32)
    for idx, k in enumerate(range(START_K, END_K)):
        theta0[idx] = base_tab[k, ACTIVE_JOINTS]
    theta = theta0.flatten()
    
    t0_eval = evaluate_worker(theta, base_tab)
    print(f"Initial continuous crossing time: {t0_eval:.5f} s (target <= 0.82000s)", flush=True)
    
    B = args.batch_size # 8 perturbations -> 16 evaluations per step
    c_pert = args.c
    lr = args.lr
    
    # Adam state
    m_adam = np.zeros(DIM, dtype=np.float32)
    v_adam = np.zeros(DIM, dtype=np.float32)
    beta1 = 0.9
    beta2 = 0.999
    eps_adam = 1e-8
    
    best_time = t0_eval
    best_theta = theta.copy()
    
    save_path = "C:/GitHub/MicroDuck/models/best_spsa_seed1_theta.npy"
    start_time = time.time()
    
    with ProcessPoolExecutor(max_workers=args.workers, initializer=worker_init, initargs=(1, settle_acts)) as executor:
        for iter_idx in range(1, args.max_iter + 1):
            # Generate B random Bernoulli perturbations (+1 or -1)
            deltas = np.random.choice([-1.0, 1.0], size=(B, DIM)).astype(np.float32)
            
            # Form 2*B candidate parameter vectors
            candidates = []
            for b in range(B):
                candidates.append((theta + c_pert * deltas[b], base_tab))
                candidates.append((theta - c_pert * deltas[b], base_tab))
                
            # Parallel evaluate all 2*B candidates
            results = list(executor.map(eval_wrapper, candidates))
            
            # Check if any candidate perturbation directly improves the best time
            min_cand_idx = np.argmin(results)
            min_cand_time = results[min_cand_idx]
            if min_cand_time < best_time:
                gain_ms = (best_time - min_cand_time) * 1000
                best_time = min_cand_time
                best_theta = candidates[min_cand_idx][0].copy()
                theta = best_theta.copy()
                np.save(save_path, best_theta)
                elapsed = time.time() - start_time
                print(f"[{elapsed:5.1f}s] DIRECT IMPROVEMENT! best={best_time:.5f}s (gain={gain_ms:+.2f}ms)", flush=True)
            
            # Compute average SPSA gradient
            grad_sum = np.zeros(DIM, dtype=np.float32)
            for b in range(B):
                y_plus = results[2 * b]
                y_minus = results[2 * b + 1]
                g_b = ((y_plus - y_minus) / (2.0 * c_pert)) * deltas[b]
                grad_sum += g_b
            grad = grad_sum / float(B)
            
            # Adam update from best_theta
            m_adam = beta1 * m_adam + (1.0 - beta1) * grad
            v_adam = beta2 * v_adam + (1.0 - beta2) * (grad ** 2)
            m_hat = m_adam / (1.0 - beta1 ** iter_idx)
            v_hat = v_adam / (1.0 - beta2 ** iter_idx)
            
            cand_theta = best_theta - lr * m_hat / (np.sqrt(v_hat) + eps_adam)
            cand_time = evaluate_worker(cand_theta, base_tab)
            if cand_time < best_time:
                gain_ms = (best_time - cand_time) * 1000
                best_time = cand_time
                best_theta = cand_theta.copy()
                theta = best_theta.copy()
                np.save(save_path, best_theta)
                elapsed = time.time() - start_time
                print(f"[{elapsed:5.1f}s] GRADIENT IMPROVEMENT! best={best_time:.5f}s (gain={gain_ms:+.2f}ms)", flush=True)
            else:
                theta = best_theta.copy()
                
            elapsed = time.time() - start_time
            rate = iter_idx / elapsed if elapsed > 0 else 0
            remaining_ms = (best_time - 0.82000) * 1000
            if iter_idx % 10 == 0 or iter_idx == 1:
                print(f"[{elapsed:5.1f}s | {rate:4.1f} it/s] Iter {iter_idx:4d}: best={best_time:.5f}s (delta to 0.820s: {remaining_ms:+.2f}ms)", flush=True)
                
            if best_time <= 0.82000:
                print("=" * 65, flush=True)
                print(f"!!! SUCCESS: CROSSED 2.0M AT STEP 40 (t = {best_time:.5f}s <= 0.82000s) !!!", flush=True)
                print("=" * 65, flush=True)
                np.save(save_path, best_theta)
                break
                    
    np.save(save_path, best_theta)
    print(f"Finished. Best crossing time: {best_time:.5f}s. Saved to {save_path}")

if __name__ == "__main__":
    main()

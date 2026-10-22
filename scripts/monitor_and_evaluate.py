"""
MicroDuck Sprint-2m Autonomous Monitor & Benchmark Evaluator.
Tracks GPU training progress, detects new checkpoints, evaluates them
using the deterministic official Arena scoring suite across 5 seeds,
and tracks progress toward taking #1 on the leaderboard.
"""

import os
import sys
import glob
import json
import time
import subprocess
from pathlib import Path

MICRODUCK_DIR = Path("C:/GitHub/MicroDuck")
CHALLENGES_DIR = Path("C:/GitHub/microduck_challenges")
RUNS_DIR = CHALLENGES_DIR / "logs" / "rsl_rl" / "sprint_2m"
PYTHON_EXE = CHALLENGES_DIR / ".venv" / "Scripts" / "python.exe"
LOG_RECORD = MICRODUCK_DIR / "logs" / "training_eval_history.json"

LOG_RECORD.parent.mkdir(parents=True, exist_ok=True)

def get_latest_run_dir():
    run_dirs = sorted(glob.glob(str(RUNS_DIR / "*sprint_run1*")), key=os.path.getmtime)
    if not run_dirs:
        # fallback to any run
        run_dirs = sorted(glob.glob(str(RUNS_DIR / "*")), key=os.path.getmtime)
    return Path(run_dirs[-1]) if run_dirs else None

def get_training_progress(run_dir):
    log_file = None
    # find log file for this run
    for p in [run_dir / "train.log", CHALLENGES_DIR / "logs" / "sprint_run1.log"]:
        if p.exists():
            log_file = p
            break
    
    # Or inspect background task log / stdout if any
    task_logs = sorted(glob.glob(r"C:\Users\elect\.gemini\antigravity-ide\brain\*\.system_generated\tasks\task-*.log"), key=os.path.getmtime)
    for tl in reversed(task_logs):
        try:
            with open(tl, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
                if "sprint_run1" in content or "Mjlab-Sprint2m-MicroDuck" in content:
                    log_file = Path(tl)
                    break
        except Exception:
            pass

    if not log_file or not log_file.exists():
        return {"status": "running", "iteration": 0, "fps": 0, "reward": 0.0}

    with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    last_iter = 0
    fps = 0
    reward = 0.0
    ep_len = 0.0
    for line in lines[-200:]:
        if "Learning iteration" in line:
            parts = line.strip().split()
            for p in parts:
                if "/" in p:
                    try:
                        last_iter = int(p.split("/")[0])
                    except ValueError:
                        pass
        elif "Steps per second:" in line:
            try:
                fps = int(line.split(":")[-1].strip())
            except ValueError:
                pass
        elif "Mean reward:" in line:
            try:
                reward = float(line.split(":")[-1].strip())
            except ValueError:
                pass
        elif "Mean episode length:" in line:
            try:
                ep_len = float(line.split(":")[-1].strip())
            except ValueError:
                pass

    return {
        "status": "running",
        "iteration": last_iter,
        "fps": fps,
        "mean_reward": reward,
        "mean_ep_len": ep_len,
        "log_path": str(log_file)
    }

def get_checkpoints(run_dir):
    if not run_dir or not run_dir.exists():
        return []
    pts = glob.glob(str(run_dir / "model_*.pt"))
    checkpoints = []
    for p in pts:
        fname = Path(p).stem
        try:
            num = int(fname.split("_")[-1])
            checkpoints.append((num, Path(p)))
        except ValueError:
            pass
    return sorted(checkpoints, key=lambda x: x[0])

def evaluate_checkpoint(run_dir, ckpt_num):
    print(f"--> Evaluating Checkpoint {ckpt_num} from {run_dir.name}...")
    dry_run_dir = MICRODUCK_DIR / "eval_tmp" / f"ckpt_{ckpt_num}"
    dry_run_dir.mkdir(parents=True, exist_ok=True)
    
    publish_exe = CHALLENGES_DIR / ".venv" / "Scripts" / "publish.exe"
    arena_exe = CHALLENGES_DIR / ".venv" / "Scripts" / "arena.exe"
    
    # Run publish --dry-run
    cmd_export = [
        str(publish_exe),
        "--run", str(run_dir),
        "--checkpoint", str(ckpt_num),
        "--repo", "local/eval",
        "--dry-run",
        "--allow-dirty"
    ]
    res_export = subprocess.run(cmd_export, cwd=str(CHALLENGES_DIR), capture_output=True, text=True)
    
    # Locate exported policy.onnx in publish-eval/
    policy_onnx = CHALLENGES_DIR / "publish-eval" / "policy.onnx"
    if not policy_onnx.exists():
        return {"error": "ONNX export failed", "details": res_export.stderr or res_export.stdout}

    # Run arena score
    cmd_score = [
        str(arena_exe), "score",
        str(policy_onnx),
        "--event", "sprint-2m"
    ]
    res_score = subprocess.run(cmd_score, cwd=str(CHALLENGES_DIR), capture_output=True, text=True)
    score_out = res_score.stdout
    
    # Parse score
    score_val = None
    seeds_fin = None
    best_vx = None
    for l in score_out.splitlines():
        if "score " in l:
            try:
                score_val = float(l.split()[1])
            except Exception:
                pass
        if "scored at" in l:
            try:
                best_vx = float(l.split()[2])
            except Exception:
                pass
        if "seeds finished:" in l:
            try:
                seeds_fin = int(l.split(":")[-1].strip().split("/")[0])
            except Exception:
                pass

    return {
        "checkpoint": ckpt_num,
        "score_s": score_val,
        "command_vx": best_vx,
        "seeds_finished": seeds_fin,
        "raw": score_out
    }

def main():
    run_dir = get_latest_run_dir()
    progress = get_training_progress(run_dir)
    print("=" * 60)
    print(" MICRODUCK SPRINT-2M TRAINING & EVALUATION STATUS ")
    print("=" * 60)
    print(f"Active Run: {run_dir.name if run_dir else 'None'}")
    print(f"Iteration:  {progress.get('iteration', 0)} / 3000")
    print(f"Throughput: {progress.get('fps', 0):,} steps/sec (GPU cuda:0)")
    print(f"Mean Rew:   {progress.get('mean_reward', 0.0):.3f}")
    print(f"Mean EpLen: {progress.get('mean_ep_len', 0.0):.1f} steps")
    print("-" * 60)

    # Load evaluated history
    history = {}
    if LOG_RECORD.exists():
        try:
            with open(LOG_RECORD, "r") as f:
                history = json.load(f)
        except Exception:
            history = {}

    ckpts = get_checkpoints(run_dir)
    print(f"Checkpoints saved so far: {[c[0] for c in ckpts]}")

    for num, path in ckpts:
        key = f"{run_dir.name}_ckpt_{num}"
        if key not in history:
            eval_res = evaluate_checkpoint(run_dir, num)
            history[key] = eval_res
            with open(LOG_RECORD, "w") as f:
                json.dump(history, f, indent=2)
            print(f"Checkpoint {num} Result: Score={eval_res.get('score_s')}s, Speed={eval_res.get('command_vx')}m/s, Finished={eval_res.get('seeds_finished')}/5")
        else:
            prev = history[key]
            print(f"Checkpoint {num} (Cached): Score={prev.get('score_s')}s, Speed={prev.get('command_vx')}m/s, Finished={prev.get('seeds_finished')}/5")

    print("=" * 60)

if __name__ == "__main__":
    main()

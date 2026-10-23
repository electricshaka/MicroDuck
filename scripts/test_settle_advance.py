"""
Test advancing the settle launch phase to increase exit velocity and cross on step 40 (0.820s).
"""

import sys
sys.path.append('scripts')
import numpy as np
import mujoco
from arena.manifest import load_manifest, EVENTS_DIR
from arena.runner import (_Addresses, _streams, Servos, _perturb, DEFAULT_POSE, ACTION_SCALE, DECIMATION, NUM_JOINTS, ramp_dir)
from arena.course import compile_course
from optimize_seed_table import record_seed, sim_actions

manifest = load_manifest(EVENTS_DIR / 'sprint-2m.toml')
model = compile_course(manifest)
adr = _Addresses.of(model, manifest.start.keyframe, '')
direction = ramp_dir(manifest.course.slope_deg)

# Let's inspect Seed 4 baseline
base_acts = record_seed(4)

def test_settle_advance(shift_steps):
    """
    Shifts the launch sequence earlier by shift_steps during the 25 settle steps.
    """
    test_acts = base_acts.copy()
    if shift_steps > 0:
        # Move settle actions earlier: acts[k] comes from acts[k + shift]
        # For settle steps 5..24
        # We compress or shift steps 5..24
        orig_settle = base_acts[:25].copy()
        # Interpolate 25 steps over slightly accelerated time
        t_orig = np.linspace(0, 1, 25)
        # Shift launch: accelerate the second half
        t_new = np.linspace(0, 1 + shift_steps/25.0, 25)
        t_new = np.clip(t_new, 0, 1)
        for j in range(14):
            test_acts[:25, j] = np.interp(t_new, t_orig, orig_settle[:, j])
            
    st, step, prog, t = sim_actions(4, test_acts)
    return st, step, prog, t

print("Testing settle shift on Seed 4:")
for s in [0, 0.5, 1.0, 1.5, 2.0]:
    st, step, prog, t = test_settle_advance(s)
    print(f"Shift {s}: status={st}, step={step}, t={t}s, prog={prog:.4f}m")

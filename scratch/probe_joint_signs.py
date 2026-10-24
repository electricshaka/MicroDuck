"""Probe: which joint, and which sign, lifts a foot off the floor — to pick the mid-step pose
values for rewards.spawn_toprock. Prints foot geom world z for the default STAND pose, then one
joint perturbed at a time (both signs)."""
import importlib
import sys

import mujoco
import numpy as np

sys.path.insert(0, r"C:\GitHub\microduck-hs11b\tests")
import microduck_challenges  # noqa: F401  (registers tasks)
from arena.constants import JOINT_NAMES

from test_got_a_talent import _training_model

cfg = importlib.import_module("microduck_challenges.got_a_talent.env").make_env_cfg()
model = _training_model(cfg.scene.entities["robot"])
data = mujoco.MjData(model)

qa = {n: int(model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)])
      for n in JOINT_NAMES}
free = int(model.jnt_qposadr[model.body_jntadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")]])


def reset_standing(z=0.115):
    mujoco.mj_setConst = None
    data.qpos[:] = 0.0
    data.qpos[free + 2] = z
    data.qpos[free + 3] = 1.0  # upright quaternion (w,x,y,z)
    for i, n in enumerate(JOINT_NAMES):  # DEFAULT_POSE is all-zeros offset from keyframe default
        data.qpos[qa[n]] = 0.0
    mujoco.mj_forward(model, data)


def feet_z():
    out = []
    for side in ("left", "right"):
        g = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"{side}_foot_collision")
        out.append(round(float(data.geom_xpos[g, 2]), 4))
    return out


reset_standing()
base = feet_z()
print("standing feet z (L,R):", base, "trunk qpos z:", round(float(data.qpos[free + 2]), 4))
for name in ("left_knee", "left_hip_pitch", "left_ankle", "left_hip_roll",
             "right_knee", "right_hip_pitch", "right_ankle", "right_hip_roll"):
    for d in (0.3, 0.6, -0.3, -0.6):
        reset_standing()
        data.qpos[qa[name]] += d
        mujoco.mj_forward(model, data)
        print(f"{name} {'+' if d > 0 else '-'}{abs(d):.1f}: feet z (L,R) {feet_z()}")

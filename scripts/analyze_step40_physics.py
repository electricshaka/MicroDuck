"""
Deep Physics Analysis of MicroDuck at Step 40 (0.820s) across all 5 seeds.
Examines position, velocity, kinetic energy, center of mass, and contact state.
"""

import mujoco
from arena.manifest import load_manifest, EVENTS_DIR
from arena.course import compile_course
from arena.runner import (_Addresses, _streams, Servos, _perturb, DEFAULT_POSE, ACTION_SCALE, DECIMATION, NUM_JOINTS, ramp_dir)
from arena.policy import load_policy
import numpy as np

manifest = load_manifest(EVENTS_DIR / "sprint-2m.toml")
model = compile_course(manifest)
adr = _Addresses.of(model, manifest.start.keyframe, "")
direction = ramp_dir(manifest.course.slope_deg)
command = np.asarray(manifest.command_at(1.15), dtype=np.float64)
policy = load_policy("C:/GitHub/MicroDuck/models/leader_jrubiosainz_0840.onnx")

print("=" * 75)
print(" SEED COMPARISON AT STEP 40 (0.820s)")
print("=" * 75)
print("Seed | Prog (m) | Dist to 2m | Fwd Vel (m/s) | Vert Vel | Pitch (deg) | Subtree CoM X")
print("-----+----------+------------+---------------+----------+-------------+--------------")

for seed in [1, 2, 3, 4, 5]:
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

    for s in range(25):
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
        last_act = act
        target = DEFAULT_POSE + act * ACTION_SCALE
        for _ in range(DECIMATION):
            srv.drive(d, target)
            mujoco.mj_step(model, d)
        srv.drive(d, target)
        mujoco.mj_forward(model, d)

    start_prog = float(direction @ d.qpos[0:3])

    for step in range(41):
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
        last_act = act
        target = DEFAULT_POSE + act * ACTION_SCALE
        for _ in range(DECIMATION):
            srv.drive(d, target)
            mujoco.mj_step(model, d)
        srv.drive(d, target)
        mujoco.mj_forward(model, d)

    prog = float(direction @ d.qpos[0:3]) - start_prog
    dist = 2.0 - prog
    fwd_vel = float(direction @ d.qvel[0:3])
    vert_vel = d.qvel[2]
    rot = d.xmat[adr.trunk].reshape(3, 3)
    pitch = np.degrees(np.arcsin(-rot[2, 0]))
    com_x = d.subtree_com[1][0] - d.qpos[0]
    print(f" {seed:2d}  | {prog:8.5f} | {dist*100:8.2f}cm | {fwd_vel:13.3f} | {vert_vel:8.3f} | {pitch:11.2f} | {com_x*100:10.2f}cm")

"""hs13 overhaul measurements (1): the KNEE BOUNCE and the CROSS-STEP, in static poses.

The user's Phase-1 fix wants the 0-2 s opening to be a live 108 BPM knee bounce (108 BPM = 0.555 s
per beat) with alternating cross-steps riding on it. bounce_probe.py showed the timeline alone is
FLAT (the sit flag cannot oscillate at 1.8 Hz), so this must be a trained regime — and the regime
needs MEASURED numbers, the way every previous sprint measured before writing rewards:

  * the bounce TROUGH: symmetric knee flexion (left knee +, right knee −, the lift signs of
    probe_joint_signs.py) with the feet kept at floor level — how far does the trunk DROP below
    the 0.115 m stand? how upright stays the trunk (does hip_pitch have to compensate)?
    the bar: a visible drop (>= 0.015 m) with feet planted and trunk still near vertical, and the
    trough must sit ABOVE the crouch floor (z >= 0.085) so it reads as a bounce, not a sit.
  * the CROSS-STEP reach from a trough: one hip_pitch forward (left NEGATIVE / right POSITIVE,
    the downrock signs), the other planted — where does the stepping foot go (forward reach, off
    the floor?) and does the planted foot stay down.

For each pose: root z is lowered until the LOWER foot site reaches ~0.005 m (contact), and the
report reads trunk z (the drop from stand), projected gravity (upright check) and both foot sites.

    uv run --no-sync python scratch/bounce_env_probe.py   (CPU only, a couple of minutes)
"""
import sys

sys.path.insert(0, r"C:\GitHub\microduck-private\src")

import torch


def main() -> None:
    from microduck_challenges.got_a_talent import tasks  # noqa: F401
    from microduck_challenges.got_a_talent.env import make_env_cfg
    from mjlab.envs import ManagerBasedRlEnv

    cfg = make_env_cfg()
    cfg.scene.num_envs = 4
    cfg.sim.device = "cpu"
    env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
    env.reset()

    robot = env.scene["robot"]
    d = robot.data
    site_names = list(robot.site_names)
    lf, rf = site_names.index("left_foot"), site_names.index("right_foot")

    home = d.default_joint_pos[0].clone()
    cols = torch.tensor(list(range(7, 21)), device="cpu")
    ids = torch.tensor([0], device="cpu", dtype=torch.long)
    print(f"HOME stand: kneeL {float(home[3]):+.3f} kneeR {float(home[12]):+.3f} "
          f"hipL {float(home[2]):+.3f} hipR {float(home[11]):+.3f} "
          f"ankL {float(home[4]):+.3f} ankR {float(home[13]):+.3f}", flush=True)

    def pose_at(tag, offsets, root_z):
        """offsets: dict servo-index -> absolute joint value. Sets joints + root z, forwards, reads."""
        p = home.clone()
        for j, v in offsets.items():
            p[j] = v
        env.sim.data.qpos[ids.unsqueeze(1).long(), cols.unsqueeze(0)] = p.unsqueeze(0)
        env.sim.data.qpos[0, 2] = root_z
        env.sim.forward()
        z_tr = float(env.sim.data.qpos[0, 2])
        grav = d.projected_gravity_b[0]
        pos = d.site_pos_w[0]
        lfz, rfz = float(pos[lf][2]), float(pos[rf][2])
        lfx, rfx = float(pos[lf][0]), float(pos[rf][0])
        print(f"{tag:56s} z {z_tr:.3f} gz {float(grav[2]):+.2f} "
              f"footL z {lfz:+.3f} x {lfx:+.3f} | footR z {rfz:+.3f} x {rfx:+.3f}", flush=True)
        return z_tr

    # Stand reference at the spawn height (feet on the floor there).
    z0 = pose_at("STAND home pose @ z 0.115", {}, 0.115)

    # Knee-flexion troughs: feet planted means the ROOT drops about as much as the knees bend;
    # we search root z so the LOWER foot site sits at ~0.005, then report the drop.
    for knee in (0.20, 0.35, 0.50):
        for hipc in (0.0, -0.15, -0.30):  # hip_pitch compensation: L negative flexes L forward
            off = {3: float(home[3]) + knee, 12: float(home[12]) - knee}
            for hip_side, j in ((hipc, 2), (hipc, 11)):
                pass
            off[2] = float(home[2]) + hipc
            off[11] = float(home[11]) - hipc
            # root z scan: where is the lower foot at stand height? lower until it touches.
            pose_at(f"TROUGH knee {knee:+.2f} hipc {hipc:+.2f} @ stand z", off, 0.115)
            for z_try in (0.105, 0.100, 0.095, 0.090, 0.085):
                pose_at(f"    same pose @ z {z_try:.3f}", off, z_try)

    # Cross-step FROM a moderate trough (knee 0.35, hipc -0.15, z 0.095): step one hip forward.
    base = {3: float(home[3]) + 0.35, 12: float(home[12]) - 0.35,
            2: float(home[2]) - 0.15, 11: float(home[11]) + 0.15}
    for step in (0.25, 0.40, 0.55):
        off = dict(base)
        off[2] = float(home[2]) - 0.15 - step   # LEFT hip forward (negative, the downrock sign)
        off[3] = float(home[3]) + 0.15          # kick-out: straighter knee on the stepping leg
        pose_at(f"STEP-LEFT {step:.2f} fwd from trough @ z 0.095", off, 0.095)
        off = dict(base)
        off[11] = float(home[11]) + 0.15 + step  # RIGHT hip forward (positive)
        off[12] = float(home[12]) - 0.15
        pose_at(f"STEP-RIGHT {step:.2f} fwd from trough @ z 0.095", off, 0.095)


if __name__ == "__main__":
    main()

"""hs12b sign hunt (round 2): where must the legs be (absolute joint values, 14-servo order) for
an inverted L-kick — one leg a vertical needle, the other horizontal-forward?

Round 1 established: the feet are SITES (left_foot, right_foot); hip_pitch home is -0.458 (left)
/ +0.458 (right), knees ~0 (HOME legs are straight), hip limits soft ±1.414 / hard ±1.571; and
(downrock_leg's own signs) forward for the LEFT leg is NEGATIVE hip_pitch, mirrored right.
Upside down, body -z is the sky and body +x is still world-forward, so the guess is needle = home
and kick = ±1.4ish straight-leg toward the soft limit. Here we SET each candidate on the
headstand spawn, forward(), and read where the foot sites land: dz over the trunk (needle wants
+0.10ish, kick wants ~0) and dx ahead (kick wants +, needle ~0).

    uv run --no-sync python scratch/kick_env_probe.py   (CPU only, a couple of minutes)
"""
import sys

sys.path.insert(0, r"C:\GitHub\microduck-private\src")

import torch


def main() -> None:
    from microduck_challenges.got_a_talent import rewards as R
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
    hd = site_names.index("head_camera")
    pos_attr = "site_pos_w" if hasattr(d, "site_pos_w") else None
    if pos_attr is None:
        print("no site_pos_w; attributes:", [a for a in dir(d) if "pos" in a])
        sys.exit(1)

    home = d.default_joint_pos[0].clone()

    # Actually put env 0 on its head (inverts the root, sets the spindle neck/head, z 0.15).
    R.spawn_headstand(env, torch.tensor([0], device="cpu"), prob=1.0, yaw_rate_max=0.0)
    env.sim.forward()

    def report(tag, pose):
        cols = torch.tensor(list(range(7, 21)), device="cpu")
        ids = torch.tensor([0], device="cpu", dtype=torch.long)
        env.sim.data.qpos[ids.unsqueeze(1).long(), cols.unsqueeze(0)] = pose.unsqueeze(0)
        env.sim.forward()
        z_tr, x_tr, y_tr = (float(env.sim.data.qpos[0, 2]), float(env.sim.data.qpos[0, 0]),
                            float(env.sim.data.qpos[0, 1]))
        pos = getattr(d, pos_attr)[0]
        grav = d.projected_gravity_b[0]  # upright ~ (0,0,-1); truly inverted ~ (0,0,+1)
        out = []
        for i, nm in ((lf, "L"), (rf, "R"), (hd, "head")):
            p = pos[i]
            out.append(f"{nm} dz {float(p[2]) - z_tr:+.3f} dx {float(p[0]) - x_tr:+.3f}")
        hip_l = float(env.sim.data.qpos[0, 7 + 2])
        hip_r = float(env.sim.data.qpos[0, 7 + 11])
        print(f"{tag:52s} gz {float(grav[2]):+.2f} hipL {hip_l:+.2f} hipR {hip_r:+.2f}   "
              + "  ".join(out), flush=True)

    base = home.clone()
    base[5] = R.HEADSTAND_NECK_PITCH
    base[6] = R.HEADSTAND_HEAD_PITCH
    report("HOME legs, inverted (verify)", base)

    # Needle = thigh straight up the body axis. Inverted, that is hips toward 0 (home is ~45 deg
    # off-axis, so the HOME foot hangs BELOW the trunk — wrong for the needle). Try hips 0.
    p = base.clone()
    p[2] = 0.0
    p[3] = 0.0
    p[11] = 0.0
    p[12] = 0.0
    report("BOTH hips 0 (thighs up the body axis)", p)
    p = base.clone()
    p[2] = 0.0
    p[3] = -0.35   # knee straight-ish left (servo sign for straight)
    report("LEFT hip 0 knee -0.35 (single needle up), right home", p)

    # Kick candidates: try BOTH hip signs per leg (hs12's "left forward = negative" was learned
    # UPRIGHT; the robot's hips are mounted mirrored, so the inverted forward swing may differ by
    # side). A good kick foot is LOW over the trunk (dz ~0, horizontal thigh) and FORWARD (dx >0).
    for kick in (1.15, 1.30, 1.42):
        for sign in (-1.0, 1.0):
            p = base.clone()
            p[2] = sign * kick
            p[3] = 0.0
            report(f"KICK LEFT hip {sign * kick:+.2f} knee 0 (right home)", p)
            p = base.clone()
            p[11] = sign * kick
            p[12] = 0.0
            report(f"KICK RIGHT hip {sign * kick:+.2f} knee 0 (left home)", p)

    # Straight knee candidates on the forward leg once we know which hip sign reaches forward.
    p = base.clone()
    p[2], p[3] = 1.42, 0.0
    p[11], p[12] = 1.42, 0.0
    report("hips +1.42 both, knees 0 both", p)
    p = base.clone()
    p[2], p[3] = -1.42, 0.0
    p[11], p[12] = -1.42, 0.0
    report("hips -1.42 both, knees 0 both", p)


if __name__ == "__main__":
    main()

"""hs12c sweep-shape measurements (before ANY reward edit — measured, not guessed):

The hs12 sweep earned 837 deg/crouch with the pelvis FLAT at z 0.061 (butt-spin). hs12c must pay
an authentic coffee grinder instead: weight on a planted knee/foot, trunk z >= 0.085, the sweeping
leg a blade (hip ~1.2, knee 0) sweeping a WIDE perimeter. This probe SETS poses on the live env
(like kick_env_probe did for the freeze) and reads:
  - trunk z after forward();
  - contact found-flags: PELVIS geoms (trunk_base|hip_l|hip_l_2 — new sensor pattern, verified
    here before it goes into env.py), leg/ankle (body_ground), feet;
  - the blade foot's world radius around the trunk's vertical axis (the perimeter it sweeps).

    uv run --no-sync python scratch/sweep_env_probe.py     (CPU only, a couple of minutes)
"""
import sys

sys.path.insert(0, r"C:\GitHub\microduck-private\src")

import torch


def main() -> None:
    from microduck_challenges.got_a_talent import rewards as R
    from microduck_challenges.got_a_talent import tasks  # noqa: F401
    from microduck_challenges.got_a_talent.env import make_env_cfg
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.sensor import ContactMatch, ContactSensorCfg

    cfg = make_env_cfg()
    cfg.scene.num_envs = 4
    cfg.sim.device = "cpu"
    pelvis_cfg = ContactSensorCfg(
        name="pelvis_ground",
        primary=ContactMatch(
            mode="body", pattern=r"^(trunk_base|hip_l|hip_l_2)$", entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found",),
        reduce="netforce",
        num_slots=1,
    )
    cfg.scene.sensors = (*cfg.scene.sensors, pelvis_cfg)
    env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
    env.reset()

    robot = env.scene["robot"]
    d = robot.data
    site_names = list(robot.site_names)
    lf, rf = site_names.index("left_foot"), site_names.index("right_foot")

    def found(name):
        s = env.scene[name]
        f = s.data.found
        return float(f.reshape(len(f), -1)[0].max()) if f is not None else -1.0

    home = d.default_joint_pos[0].clone()

    def report(tag, pose, z):
        ids = torch.tensor([0], device="cpu", dtype=torch.long)
        cols = torch.tensor(list(range(7, 21)), device="cpu")
        env.sim.data.qpos[ids.unsqueeze(1).long(), cols.unsqueeze(0)] = pose.unsqueeze(0)
        env.sim.data.qpos[0, 2] = z  # + env origin z (origin 0 in this probe pattern)
        # FORCE UPRIGHT: this env's reset may have seeded a headstand spawn (gz read +1.00 in the
        # first run — the contacts below would be measured on a floating, inverted duck).
        env.sim.data.qpos[0, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0])
        env.sim.data.qvel[0] = 0.0
        env.sim.forward()
        z_tr = float(env.sim.data.qpos[0, 2])
        x_tr, y_tr = float(env.sim.data.qpos[0, 0]), float(env.sim.data.qpos[0, 1])
        pos = d.site_pos_w[0]
        grav = d.projected_gravity_b[0]
        feet = []
        for i, nm in ((lf, "L"), (rf, "R")):
            p = pos[i]
            rad = ((float(p[0]) - x_tr) ** 2 + (float(p[1]) - y_tr) ** 2) ** 0.5
            feet.append(f"{nm} rad {rad:.3f} dz {float(p[2]) - z_tr:+.3f}")
        print(f"{tag:46s} z {z_tr:.3f} gz {float(grav[2]):+.2f}  "
              f"pelv {found('pelvis_ground'):.0f} body {found('body_ground'):.0f} "
              f"feet {found('feet_ground_contact'):.0f}   " + "  ".join(feet), flush=True)

    def crouch():
        p = home.clone()
        for j, v in R.SWEEP_SIT_POSE.items():
            p[j] = v
        return p

    # A. The hs12 reality: flat crouch at its rest height — expect PELVIS contact (the butt
    #    platform the user named) and no foot radius.
    report("A crouch z 0.061 (hs12 butt-spin state)", crouch(), 0.061)
    report("A' crouch z 0.085 (flat butt, lifted trunk)", crouch(), 0.085)

    # B. Kneel + blade, swept leg LEFT (forward = negative hip, downrock_leg's sign) then RIGHT.
    #    Support leg stays the folded crouch leg (the planted knee).
    for side in ("left", "right"):
        for z in (0.085, 0.095, 0.105):
            p = crouch()
            if side == "left":
                p[2], p[3] = -1.2, 0.0          # blade left; support = right crouch leg
            else:
                p[11], p[12] = 1.2, 0.0         # blade right; support = left crouch leg
            report(f"B {side} blade hip {1.2 * (-1 if side == 'left' else 1):+.1f} knee 0", p, z)

    # C. Blade hip candidates at the likely spawn height (how much hip does a knee-support pose need?)
    for hip in (0.8, 1.0, 1.4):
        p = crouch()
        p[2], p[3] = -hip, 0.0
        report(f"C left blade hip {-hip:+.1f} knee 0 z 0.095", p, 0.095)

    # D. Blade with BOTH knees straight and hips home (skate rest) — does a knee rest exist?
    p = crouch()
    p[2], p[11] = -0.408, 0.408                  # crouch hips stay; uncurl both knees
    report("D both knees 0, crouch hips (long legs) z 0.095", p, 0.095)
    report("D' both knees 0, crouch hips z 0.115", p, 0.115)


if __name__ == "__main__":
    main()

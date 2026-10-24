"""Direct check that spawn_toprock lands: build a small CPU env, reset until a mid-step spawn
lands, confirm the toprock flag is up, then hold the default action and watch rewards.toprock_step
pay as the raised foot comes down (called once per step, exactly like the reward manager does).

    uv run --no-sync python scratch/spawn_env_test.py   (CPU only, a few minutes)
"""
import sys

sys.path.insert(0, r"C:\GitHub\microduck-hs11b\src")

import torch

import microduck_challenges  # noqa: F401  (registers tasks)


def main() -> None:
    from microduck_challenges.got_a_talent import rewards as R
    from microduck_challenges.got_a_talent import tasks  # noqa: F401
    from microduck_challenges.got_a_talent.env import make_env_cfg
    from mjlab.envs import ManagerBasedRlEnv

    cfg = make_env_cfg()
    cfg.scene.num_envs = 8
    cfg.sim.device = "cpu"
    env = ManagerBasedRlEnv(cfg=cfg, device="cpu")

    seeded: list[int] = []
    for attempt in range(8):
        env.reset()
        flags = env.command_manager.get_command("body_pose")[:, 0]
        seeded = (flags > 0.5).nonzero().flatten().tolist()
        print(f"reset {attempt}: seeded envs {seeded}", flush=True)
        if seeded:
            break
    if not seeded:
        print("NO SPAWN after 8 resets — event or mark broken")
        sys.exit(1)

    e = seeded[0]
    act = torch.zeros(env.num_envs, env.action_manager.action.shape[1])
    payments = []
    for step in range(200):  # 4 s of sim: the raised foot must come down
        env.step(act)
        pay = float(R.toprock_step(env).max())  # per-env term; watch any env pay
        if pay > 0.0:
            payments.append((step, pay))
        if step % 20 == 0 or pay > 0.0:
            found = env.scene.sensors["feet_ground_contact"].data.found[e].tolist()
            z = float(env.scene["robot"].data.root_link_pos_w[e, 2])
            flag = float(env.command_manager.get_command("body_pose")[e, 0])
            print(f"step {step:3d} seed env {e}: feet {found} z {z:.3f} flag {flag:.0f} pay {pay:.2f}",
                  flush=True)
    print(f"payments seen: {len(payments)} first {payments[:3]}")
    if payments:
        print("SPAWN LANDS AND PAYS — the seeded step pays without any policy skill")


if __name__ == "__main__":
    main()

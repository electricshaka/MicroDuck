"""hs12 mechanism check: does spawn_seated_sweep land and PAY? Build a small CPU env, reset until
the sweep flag is up on a crouch-height env (the seed, not a draw — a draw on a standing start
pays nothing), then hold the default action and watch rewards.downrock_sweep / downrock_leg pay
as the seeded crouch turns and settles (called once per step like the reward manager; the term
also runs inside the manager, so the frontier double-advances here — existence of pay is the
question, not its magnitude).

    uv run --no-sync python scratch/sweep_env_test.py   (CPU only, a few minutes)
"""
import sys

sys.path.insert(0, r"C:\GitHub\microduck-hs12\src")

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
    for attempt in range(10):
        env.reset()
        flags = env.command_manager.get_command("body_pose")[:, 1]
        z = env.scene["robot"].data.root_link_pos_w[:, 2]
        seeded = ((flags > 0.5) & (z < 0.10)).nonzero().flatten().tolist()
        print(f"reset {attempt}: sweep-flag envs at crouch height {seeded}", flush=True)
        if seeded:
            break
    if not seeded:
        print("NO SEED after 10 resets — event or mark broken")
        sys.exit(1)

    e = seeded[0]
    act = torch.zeros(env.num_envs, env.action_manager.action.shape[1])
    sweeps, legs = [], []
    for step in range(200):  # 4 s: the seeded turn must pay while the crouch holds
        env.step(act)
        sw = float(R.downrock_sweep(env).max())
        lg = float(R.downrock_leg(env).max())
        if sw > 0.0:
            sweeps.append((step, sw))
        if lg > 0.0:
            legs.append((step, lg))
        if step % 20 == 0 or sw > 0.0:
            z = float(env.scene["robot"].data.root_link_pos_w[e, 2])
            turned = float(env.command_manager.get_term("body_pose").swept_turned[e])
            flag = float(env.command_manager.get_command("body_pose")[e, 1])
            print(f"step {step:3d} seed env {e}: z {z:.3f} turned {turned:+.2f} rad "
                  f"flag {flag:.0f} sweep {sw:.2f} leg {lg:.2f}", flush=True)
    print(f"sweep paid {len(sweeps)} steps, first {sweeps[:3]}")
    print(f"leg shape paid {len(legs)} steps, peak {max((p for _, p in legs), default=0.0):.2f}")
    if sweeps:
        print("SEED PAYS — the seeded sweep pays from the first steps with no policy skill")


if __name__ == "__main__":
    main()

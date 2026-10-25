"""MicroDuck Got a Talent trains in the Arena's physics, and reads the command a timeline carries."""

import importlib
import math
import types
from pathlib import Path

import numpy as np
import pytest
import torch

import microduck_challenges  # noqa: F401  (registers every task)
from arena.assets import ASSET_ROOT
from arena.constants import CTRL_DT, DECIMATION, DEFAULT_POSE, JOINT_NAMES, TRACE_WIDTH
from arena.course import compile_course
from arena.manifest import load_event
from arena.timeline import Timeline, load_troupe
from mjlab_microduck.tasks import mdp as microduck_mdp


@pytest.fixture(scope="module")
def cfg():
    return importlib.import_module("microduck_challenges.got_a_talent.env").make_env_cfg()


@pytest.fixture(scope="module")
def arena():
    return compile_course(load_event("got-a-talent"))


def _training_model(entity):
    """The robot as training compiles it: its MJCF with the entity's collision
    settings applied (mjlab's Entity runs each CollisionCfg.edit_spec on the spec)."""
    spec = entity.spec_fn()
    for collision in entity.collisions:
        collision.edit_spec(spec)
    return spec.compile()


def _body(model, geom):
    return model.body(int(model.geom_bodyid[geom])).name


def _robot_geoms(model, skip):
    return [g for g in range(model.ngeom)
            if (model.geom_contype[g] or model.geom_conaffinity[g]) and _body(model, g) not in skip]


def _colliding(model, skip):
    """(body, geom type) of every geom that can collide, bodies in `skip` left out."""
    return sorted((_body(model, g), int(model.geom_type[g])) for g in _robot_geoms(model, skip))


# mjlab's flat terrain (terrain_entity._import_ground_plane) is a plane with
# MuJoCo's default contact: priority 0, condim 3, sliding friction 1.
FLOOR = (0, 3, 1.0)


def _floor_contacts(model, skip):
    """(body, condim, sliding friction) of each robot geom's contact with that floor.
    MuJoCo pairs two geoms by priority first; at equal priority it takes the larger
    condim and the larger friction."""
    out = []
    for g in _robot_geoms(model, skip):
        priority, condim, friction = int(model.geom_priority[g]), int(model.geom_condim[g]), float(model.geom_friction[g][0])
        if priority > FLOOR[0]:
            pair = (condim, friction)
        elif priority < FLOOR[0]:
            pair = FLOOR[1:]
        else:
            pair = (max(condim, FLOOR[1]), max(friction, FLOOR[2]))
        out.append((_body(model, g), *pair))
    return sorted(out)


def test_got_a_talent_collides_like_the_arena(cfg, arena):
    # A sit puts the trunk and hips on the floor: training must collide with the
    # same robot geoms as the Arena's stage, or a seated trunk sinks in one and
    # not the other.
    robot = _training_model(cfg.scene.entities["robot"])
    assert _colliding(robot, skip={"world"}) == _colliding(arena, skip={"world", "course"})


def test_got_a_talent_touches_the_floor_like_the_arena(cfg, arena):
    """Every robot geom meets the floor with the Arena's condim and friction.
    Robot-on-robot contacts are where the two differ: training gives its non-foot
    geoms condim 1 (FULL_COLLISION), so two of them touch without friction, and
    the Arena's are all condim 3."""
    floor = arena.geom("floor").id
    assert (int(arena.geom_priority[floor]), int(arena.geom_condim[floor]), float(arena.geom_friction[floor][0])) == FLOOR
    robot = _training_model(cfg.scene.entities["robot"])
    assert _floor_contacts(robot, skip={"world"}) == _floor_contacts(arena, skip={"world", "course"})


def test_got_a_talent_solves_contacts_like_the_arena(cfg, arena):
    assert cfg.sim.mujoco.iterations == arena.opt.iterations
    assert cfg.sim.mujoco.ls_iterations == arena.opt.ls_iterations
    assert math.isclose(cfg.sim.mujoco.timestep, arena.opt.timestep)
    assert cfg.decimation == DECIMATION


def test_got_a_talent_plays_on_after_a_fall_like_the_stage(cfg):
    assert "fell_over" not in cfg.terminations
    assert "nan_state" in cfg.terminations


def test_got_a_talent_reads_what_a_timeline_carries(cfg):
    """twist.vx is the sit flag, head_pose the 4 head deltas, body[0] the bounce flag:
    a keyframe's 13 numbers, in the 61-D observation's command slot."""
    from microduck_challenges.got_a_talent import rewards as R

    assert isinstance(cfg.commands["twist"], microduck_mdp.SitStandCommandCfg)
    assert isinstance(cfg.commands["body_pose"], R.ToprockCommandCfg)
    assert len(cfg.commands["head_pose"].ranges) == 4
    actor = cfg.observations["actor"].terms
    assert list(actor)[-3:] == ["command", "head_command", "body_command"]
    assert actor["head_command"].params == {"command_name": "head_pose"}
    assert actor["body_command"].func is not microduck_mdp.zero_command_padding
    assert actor["body_command"].params == {"command_name": "body_pose"}


def test_checks_head_joint_order_matches_head_pose_trackings_own(cfg):
    """check.HEAD_JOINTS reads the head command's 4 dims as
    [neck_pitch, head_pitch, head_yaw, head_roll]; that must be the same order
    head_pose_tracking's own find_joints_by_actuator_names(_NECK_JOINT_PATTERNS)
    resolves them in, given this robot's actual joint declaration order."""
    from microduck_challenges.got_a_talent.check import HEAD_JOINTS

    entity = cfg.scene.entities["robot"].build()
    _, names = entity.find_joints_by_actuator_names(microduck_mdp._NECK_JOINT_PATTERNS)
    assert [JOINT_NAMES[i] for i in HEAD_JOINTS] == names


def test_head_spindle_penalty_is_quadratic_and_gated():
    """hs10's spindle lock: 0 at SPINDLE_POSE, quadratic in each joint's deviation,
    and paid only while the headstand flag is up."""
    from microduck_challenges.got_a_talent import rewards as R

    joint_pos = torch.zeros(3, 14)
    joint_pos[1, 5:9] = torch.tensor([0.0, 0.0, 0.0, 0.0])      # pitch 0.5 + 1.0 rad off
    joint_pos[2, 5:9] = torch.tensor([0.0, 0.0, 0.5, 0.0])      # plus half a rad of yaw
    env = types.SimpleNamespace(
        scene={"robot": types.SimpleNamespace(data=types.SimpleNamespace(joint_pos=joint_pos))},
        command_manager=types.SimpleNamespace(
            get_command=lambda name: torch.tensor([[0.0, 0.0, 0.0],
                                                   [0.0, 1.0, 0.0],
                                                   [0.0, 1.0, 1.0]])),
    )
    got = R.head_spindle_penalty(env)
    assert torch.allclose(got, torch.tensor([0.0, 0.5 ** 2 + 1.0 ** 2, 0.5 ** 2 + 1.0 ** 2 + 0.5 ** 2]))

    on_spindle = torch.zeros(1, 14)
    on_spindle[0, 5:9] = torch.tensor(R.SPINDLE_POSE)
    env2 = types.SimpleNamespace(
        scene=env.scene,
        command_manager=types.SimpleNamespace(
            get_command=lambda name: torch.tensor([[0.0, 1.0, 1.0]])),
    )
    env2.scene["robot"].data.joint_pos = on_spindle
    assert R.head_spindle_penalty(env2)[0] == 0.0


def test_head_spindle_pose_is_the_spawn_pose(cfg):
    """The spindle the penalty centers on is the pose the headstand spawn puts the neck in,
    so resuming spawns start on the penalty's minimum."""
    from microduck_challenges.got_a_talent import rewards as R

    assert R.SPINDLE_POSE == (R.HEADSTAND_NECK_PITCH, R.HEADSTAND_HEAD_PITCH, 0.0, 0.0)
    term = cfg.rewards["head_spindle_penalty"]
    assert term.weight < 0.0
    assert term.params == {"command_name": "twist"}


def _toprock_cmd(n: int = 1):
    """The footwork, sweep and bounce/step state ToprockCommand carries (see its __init__),
    sans the mjlab base."""
    return types.SimpleNamespace(
        command=torch.zeros(n, 6),
        last_foot=-torch.ones(n, dtype=torch.long),
        prev_down=torch.zeros(n, 2, dtype=torch.bool),
        since_paid=torch.full((n,), 10.0),
        since_valid=torch.full((n,), 10.0),
        airborne_run=torch.zeros(n, dtype=torch.long),
        swept_turned=torch.zeros(n),
        swept_frontier=torch.zeros(n),
        b_z_top=torch.zeros(n),
        b_prev_falling=torch.zeros(n, dtype=torch.bool),
        b_since_valid=torch.full((n,), 10.0),
        b_since_beat=torch.full((n,), 10.0),
        s_side=-torch.ones(n, dtype=torch.long),
        s_since_valid=torch.full((n,), 10.0),
        s_since_peak=torch.full((n,), 10.0),
    )


def _toprock_env(cmd, gravity_z=-1.0, z=0.115):
    """A standing duck stepping on a flat stage, seen by rewards.toprock_step: the feet sensor's
    found tensor is filled per step by the test."""
    sensor = types.SimpleNamespace(data=types.SimpleNamespace(found=torch.zeros(1, 2)))
    robot = types.SimpleNamespace(data=types.SimpleNamespace(
        projected_gravity_b=torch.tensor([[0.0, 0.0, gravity_z]]),
        root_link_pos_w=torch.tensor([[0.0, 0.0, z]]),
    ))

    class Scene(dict):  # like the real env's scene: scene["robot"] and scene.sensors
        def __init__(self, robot, sensors):
            super().__init__(robot=robot)
            self.sensors = {"feet_ground_contact": sensors}
            self.terrain = types.SimpleNamespace(env_origins=torch.zeros(1, 3))

    return types.SimpleNamespace(
        scene=Scene(robot, sensor),
        step_dt=0.02,
        command_manager=types.SimpleNamespace(
            get_term=lambda name: cmd,
            get_command=lambda name: cmd.command),
    )


def test_toprock_flag_only_rides_a_standing_posture():
    """body[0] (the hs13 bounce flag) is forced down every step while a sit or a headstand is
    commanded — the stage only ever raises it during the standing opening, and training never
    shows it elsewhere. body[2] (the cross-step flag) rides ONLY the bounce flag itself: steps
    are steps ON the rhythm, never on a quiet stand or a floor move. body[1] (hs12) is the mirror
    image of body[0]: it rides ONLY a commanded sit (downrock is floorwork). body[4] (hs12b)
    rides ONLY a commanded headstand (the L-kick is a freeze held on the head)."""
    from microduck_challenges.got_a_talent import rewards as R

    cmd = R.ToprockCommand.__new__(R.ToprockCommand)
    cmd._flag = torch.ones(3)
    cmd._step_flag = torch.ones(3)
    cmd._sweep_flag = torch.ones(3)
    cmd._kick_flag = torch.ones(3)
    cmd._command = torch.zeros(3, 6)
    twist = torch.tensor([[0.0, 0.0, 0.0],   # stand
                          [1.0, 0.0, 0.0],   # sit
                          [0.0, 1.0, 0.0]])  # headstand
    cmd._env = types.SimpleNamespace(command_manager=types.SimpleNamespace(
        get_command=lambda name: twist))
    R.ToprockCommand._update_command(cmd)
    assert cmd._command[:, 0].tolist() == [1.0, 0.0, 0.0]   # bounce rides the stand
    assert cmd._command[:, R.STEP].tolist() == [1.0, 0.0, 0.0]  # steps ride the bounce, nothing else
    assert cmd._command[:, 1].tolist() == [0.0, 1.0, 0.0]   # sweep rides the sit
    assert cmd._command[:, R.KICK].tolist() == [0.0, 0.0, 1.0]  # L-kick rides the headstand
    assert cmd._command[:, [3, 5]].abs().max() == 0.0       # the last two slots stay zero


def test_toprock_step_pays_dense_while_the_rhythm_holds():
    """hs11b's dense footwork: a valid alternating step (new contact by the foot NOT last down,
    on a frame with only that foot on the floor, in tempo) opens a hold_s window paid on EVERY
    step — the headstand's shape, which is why it is learnable. Keeping the reward means keeping
    the rhythm: the same foot twice and two-footed hops never reopen the window, a step counts
    only in tempo, and standing still earns nothing while the flag is up."""
    from microduck_challenges.got_a_talent import rewards as R

    cmd = _toprock_cmd()
    cmd.command[0, 0] = 1.0  # toprock up
    env = _toprock_env(cmd)

    def step(left, right):
        env.scene.sensors["feet_ground_contact"].data.found = torch.tensor(
            [[float(left), float(right)]])
        return float(R.toprock_step(env))

    assert step(1, 0) == 1.0        # first foot down: opens the window
    assert step(0, 0) == 1.0        # the foot swinging through the air: DENSE — still inside it
    cmd.since_valid[0] = 0.8
    cmd.since_paid[0] = 10.0
    assert step(0, 0) == 0.0        # window lapsed: nothing pays while feet hover (airborne 2)
    assert step(0, 0) == 0.0        # (airborne 3: now flying)
    assert step(1, 1) == 0.0        # landing both feet after flight: a HOP — fresh, in tempo, no
    assert step(0, 0) == 0.0        # window still closed; clear the floor for new contacts
    cmd.since_paid[0] = 0.6         # the tempo timer has run out
    assert step(0, 1) == 0.0        # the SAME foot (last down) landing again: no step, no window
    assert step(0, 0) == 0.0        # nothing pays in the air with a closed window
    cmd.since_paid[0] = 0.6
    assert step(1, 0) == 1.0        # the OTHER foot, on the floor throughout, in tempo: reopens
    assert step(1, 0) == 1.0        # inside the fresh window, even as the same foot stays down
    cmd.command[0, 0] = 0.0         # flag down: the gate closes, the same stepping earns nothing
    cmd.since_valid[0] = 0.1
    assert step(0, 1) == 0.0


def test_toprock_step_needs_a_standing_duck():
    """The gate is flat: tilted past 15° or off stand height, even a perfect step pays nothing."""
    from microduck_challenges.got_a_talent import rewards as R

    cmd = _toprock_cmd()
    cmd.command[0, 0] = 1.0
    tilted = _toprock_env(cmd, gravity_z=-0.9)  # 25.8° off vertical
    tilted.scene.sensors["feet_ground_contact"].data.found = torch.tensor([[1.0, 0.0]])
    assert float(R.toprock_step(tilted)) == 0.0
    low = _toprock_env(cmd, z=0.06)             # seated height
    low.scene.sensors["feet_ground_contact"].data.found = torch.tensor([[1.0, 0.0]])
    assert float(R.toprock_step(low)) == 0.0


def test_the_bounce_regime_is_retired(cfg):
    """hs13 + hs13b verdict (measured): the knee BOUNCE showed 0 cycles on stage at all 12
    checkpoints of both runs, and the cross-steps never alternated — the bounce rewards and seed
    are RETIRED registrations (functions stay tested in rewards.py, the toprock and grinder
    precedent). The old toprock terms stay retired too; posture_stillness keeps resting under
    the body flags (the wrapper also carries hs12's sweep flag); the command draws STAY, because
    body[0]/body[2] are part of the 61-number observation distribution the live policy was
    trained on — retiring a regime must not shift the resume base."""
    assert "bounce_hold" not in cfg.rewards and "bounce_step" not in cfg.rewards
    assert "spawn_mid_bounce" not in cfg.events
    assert "toprock_step" not in cfg.rewards and "toprock_sway" not in cfg.rewards
    assert "spawn_toprock" not in cfg.events
    assert cfg.commands["body_pose"].bounce_prob > 0.0
    assert cfg.commands["body_pose"].step_prob > 0.0
    still = cfg.rewards["posture_stillness"]
    assert still.func.__name__ == "unless_toprock"
    assert still.params["gate_command"] == "twist"
    assert "term" in still.params


def test_unless_toprock_rests_for_either_flag():
    """posture_stillness pauses for the headstand flag AND either dance flag (toprock, sweep) —
    all gates live in one flat wrapper, since the reward params are a flat dict and cannot nest
    wrappers."""
    from microduck_challenges.got_a_talent import rewards as R

    def env_with(twist_flag: float, body_flag: float, sweep_flag: float = 0.0):
        cmds = {"twist": torch.tensor([[0.0, twist_flag, 0.0]]),
                "body_pose": torch.tensor([[body_flag, sweep_flag, 0.0, 0.0, 0.0, 0.0]])}
        return types.SimpleNamespace(command_manager=types.SimpleNamespace(
            get_command=lambda name: cmds[name]))

    quiet = lambda env: torch.full((1,), 1.0)  # a stillness that always pays 1.0
    pay = lambda env: float(R.unless_toprock(env, quiet))
    assert pay(env_with(0.0, 0.0)) == 1.0      # neither flag: the still stand earns its stillness
    assert pay(env_with(1.0, 0.0)) == 0.0      # headstand up: rests, as unless_headstand did
    assert pay(env_with(0.0, 1.0)) == 0.0      # toprock up: rests now, so stepping can out-earn it
    assert pay(env_with(0.0, 0.0, 1.0)) == 0.0  # sweep up (hs12): a quiet crouch must not
    # out-earn the sweep                                        sweep the crouch


def test_spawn_toprock_marks_the_flag_and_skips_the_headstand(monkeypatch):
    """rewards.spawn_toprock seeds a mid-step stand on a share of resets: the toprock command's
    spawned mark goes up (its reset raises the flag from it), qpos moves — and an env already put
    on its head by spawn_headstand is left exactly as that spawn made it."""
    from microduck_challenges.got_a_talent import rewards as R

    monkeypatch.setattr("mjlab_microduck.tasks.mdp._servo_joint_ids",
                        lambda env, asset: list(range(14)))
    n = 4
    head_down = torch.tensor([False, True, False, False])  # env 1 is on its head already
    body = types.SimpleNamespace(spawned=torch.zeros(n, dtype=torch.bool))
    env = types.SimpleNamespace(
        device="cpu",
        scene={"robot": types.SimpleNamespace(data=types.SimpleNamespace(
            default_joint_pos=torch.zeros(n, 14)))},
        sim=types.SimpleNamespace(data=types.SimpleNamespace(
            qpos=torch.zeros(n, 21), qvel=torch.zeros(n, 6))),
        command_manager=types.SimpleNamespace(get_term=lambda name: {
            "twist": types.SimpleNamespace(spawned=head_down), "body_pose": body}[name]),
    )
    env.sim.data.qpos[:, 3] = 1.0  # upright free quaternion
    R.spawn_toprock(env, torch.arange(n), prob=1.0, joint_noise_std=0.0)

    assert body.spawned.tolist() == [True, False, True, True]  # the head-down env was skipped
    lifted = env.sim.data.qpos[:, 2] > 0.0
    assert lifted.tolist() == [True, False, True, True]        # it stands a touch higher
    assert torch.equal(env.sim.data.qpos[1], torch.tensor(
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, *([0.0] * 14)]))   # untouched, quaternion and all
    # exactly one foot's three lift joints moved (per side), by the mirrored signs
    for i in (0, 2, 3):
        moved = [j for j in range(14) if env.sim.data.qpos[i, 7 + j].abs() > 0.0]
        assert sorted(moved) in ([1, 2, 3], [10, 11, 12])
        sign = 1 if moved == [1, 2, 3] else -1
        assert all(env.sim.data.qpos[i, 7 + j] * sign > 0 for j in moved)


def test_the_spawn_pose_really_lifts_one_foot_off_the_floor(cfg):
    """The lift pose's SIGNS, measured on the training model (hs11b): right-leg joints are mounted
    mirrored, so a right-foot lift is hip_roll/hip_pitch/knee all NEGATIVE; a wrong sign would
    press the foot through the floor and the seed would be a fall, not a step."""
    import mujoco

    from microduck_challenges.got_a_talent import rewards as R

    robot = _training_model(cfg.scene.entities["robot"])
    data = mujoco.MjData(robot)
    trunk = mujoco.mj_name2id(robot, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")
    free = int(robot.jnt_qposadr[robot.body_jntadr[trunk]])
    adr = {n: int(robot.jnt_qposadr[mujoco.mj_name2id(robot, mujoco.mjtObj.mjOBJ_JOINT, n)])
           for n in JOINT_NAMES}
    foot = {s: mujoco.mj_name2id(robot, mujoco.mjtObj.mjOBJ_GEOM, f"{s}_foot_collision")
            for s in ("left", "right")}

    def standing_z(side):
        data.qpos[:] = 0.0
        data.qpos[free + 2] = 0.115
        data.qpos[free + 3] = 1.0
        mujoco.mj_forward(robot, data)
        return float(data.geom_xpos[foot[side], 2])

    base = standing_z("right")
    for j, offset in zip(R.TOPROCK_LIFT_JOINTS["right"], R.TOPROCK_LIFT_POSE):
        data.qpos[adr[JOINT_NAMES[j]]] -= offset        # mirrored side
    mujoco.mj_forward(robot, data)
    lifted = float(data.geom_xpos[foot["right"], 2])
    assert lifted > base + 0.02  # the foot is clearly off the floor (probe: ~0.05 m vs 0.015)


# ── hs12: the downrock sweep (body[1]) ─────────────────────────────────────────


def _sweep_env(cmd, z: float = 0.06, yaw: float = 3.0, head_found: float = 0.0,
               body_found: float = 1.0, feet_found: float = 1.0, pelvis_found: float = 0.0):
    """A duck on a flat stage, seen by rewards.downrock_* / sweep_*: trunk height, trunk yaw rate
    and the four ground sensors (head, body/legs, feet, and hs12c's pelvis-only) are set per step
    by the test (joint_pos for the leg shape)."""
    def sensor(v):
        return types.SimpleNamespace(data=types.SimpleNamespace(found=torch.full((1, 1), v)))

    robot = types.SimpleNamespace(data=types.SimpleNamespace(
        root_link_pos_w=torch.tensor([[0.0, 0.0, z]]),
        root_link_ang_vel_w=torch.tensor([[0.0, 0.0, yaw]]),
        joint_pos=torch.zeros(1, 14)))

    class Scene(dict):
        def __init__(self, robot):
            super().__init__(robot=robot)
            self.sensors = {"head_ground": sensor(head_found),
                            "body_ground": sensor(body_found),
                            "feet_ground_contact": sensor(feet_found),
                            "pelvis_ground": sensor(pelvis_found)}
            self.terrain = types.SimpleNamespace(env_origins=torch.zeros(1, 3))

    return types.SimpleNamespace(
        scene=Scene(robot),
        step_dt=0.02,
        command_manager=types.SimpleNamespace(
            get_term=lambda name: cmd,
            get_command=lambda name: cmd.command),
    )


def test_downrock_sweep_pays_frontier_only_in_the_crouch():
    """hs12's pay (RESTORED after hs12c's grinder starved the sweep to 0-5 deg/run — see the
    rewards.py header and checkpoints/got_a_talent/hs12c/): the frontier shape at seated speed —
    NEW counter-clockwise turn, past the furthest so far, capped — so a slow real sweep pays FULL
    at the curriculum's start (rate_max 2.0). Turning back pays nothing, and reclaiming the ground
    pays nothing until the frontier moves again. Standing, head-down or flag-down, the same
    turning earns zero. The flat-platform turn at z 0.061 pays — the accepted cost, measured."""
    from microduck_challenges.got_a_talent import rewards as R

    cmd = _toprock_cmd()
    cmd.command[0, 1] = 1.0  # sweep up
    env = _sweep_env(cmd)    # crouched at z 0.06, trunk turning CCW at 3 rad/s

    assert float(R.downrock_sweep(env)) == 1.0   # 3 rad/s capped at 2: full pay
    assert float(R.downrock_sweep(env)) == 1.0   # dense: every step of the arc
    env.scene["robot"].data.root_link_ang_vel_w = torch.tensor([[0.0, 0.0, -3.0]])
    assert all(float(R.downrock_sweep(env)) == 0.0 for _ in range(4))  # turning back: nothing
    env.scene["robot"].data.root_link_ang_vel_w = torch.tensor([[0.0, 0.0, 3.0]])
    assert all(float(R.downrock_sweep(env)) == 0.0 for _ in range(4))  # reclaiming: frontier static
    assert float(R.downrock_sweep(env)) == 1.0   # past the frontier again: it pays

    fresh = _toprock_cmd(); fresh.command[0, 1] = 1.0
    standing = _sweep_env(fresh, z=0.115)
    assert float(R.downrock_sweep(standing)) == 0.0    # not the crouch: not downrock
    fresh2 = _toprock_cmd(); fresh2.command[0, 1] = 1.0
    on_head = _sweep_env(fresh2, head_found=1.0)
    assert float(R.downrock_sweep(on_head)) == 0.0     # that's a headspin, not a sweep
    none = _toprock_cmd()                              # flag down
    assert float(R.downrock_sweep(_sweep_env(none))) == 0.0


def test_sweep_progress_pays_the_regime_and_pelvis_contact_names_the_cheat():
    """DORMANT terms (hs12c revert) kept honest: the regime term pays the kneel + support alone
    (no shape, no turn — the headstand_progress role it was written for). The pelvis sensor
    penalises EXACTLY the cheat — pelvis/trunk down while the flag is up (weight -3 at its
    dormant registration); a planted knee is body_ground, NOT this sensor, and with the flag down
    sitting is the show's own pose, never a penalty."""
    from microduck_challenges.got_a_talent import rewards as R

    cmd = _toprock_cmd(); cmd.command[0, 1] = 1.0
    env = _sweep_env(cmd, z=0.10)
    assert float(R.sweep_progress(env)) == 1.0
    env.scene["robot"].data.root_link_pos_w = torch.tensor([[0.0, 0.0, 0.061]])
    assert float(R.sweep_progress(env)) == 0.0             # the flat butt is not the regime
    cmd2 = _toprock_cmd(); cmd2.command[0, 1] = 1.0
    air = _sweep_env(cmd2, z=0.10, body_found=0.0, feet_found=0.0)
    assert float(R.sweep_progress(air)) == 0.0             # nothing planted: no support

    cmd3 = _toprock_cmd(); cmd3.command[0, 1] = 1.0
    butt = _sweep_env(cmd3, z=0.061, pelvis_found=1.0)     # the platform, flag up: full earn
    assert float(R.sweep_pelvis_contact(butt)) == 1.0      # (weight is negative: it costs)
    clean = _sweep_env(cmd3, z=0.095, pelvis_found=0.0)    # kneel: pelvis off the floor
    assert float(R.sweep_pelvis_contact(clean)) == 0.0
    cmd3.command[0, 1] = 0.0                              # flag down: sitting is not a crime
    sit = _sweep_env(cmd3, z=0.061, pelvis_found=1.0)
    assert float(R.sweep_pelvis_contact(sit)) == 0.0


def test_unless_sit_rest_rests_for_either_regime_flag():
    """hs12c's composite wrapper: the sit term pays only while NEITHER regime that replaces
    sitting is commanded — the headstand flag (hs2's tripod-war rule, kept) or the sweep flag
    (new: the flat crouch and the grinder cannot both be the target)."""
    from microduck_challenges.got_a_talent import rewards as R

    def env_with(headstand, sweep):
        twist = torch.tensor([[0.0, headstand, 0.0]])
        body = torch.tensor([[0.0, sweep, 0.0, 0.0, 0.0, 0.0]])
        return types.SimpleNamespace(command_manager=types.SimpleNamespace(
            get_command=lambda name: twist if name == "twist" else body))

    pays = lambda env: float(R.unless_sit_rest(
        env, lambda e, **kw: torch.tensor([1.0])))
    assert pays(env_with(0.0, 0.0)) == 1.0
    assert pays(env_with(1.0, 0.0)) == 0.0      # headstand up: rests, as unless_headstand did
    assert pays(env_with(0.0, 1.0)) == 0.0      # sweep up: rests, the hs12c addition


def test_downrock_leg_pays_one_leg_forward_either_side():
    """The arc's SHAPE: hip ~1.15 rad forward and knee near straight, either leg (the right side
    is mounted mirrored). A deep crouch with both legs folded pays ~0 — the shape is seasoning
    on the turn, not a pose to park in (and the seated gate closes it outside the crouch)."""
    from microduck_challenges.got_a_talent import rewards as R

    cmd = _toprock_cmd()
    cmd.command[0, 1] = 1.0
    env = _sweep_env(cmd, yaw=0.0)

    folded = env.scene["robot"].data.joint_pos
    folded[0, 2], folded[0, 3] = -0.4079, 1.35       # the SIT crouch on the left
    folded[0, 11], folded[0, 12] = 0.4079, -1.35     # and on the right
    assert float(R.downrock_leg(env)) < 0.01

    env.scene["robot"].data.joint_pos[0, 2] = -1.15  # left leg swept out
    env.scene["robot"].data.joint_pos[0, 3] = 0.25
    assert float(R.downrock_leg(env)) == pytest.approx(1.0)

    env.scene["robot"].data.joint_pos = torch.zeros(1, 14)
    env.scene["robot"].data.joint_pos[0, 11] = 1.15  # right leg (mirrored signs)
    env.scene["robot"].data.joint_pos[0, 12] = -0.25
    assert float(R.downrock_leg(env)) == pytest.approx(1.0)

    cmd.command[0, 1] = 0.0                          # flag down: the shape is just sitting
    assert float(R.downrock_leg(env)) == 0.0


def test_spawn_seated_sweep_marks_both_terms_and_sweeps_one_leg(monkeypatch):
    """rewards.spawn_seated_sweep seeds the crouch MID-SWEEP: the twist command's seated mark
    (its reset forces the sit flag up, headstand down), this command's sweep mark, the SIT
    keyframe with exactly ONE leg swept out (hip 1.15, knee 0.25 — hs12's shape, restored by the
    hs12c revert), crouch height and a live counter-clockwise trunk
    rate — while a head-down spawn is left exactly as it was and any toprock mark is dropped."""
    from microduck_challenges.got_a_talent import rewards as R

    monkeypatch.setattr("mjlab_microduck.tasks.mdp._servo_joint_ids",
                        lambda env, asset: list(range(14)))
    n = 4
    head_down = torch.tensor([False, True, False, False])   # env 1 is on its head already
    twist = types.SimpleNamespace(spawned=head_down,
                                  seated_spawned=torch.zeros(n, dtype=torch.bool))
    body = types.SimpleNamespace(spawned=torch.tensor([True, False, True, False]),  # toprock seeds
                                 sweep_spawned=torch.zeros(n, dtype=torch.bool))

    class Scene(dict):
        def __init__(self, robot):
            super().__init__(robot=robot)
            self.env_origins = torch.zeros(n, 3)   # _env_origin_z reads scene.env_origins directly

    env = types.SimpleNamespace(
        device="cpu",
        scene=Scene(types.SimpleNamespace(data=types.SimpleNamespace(
            default_joint_pos=torch.zeros(n, 14)))),
        sim=types.SimpleNamespace(data=types.SimpleNamespace(
            qpos=torch.zeros(n, 21), qvel=torch.zeros(n, 6))),
        command_manager=types.SimpleNamespace(
            get_term=lambda name: {"twist": twist, "body_pose": body}[name]),
    )
    R.spawn_seated_sweep(env, torch.arange(n), prob=1.0, joint_noise_std=0.0)

    assert twist.seated_spawned.tolist() == [True, False, True, True]   # head-down env skipped
    assert body.sweep_spawned.tolist() == [True, False, True, True]
    assert body.spawned.tolist() == [False, False, False, False]        # toprock seeds replaced
    assert env.sim.data.qpos[1].abs().max() == 0.0                      # env 1 untouched, all of it
    for i in (0, 2, 3):
        assert env.sim.data.qpos[i, 2] == pytest.approx(R.SWEEP_SPAWN_Z)  # the crouch height
        assert 1.5 <= float(env.sim.data.qvel[i, 5]) <= 3.0               # already turning CCW
        l_hip, l_knee = float(env.sim.data.qpos[i, 9]), float(env.sim.data.qpos[i, 10])
        r_hip, r_knee = float(env.sim.data.qpos[i, 18]), float(env.sim.data.qpos[i, 19])
        bl_h, bl_k = R.SWEEP_LEG_OFFSET                      # the swept leg, as hs12 seeds it
        left_out = abs(l_hip - bl_h) < 1e-6 and abs(l_knee - bl_k) < 1e-6
        right_out = abs(r_hip + bl_h) < 1e-6 and abs(r_knee + bl_k) < 1e-6
        assert left_out != right_out                                   # exactly one leg swept out
        if left_out:                                                   # the planted leg holds the crouch
            assert (r_hip, r_knee) == pytest.approx((0.4079, -1.35))
        else:
            assert (l_hip, l_knee) == pytest.approx((-0.4079, 1.35))


def test_the_seed_sweep_pose_fits_the_robot(cfg):
    """Every joint angle the sweep seed writes — crouch keyframe and swept-leg offsets, both
    mirrored sides — sits inside the training model's joint limits. A sign slip here would spawn
    the duck through itself, not mid-sweep."""
    import mujoco

    from microduck_challenges.got_a_talent import rewards as R

    robot = _training_model(cfg.scene.entities["robot"])
    qid = {n: mujoco.mj_name2id(robot, mujoco.mjtObj.mjOBJ_JOINT, n) for n in JOINT_NAMES}

    def fits(j, value):
        lo, hi = robot.jnt_range[qid[JOINT_NAMES[j]]]
        return lo <= value <= hi

    assert all(fits(j, v) for j, v in R.SWEEP_SIT_POSE.items())
    hip, knee = R.SWEEP_LEG_OFFSET
    assert fits(2, hip) and fits(3, knee)              # left side, as seeded
    assert fits(11, -hip) and fits(12, -knee)          # right side, mirrored


def test_the_downrock_wiring_lives_in_the_env(cfg):
    """hs12b's config, restored by the hs12c REVERT: the frontier sweep pays 6.0 on the seated
    gate alone (head_sensor, no grinder sensors), the grinder terms are NOT registered
    (sweep_progress, sweep_pelvis_contact — they starved the sweep, see rewards.py's header), the
    sit composite keeps its unless_headstand rest and NOT the unless_sit_rest swap (pausing it
    under the sweep flag emptied the crouch), the seed event is registered, the sweep flag is
    drawn, and the quarter-turns-first rate_max curriculum starts at a slow turn paying FULL.
    The leg seasoning is RETIRED since hs12b — it paid flat 0.004 for all 3000 hs12 iterations
    (the run's recorded lesson: cut dead seasoning); that function and the grinder functions stay
    tested, their registrations must not come back by accident."""
    from microduck_challenges.got_a_talent import rewards as R

    sweep = cfg.rewards["downrock_sweep"]
    assert sweep.func is R.downrock_sweep
    assert sweep.weight == 6.0
    assert "downrock_leg" not in cfg.rewards           # retired seasoning stays cut
    assert sweep.params["command_name"] == "body_pose" and sweep.params["rate_max"] == 2.0
    assert "feet_sensor" not in sweep.params           # grinder gate stays off the frontier pay
    assert "sweep_progress" not in cfg.rewards         # dormant grinder terms stay unregistered
    assert "sweep_pelvis_contact" not in cfg.rewards
    pc = cfg.rewards["posture_composite"]
    assert pc.func is R.unless_headstand               # the rest-stack loop's wrapper stands
    assert cfg.commands["body_pose"].sweep_prob > 0.0
    event = cfg.events["spawn_seated_sweep"]
    assert event.func is R.spawn_seated_sweep and event.params["prob"] > 0.0
    stages = cfg.curriculum["sweep_rate_max"].params["stages"]
    assert stages[0]["rate_max"] == 2.0 and stages[-1]["rate_max"] > stages[0]["rate_max"]


# ── hs12b: the L-kick freeze (body[4]) ─────────────────────────────────────────


def _kick_env(cmd, head_found=1.0, z=0.15, headstand=1.0, sit=0.0):
    """A duck on its head on a flat stage, seen by rewards.l_kick_hold: the head sensor, the
    trunk height, the twist command and (joint_pos) the leg shape are filled per test."""
    twist = torch.tensor([[sit, headstand, 0.0]])
    head = types.SimpleNamespace(data=types.SimpleNamespace(found=torch.full((1, 1), head_found)))
    robot = types.SimpleNamespace(data=types.SimpleNamespace(
        root_link_pos_w=torch.tensor([[0.0, 0.0, z]]),
        joint_pos=torch.zeros(1, 14)))

    class Scene(dict):
        def __init__(self, robot, head):
            super().__init__(robot=robot)
            self.sensors = {"head_ground": head}
            self.terrain = types.SimpleNamespace(env_origins=torch.zeros(1, 3))

    return types.SimpleNamespace(
        scene=Scene(robot, head),
        step_dt=0.02,
        command_manager=types.SimpleNamespace(
            get_term=lambda name: cmd,
            get_command=lambda name: twist if name == "twist" else cmd.command),
    )


def test_l_kick_hold_pays_only_the_flagged_freeze_on_the_head():
    """The freeze pays on a headstand (head planted, trunk up) with body[4]=1 and the legs in
    shape; every missing condition — flag, headstand, planted head, height — zeroes it: the pay
    must never reach a state that is not the freeze."""
    from microduck_challenges.got_a_talent import rewards as R

    cmd = _toprock_cmd()
    cmd.command[0, R.KICK] = 1.0
    env = _kick_env(cmd)
    env.scene["robot"].data.joint_pos[0, 2] = R.LKICK_HIP["left"]
    env.scene["robot"].data.joint_pos[0, 11] = R.LKICK_NEEDLE_HIP["right"]
    assert float(R.l_kick_hold(env)) == pytest.approx(1.0)

    cmd.command[0, R.KICK] = 0.0
    assert float(R.l_kick_hold(env)) == 0.0            # flag down: just a headstand
    cmd.command[0, R.KICK] = 1.0
    assert float(R.l_kick_hold(_kick_env(cmd, head_found=0.0))) == 0.0   # mid-dive, head in air
    assert float(R.l_kick_hold(_kick_env(cmd, headstand=0.0, sit=1.0))) == 0.0  # seated: no ride
    assert float(R.l_kick_hold(_kick_env(cmd, z=0.05))) == 0.0           # collapsed, not up


def test_l_kick_hold_pays_either_leg_level_and_fades_off_shape():
    """Either leg may carry the kick (the robot's two sides swing differently in the world, but
    both reach level at their own hip sign); an ordinary headstand (both legs the HOME needle)
    pays ~nothing, and a half-swinged leg sits between — the gradient toward the freeze."""
    from microduck_challenges.got_a_talent import rewards as R

    cmd = _toprock_cmd()
    cmd.command[0, R.KICK] = 1.0
    env = _kick_env(cmd)
    pos = env.scene["robot"].data.joint_pos
    pos[0, 2] = R.LKICK_NEEDLE_HIP["left"]             # both legs the HOME needle: not a kick
    pos[0, 11] = R.LKICK_NEEDLE_HIP["right"]
    assert float(R.l_kick_hold(env)) < 0.01
    pos[0, 2] = R.LKICK_HIP["left"]                    # left leg level, right stays the needle
    full = float(R.l_kick_hold(env))
    assert full == pytest.approx(1.0)
    pos[0, 2] = 0.7                                    # halfway out: pays, but less
    assert float(R.l_kick_hold(env)) < full
    pos[:] = 0.0
    pos[0, 2] = R.LKICK_NEEDLE_HIP["left"]             # back to HOME on the left (needle)
    pos[0, 11] = R.LKICK_HIP["right"]                  # and the RIGHT leg level (mirrored sign)
    assert float(R.l_kick_hold(env)) == pytest.approx(1.0)


def test_spawn_l_kick_marks_both_terms_and_levels_one_leg(monkeypatch):
    """rewards.spawn_l_kick seeds the freeze IN the freeze: the twist command's headstand mark
    AND this command's kick mark, one leg swung level with a straight knee and the other leg the
    HOME needle, upside down at headstand height and AT REST — while a head-down spawn and a
    seated-sweep seed are left exactly as they were and any toprock mark is dropped."""
    from microduck_challenges.got_a_talent import rewards as R

    monkeypatch.setattr("mjlab_microduck.tasks.mdp._servo_joint_ids",
                        lambda env, asset: list(range(14)))
    n = 5
    head_down = torch.tensor([False, True, False, False, False])   # env 1 on its head already
    twist = types.SimpleNamespace(spawned=head_down.clone(),
                                  seated_spawned=torch.zeros(n, dtype=torch.bool))
    body = types.SimpleNamespace(spawned=torch.tensor([True, False, False, False, False]),
                                 sweep_spawned=torch.tensor([False, False, True, False, False]),
                                 kick_spawned=torch.zeros(n, dtype=torch.bool))

    class Scene(dict):
        def __init__(self, robot):
            super().__init__(robot=robot)
            self.env_origins = torch.zeros(n, 3)

    home = torch.zeros(n, 14)                        # the real HOME, hips ±0.458, knees ∓0.005
    home[:, 2], home[:, 3] = -0.458, -0.005
    home[:, 11], home[:, 12] = 0.458, 0.005
    env = types.SimpleNamespace(
        device="cpu",
        scene=Scene(types.SimpleNamespace(data=types.SimpleNamespace(
            default_joint_pos=home))),
        sim=types.SimpleNamespace(data=types.SimpleNamespace(
            qpos=torch.zeros(n, 21), qvel=torch.zeros(n, 6))),
        command_manager=types.SimpleNamespace(
            get_term=lambda name: {"twist": twist, "body_pose": body}[name]),
    )
    R.spawn_l_kick(env, torch.arange(n), prob=1.0, joint_noise_std=0.0)

    assert body.kick_spawned.tolist() == [True, False, False, True, True]  # envs 1, 2 skipped
    assert twist.spawned.tolist() == [True, True, False, True, True]       # headstand rides too
    assert body.spawned.tolist() == [False, False, False, False, False]    # toprock marks dropped
    for i in (1, 2):                                                       # the skips, untouched
        assert env.sim.data.qpos[i].abs().max() == 0.0
    for i in (0, 3, 4):
        assert env.sim.data.qpos[i, 2] == pytest.approx(R.HEADSTAND_Z)     # headstand height
        assert env.sim.data.qvel[i].abs().max() == 0.0                     # frozen, not spinning
        l_hip, l_knee = float(env.sim.data.qpos[i, 9]), float(env.sim.data.qpos[i, 10])
        r_hip, r_knee = float(env.sim.data.qpos[i, 18]), float(env.sim.data.qpos[i, 19])
        left_kick = abs(l_hip - R.LKICK_HIP["left"]) < 1e-6 and abs(l_knee) < 1e-6
        right_kick = abs(r_hip - R.LKICK_HIP["right"]) < 1e-6 and abs(r_knee) < 1e-6
        assert left_kick != right_kick                                     # exactly one leg level
        if left_kick:                                                      # the needle holds HOME
            assert (r_hip, r_knee) == pytest.approx((R.LKICK_NEEDLE_HIP["right"], 0.005))
        else:
            assert (l_hip, l_knee) == pytest.approx((R.LKICK_NEEDLE_HIP["left"], -0.005))


def test_the_seed_kick_pose_fits_the_robot(cfg):
    """Every joint angle the kick seed writes — kick hip either side, straight knees, the HOME
    needle — sits inside the training model's joint limits. 1.42 is deliberately inside the hard
    1.571 but PAST the soft 1.414: the leg only goes level at the very end of its travel."""
    import mujoco

    from microduck_challenges.got_a_talent import rewards as R

    robot = _training_model(cfg.scene.entities["robot"])
    qid = {n: mujoco.mj_name2id(robot, mujoco.mjtObj.mjOBJ_JOINT, n) for n in JOINT_NAMES}

    def fits(j, value):
        lo, hi = robot.jnt_range[qid[JOINT_NAMES[j]]]
        return lo <= value <= hi

    assert fits(2, R.LKICK_HIP["left"]) and fits(3, R.LKICK_KNEE)
    assert fits(11, R.LKICK_HIP["right"]) and fits(12, R.LKICK_KNEE)
    assert fits(2, R.LKICK_NEEDLE_HIP["left"]) and fits(11, R.LKICK_NEEDLE_HIP["right"])


def test_the_lkick_wiring_lives_in_the_env(cfg):
    """hs12b's config: the freeze pays 6.0 (the move's weight), it is gated on the kick flag in
    slot 4, the seed event is registered, and the flag is drawn — all of it off unless the
    headstand posture carries it (the command masks that part; the tests above the gate)."""
    from microduck_challenges.got_a_talent import rewards as R

    assert R.KICK == 4                       # the slot the show will raise for the freeze
    kick = cfg.rewards["l_kick_hold"]
    assert kick.func is R.l_kick_hold and kick.weight == 6.0
    assert kick.params["command_name"] == "body_pose"
    assert cfg.commands["body_pose"].kick_prob > 0.0
    event = cfg.events["spawn_l_kick"]
    assert event.func is R.spawn_l_kick and event.params["prob"] > 0.0


def test_the_feet_sensor_column_order(cfg):
    """rewards.toprock_step treats found[:, 0] and found[:, 1] as the two feet; the alternation
    is order-agnostic, but pin the actual columns so the docs stay true."""
    import mujoco

    from microduck_challenges.got_a_talent import rewards as R

    assert R.TOPROCK == 0  # the flag slot the check plays from a keyframe's body[0]
    entity = cfg.scene.entities["robot"].build()
    ids, names = entity.find_geoms([r"^(left_foot_collision|right_foot_collision)$"])
    assert sorted(names) == ["left_foot_collision", "right_foot_collision"]
    robot = _training_model(cfg.scene.entities["robot"])
    li = mujoco.mj_name2id(robot, mujoco.mjtObj.mjOBJ_GEOM, "left_foot_collision")
    ri = mujoco.mj_name2id(robot, mujoco.mjtObj.mjOBJ_GEOM, "right_foot_collision")
    assert li != -1 and ri != -1


# ── hs13: the breakdance overhaul — the beat bounce (body[0]) and cross-steps (body[2]) ──


def _bounce_env(cmd):
    """A standing duck on a flat stage, seen by rewards.bounce_hold / bounce_step: trunk z and
    vertical velocity, tilt, knee flexion and the feet sensor's found row are set per step by the
    test (joint_pos relative to a zero default_joint_pos — the tests speak in offsets)."""
    sensor = types.SimpleNamespace(data=types.SimpleNamespace(
        found=torch.tensor([[1.0, 1.0]])))
    robot = types.SimpleNamespace(data=types.SimpleNamespace(
        projected_gravity_b=torch.tensor([[0.0, 0.0, -1.0]]),
        root_link_pos_w=torch.tensor([[0.0, 0.0, 0.115]]),
        root_link_lin_vel_w=torch.tensor([[0.0, 0.0, 0.0]]),
        joint_pos=torch.zeros(1, 14),
        default_joint_pos=torch.zeros(1, 14),
    ))

    class Scene(dict):
        def __init__(self, robot, sensors):
            super().__init__(robot=robot)
            self.sensors = {"feet_ground_contact": sensors}
            self.terrain = types.SimpleNamespace(env_origins=torch.zeros(1, 3))

    return types.SimpleNamespace(
        scene=Scene(robot, sensor),
        step_dt=0.02,
        command_manager=types.SimpleNamespace(
            get_term=lambda name: cmd,
            get_command=lambda name: cmd.command),
    )


def test_bounce_hold_opens_its_window_only_on_a_real_cycle():
    """hs13's rhythm contract: one VALID cycle (fell >= 1.2 cm from its own top, having REACHED
    the top, knees flexed, a foot planted, in tempo) opens a dense window paid every step; the
    window lapses unless the next real cycle comes. A twitch, a sag with no rise back, straight
    knees, a hop, spam and standing still never open it, and the flag down closes everything."""
    from microduck_challenges.got_a_talent import rewards as R

    cmd = _toprock_cmd()
    cmd.command[0, 0] = 1.0  # bounce up
    env = _bounce_env(cmd)
    step = lambda **kw: float(R.bounce_hold(_apply(env, **kw)))

    def _apply(env, **kw):
        d = env.scene["robot"].data
        d.root_link_pos_w[0, 2] = kw.get("z", 0.115)
        d.root_link_lin_vel_w[0, 2] = kw.get("vz", 0.0)
        d.projected_gravity_b[0, 2] = kw.get("gravity_z", -1.0)
        knee = kw.get("knee", 0.35)
        d.joint_pos[0, 3] = knee
        d.joint_pos[0, 12] = -knee
        feet = kw.get("feet", (1.0, 1.0))
        env.scene.sensors["feet_ground_contact"].data.found = torch.tensor([[feet[0], feet[1]]])
        return env

    assert step(z=0.115, vz=-0.05) == 0.0      # sinking from the stand: no cycle yet, cold window
    assert step(z=0.096, vz=+0.05) == 1.0      # the trough: 1.9 cm drop, knees, planted, tempo:
    assert step(z=0.115, vz=0.0) == 1.0         # DENSE — the rise pays too (headstand_hold shape)
    cmd.b_since_valid[0] = 0.8                  # the window lapses...
    assert step(z=0.115, vz=0.0) == 0.0         # ...with no new cycle: stillness earns nothing

    cmd2 = _toprock_cmd(); cmd2.command[0, 0] = 1.0; env2 = _bounce_env(cmd2)
    s2 = lambda **kw: float(R.bounce_hold(_apply(env2, **kw)))
    assert s2(z=0.115, vz=-0.05) == 0.0
    assert s2(z=0.109, vz=+0.05) == 0.0         # a 0.6 cm twitch: below the drop bar, no window
    cmd3 = _toprock_cmd(); cmd3.command[0, 0] = 1.0; env3 = _bounce_env(cmd3)
    s3 = lambda **kw: float(R.bounce_hold(_apply(env3, **kw)))
    assert s3(z=0.102, vz=-0.05) == 0.0
    assert s3(z=0.089, vz=+0.05) == 0.0         # drop enough, but never reached the top (0.102
                                                # vs BOUNCE_Z_TOP 0.104 — hs13b fallback (a)):
    cmd4 = _toprock_cmd(); cmd4.command[0, 0] = 1.0; env4 = _bounce_env(cmd4)
    s4 = lambda **kw: float(R.bounce_hold(_apply(env4, **kw)))
    assert s4(z=0.115, vz=-0.05, knee=0.05) == 0.0
    assert s4(z=0.096, vz=+0.05, knee=0.05) == 0.0   # straight knees: a hip-hinge, not a bounce
    assert s4(z=0.115, vz=-0.05) == 0.0
    assert s4(z=0.096, vz=+0.05, feet=(0.0, 0.0)) == 0.0  # both feet airborne: a HOP, not a bounce
    cmd5 = _toprock_cmd(); cmd5.command[0, 0] = 1.0; env5 = _bounce_env(cmd5)
    s5 = lambda **kw: float(R.bounce_hold(_apply(env5, **kw)))
    assert s5(z=0.115, vz=-0.05) == 0.0
    assert s5(z=0.096, vz=+0.05) == 1.0         # one real cycle: the window is open...
    cmd5.b_since_beat[0] = 0.04                 # ...but a SECOND trough this fast is spam
    assert s5(z=0.115, vz=-0.05) == 1.0         # (window still open — paying, no new cycle yet)
    cmd5.b_since_valid[0] = 10.0                # close the window by hand to watch the beat gate
    assert s5(z=0.096, vz=+0.05) == 0.0         # real drop, off the beat: no cycle, stays closed
    cmd5.b_since_beat[0] = 0.5                  # now on the beat: the cycle counts again
    assert s5(z=0.115, vz=-0.05) == 0.0
    assert s5(z=0.096, vz=+0.05) == 1.0
    cmd5.command[0, 0] = 0.0                    # flag down: even a fresh window earns nothing
    cmd5.b_since_valid[0] = 0.1
    assert s5(z=0.110, vz=0.0) == 0.0
    cmd5.command[0, 0] = 1.0
    cmd5.b_since_valid[0] = 0.1
    assert s5(z=0.060, vz=0.0) == 0.0           # at crouch height the bounce regime is closed
    assert s5(z=0.110, vz=0.0, gravity_z=-0.9) == 0.0  # and tilted past 25.8°


def test_bounce_step_counts_alternating_hips_only():
    """The cross-steps (body[2], riding the bounce): a peak is one hip 0.45 rad forward of home,
    then the OTHER side's peak counts (hs11's alternation rule), capped in tempo; each peak
    refreshes a dense window. Same side twice, half-hearted shuffles, spam and the flag down
    leave the window to lapse. Left forward is NEGATIVE, right POSITIVE (the downrock signs)."""
    from microduck_challenges.got_a_talent import rewards as R

    cmd = _toprock_cmd()
    cmd.command[0, 0] = 1.0
    cmd.command[0, 2] = 1.0  # step up (it only ever rides the bounce)
    env = _bounce_env(cmd)

    def step(l_dev=0.0, r_dev=0.0, z=0.115, gravity_z=-1.0):
        d = env.scene["robot"].data
        d.joint_pos[0, 2] = l_dev
        d.joint_pos[0, 11] = r_dev
        d.root_link_pos_w[0, 2] = z
        d.projected_gravity_b[0,2] = gravity_z
        return float(R.bounce_step(env))

    assert step(l_dev=-0.55) == 1.0        # LEFT hip forward: first peak opens the window
    assert step(l_dev=-0.55) == 1.0        # inside the window: DENSE
    cmd.s_since_valid[0] = 1.2             # window lapses...
    assert step(l_dev=-0.55) == 0.0        # ...and the SAME side again never reopens it
    cmd.s_since_peak[0] = 0.5
    assert step(r_dev=+0.55) == 1.0        # the RIGHT side, past the cap: alternation pays
    cmd.s_since_valid[0] = 1.2
    assert step(l_dev=-0.30) == 0.0        # 0.30 < 0.45: a shuffle is not a step
    cmd.s_since_peak[0] = 0.5
    assert step(l_dev=-0.55) == 1.0        # a real left peak: the window reopens
    cmd.s_since_valid[0] = 1.2             # lapse it again...
    assert step(r_dev=+0.55) == 0.0        # ...this right peak is due, but:
    cmd.s_since_peak[0] = 0.0
    assert step(r_dev=+0.55) == 0.0        # 0.0 s after the last: faster than the cap, not counted
    cmd.command[0, 2] = 0.0                # step flag down: the bounce alone earns this nothing
    cmd.s_since_valid[0] = 0.1
    assert step(l_dev=-0.55) == 0.0
    cmd.command[0, 2] = 1.0
    cmd.s_since_valid[0] = 0.1
    assert step(l_dev=-0.55, z=0.06) == 0.0        # regime gates: not at crouch height...
    assert step(l_dev=-0.55, gravity_z=-0.9) == 0.0  # ...and not leaning


def test_spawn_mid_bounce_marks_both_flags_and_shows_a_step(monkeypatch):
    """The seed puts episodes IN the trough: knees flexed, both hips bent the bounce way, exactly
    one hip a full step forward (the shown cross-step), a small sink to start the first cycle —
    and it marks BOTH flags (its reset raises bounce and step from them). The headstand, sweep and
    kick seeds are left exactly as they made it."""
    from microduck_challenges.got_a_talent import rewards as R

    monkeypatch.setattr("mjlab_microduck.tasks.mdp._servo_joint_ids",
                        lambda env, asset: list(range(14)))
    monkeypatch.setattr("mjlab_microduck.tasks.mdp._env_origin_z",
                        lambda env, ids: torch.zeros(len(ids), device="cpu"))
    n = 5
    taken_twist = torch.tensor([False, True, False, False, False])   # env 1 on its head
    body = types.SimpleNamespace(
        spawned=torch.zeros(n, dtype=torch.bool),
        step_spawned=torch.zeros(n, dtype=torch.bool),
        sweep_spawned=torch.tensor([False, False, True, False, False]),  # env 2 mid-sweep
        kick_spawned=torch.tensor([False, False, False, False, True]))   # env 4 in the freeze
    env = types.SimpleNamespace(
        device="cpu",
        scene={"robot": types.SimpleNamespace(data=types.SimpleNamespace(
            default_joint_pos=torch.zeros(n, 14)))},
        sim=types.SimpleNamespace(data=types.SimpleNamespace(
            qpos=torch.zeros(n, 21), qvel=torch.zeros(n, 6))),
        command_manager=types.SimpleNamespace(get_term=lambda name: {
            "twist": types.SimpleNamespace(spawned=taken_twist), "body_pose": body}[name]),
    )
    R.spawn_mid_bounce(env, torch.arange(n), prob=1.0, joint_noise_std=0.0)

    assert body.spawned.tolist() == [True, False, False, True, False]      # taken seeds skipped
    assert body.step_spawned.tolist() == [True, False, False, True, False]  # step rides the bounce
    for i in (0, 3):
        assert float(env.sim.data.qpos[i, 2]) == pytest.approx(R.BOUNCE_SPAWN_Z)
        assert float(env.sim.data.qvel[i, 2]) == pytest.approx(R.BOUNCE_SPAWN_VZ)     # trough fast
        q = env.sim.data.qpos[i, 7:21]
        hl, hr = float(q[2]), float(q[11])
        assert (hl <= -0.5) ^ (hr >= 0.5)          # exactly one hip shown a full step forward
        assert hl < -0.1 and hr > 0.1              # both hips bent the upright-keeping way
        assert 0.1 <= float(q[3]) <= 0.4 and -0.4 <= float(q[12]) <= -0.1  # knees flexed together
    for i in (1, 2, 4):                             # untouched: every number still zero
        assert torch.equal(env.sim.data.qpos[i], torch.zeros(21))


def adr_of(robot, name):
    """qpos address of one joint on the training model."""
    import mujoco

    return int(robot.jnt_qposadr[mujoco.mj_name2id(robot, mujoco.mjtObj.mjOBJ_JOINT, name)])


def test_the_seed_bounce_pose_is_a_real_crouch_free_squat(cfg):
    """The seed pose's SIGNS, measured on the training model at mjlab's HOME pose (the entity's
    init_state, not the model's zero-default): flexing the knees (left +, right −, the lift signs)
    plus the mirrored hip bend buys back the root drop — at the seed z the planted foot stays at
    the floor (a planted squat the physics can hold), the shown stepping foot lifts a touch more,
    and the seed z is a real drop below the stand, past the drop bar."""
    import re

    import mujoco

    from microduck_challenges.got_a_talent import rewards as R

    robot = _training_model(cfg.scene.entities["robot"])
    data = mujoco.MjData(robot)
    home = data.qpos.copy()
    for pat, val in cfg.scene.entities["robot"].init_state.joint_pos.items():
        for n in JOINT_NAMES:
            if re.fullmatch(pat, n):
                home[adr_of(robot, n)] = val
    trunk = mujoco.mj_name2id(robot, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")
    free = int(robot.jnt_qposadr[robot.body_jntadr[trunk]])
    adr = {n: adr_of(robot, n) for n in JOINT_NAMES}
    foot = {s: mujoco.mj_name2id(robot, mujoco.mjtObj.mjOBJ_GEOM, f"{s}_foot_collision")
            for s in ("left", "right")}

    def foot_zs(z):
        data.qpos[:] = home
        data.qpos[free + 2] = z
        data.qpos[free + 3] = 1.0
        mujoco.mj_forward(robot, data)
        return {s: float(data.geom_xpos[foot[s], 2]) for s in ("left", "right")}

    stand = foot_zs(0.115)
    sagged = foot_zs(R.BOUNCE_SPAWN_Z)           # same legs, root dropped: feet punch down
    assert sagged["left"] < stand["left"] - 0.010

    def seed(step_left):
        data.qpos[:] = home
        if step_left:
            data.qpos[adr[JOINT_NAMES[3]]] += R.BOUNCE_KNEE_FLEX - 0.20
            data.qpos[adr[JOINT_NAMES[2]]] -= R.BOUNCE_HIP_BEND + R.BOUNCE_STEP_SHOW
            data.qpos[adr[JOINT_NAMES[12]]] -= R.BOUNCE_KNEE_FLEX
            data.qpos[adr[JOINT_NAMES[11]]] += R.BOUNCE_HIP_BEND
        else:
            data.qpos[adr[JOINT_NAMES[12]]] -= R.BOUNCE_KNEE_FLEX - 0.20
            data.qpos[adr[JOINT_NAMES[11]]] += R.BOUNCE_HIP_BEND + R.BOUNCE_STEP_SHOW
            data.qpos[adr[JOINT_NAMES[3]]] += R.BOUNCE_KNEE_FLEX
            data.qpos[adr[JOINT_NAMES[2]]] -= R.BOUNCE_HIP_BEND
        data.qpos[free + 2] = R.BOUNCE_SPAWN_Z
        data.qpos[free + 3] = 1.0
        mujoco.mj_forward(robot, data)
        return {s: float(data.geom_xpos[foot[s], 2]) for s in ("left", "right")}

    for step_left, lifted_side in ((True, "left"), (False, "right")):
        z = seed(step_left)
        planted = "right" if step_left else "left"
        assert z[planted] > sagged[planted] + 0.010   # the knees bought the drop back...
        assert z[planted] > stand[planted] - 0.012    # ...to the floor, not deep through it
        assert z[lifted_side] >= z[planted] + 0.005   # the shown step lifts a touch (a shuffle)
    assert 0.115 - R.BOUNCE_SPAWN_Z >= R.BOUNCE_DROP  # the seed sits below a full drop bar


TIMELINE = Path(microduck_challenges.__file__).parent / "got_a_talent" / "timeline.json"


def test_the_reference_timeline_plays_on_the_arena():
    timeline = load_troupe(TIMELINE, load_event("got-a-talent")).ducks[0].timeline
    assert timeline.duration_s == 20.0
    assert len(timeline.times) == 9


def test_the_reference_timeline_stays_inside_what_training_saw(cfg):
    """The Arena guardrail, widened to the breakdance envelope the stage verified.

    Pollen's invariants hold as before: flags are only ever 0 or 1, slots the policy was
    never trained on stay zero, and each head angle stays inside the widened curriculum's
    own range. Since hs12 the sweep flag (body[1]) is a trained command: it may rise in the
    timeline, only ever to 1 — the knee-drop sweep played 464 deg on the Arena's stage with
    hs12/model_2250 (full 20.00 s, dive and headstand unchanged, 2026-10-24). Since hs12b the
    L-kick freeze flag (body[4]) may rise too: at 18.0 s it turns the dead-still headstand
    into the freeze pop — one leg back to the needle — stage-verified with hs12b/model_2000
    (kick_probe 3/3 seeds: freeze kick 0.01 rad, needle 0.10 rad, drift 11 deg, up 3.00 s;
    arena score full 20.00 s played, 2026-10-24).
    What the choreography adds — the toprock beat-bop (head targets 0.5–0.7 s apart,
    a 0.7 rad judge glance = 50 % of the ±1.4 rad yaw, a 0.2 rad nod roll = 65 % of
    ±0.31 rad) and the 2 s knee-drop sit — sits outside the *sampling* envelope
    training saw (new head target every 2–5 s, posture dwell 3.5 s), but every one of
    those moves was played on the Arena's stage with the published checkpoint (full
    20.00 s, no fall). The bounds below encode that verified envelope; do not widen
    them further without a fresh stage check."""
    timeline = load_troupe(TIMELINE, load_event("got-a-talent")).ducks[0].timeline
    final = cfg.curriculum["head_pose_range"].params["range_stages"][-1]["ranges"]
    flags = [c[0] for c in timeline.commands]
    assert set(flags) <= {0.0, 1.0}
    head_allowance = (0.4, 0.4, 0.55, 0.7)  # of the final curriculum range, per dim
    for command in timeline.commands:
        assert set(command[:3]) <= {0.0, 1.0}
        assert set(command[8:9]) <= {0.0, 1.0}              # sweep flag: trained by hs12
        assert set(command[11:12]) <= {0.0, 1.0}            # L-kick freeze flag: trained by hs12b
        assert command[7:8] + command[9:11] + command[12:] == (0.0,) * 4  # every other body slot
        for value, cap, (_, hi) in zip(command[3:7], head_allowance, final):
            assert abs(value) <= cap * hi
    postures = ["headstand" if c[1] >= 0.5 else ("sit" if c[0] else "stand")
                for c in timeline.commands]
    changes = [t for i, t in enumerate(timeline.times) if i and postures[i] != postures[i - 1]]
    bounds = changes + [timeline.duration_s]
    assert all(b - a >= 2.0 for a, b in zip(bounds, bounds[1:]))  # the knee-drop sit, tightest
    head_changes = [t for i, t in enumerate(timeline.times)
                    if i and timeline.commands[i][3:7] != timeline.commands[i - 1][3:7]]
    assert all(b - a >= 0.5 for a, b in zip(head_changes, head_changes[1:]))  # the beat-bop


STAND = (0.0,) * 13
SIT_LEFT = (1.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0) + (0.0,) * 6
EIGHT_S = Timeline(8.0, (0.0, 2.0, 5.0), (STAND, SIT_LEFT, STAND))


def _trace(z_at, yaw_at=lambda t: 0.0, tilt_at=lambda t: 0.0, duration_s=8.0):
    """A synthetic performance: trunk height, head yaw delta and roll tilt as functions of time."""
    n = round(duration_s / CTRL_DT)
    trace = np.zeros((n, TRACE_WIDTH), dtype=np.float32)
    for i in range(n):
        t = i * CTRL_DT
        half = np.radians(tilt_at(t)) / 2
        trace[i, 2] = z_at(t)
        trace[i, 3:7] = (np.cos(half), np.sin(half), 0.0, 0.0)
        trace[i, 7:] = DEFAULT_POSE
        trace[i, 7 + JOINT_NAMES.index("head_yaw")] += yaw_at(t)
    return trace


def _seated_between(a, b):
    return lambda t: 0.06 if a <= t < b else 0.115


def _report(trace):
    from microduck_challenges.got_a_talent import check

    return check.report(trace, EIGHT_S, load_event("got-a-talent").termination)


def test_a_performance_that_follows_its_timeline_passes():
    lines, passed, head = _report(_trace(_seated_between(3.0, 5.8), yaw_at=lambda t: 0.5 if 2.3 <= t < 5.0 else 0.0))
    assert passed, lines
    assert len(lines) == 4  # three keyframes, then the fall line
    assert head < 0.01


def test_a_sit_never_reached_fails():
    lines, passed, _ = _report(_trace(lambda t: 0.115))
    assert not passed
    assert "sit: not reached within 1.5 s" in lines[1]


def test_a_sit_reached_late_fails():
    lines, passed, _ = _report(_trace(_seated_between(3.8, 5.8)))
    assert not passed
    assert "sit: not reached within 1.5 s" in lines[1]


def test_a_sit_left_early_fails():
    lines, passed, _ = _report(_trace(_seated_between(3.0, 4.0)))
    assert not passed
    assert "left at 4.00 s" in lines[1]


def test_a_head_far_from_its_command_fails():
    lines, passed, head = _report(_trace(_seated_between(3.0, 5.8), yaw_at=lambda t: 0.9 if 2.3 <= t < 5.0 else 0.0))
    assert not passed
    assert head == pytest.approx(0.4, abs=1e-3)
    assert "head: off by 0.40 rad" in lines[1]


def test_a_fall_fails():
    lines, passed, _ = _report(_trace(_seated_between(3.0, 5.8), tilt_at=lambda t: 80.0 if t >= 6.0 else 0.0))
    assert not passed
    assert lines[-1] == "fell at 6.30 s"


def test_check_plays_the_stage_and_fails_a_walker(capsys):
    """End to end on the Arena: the reference walker does not sit."""
    from microduck_challenges.got_a_talent import check

    assert check.main([str(ASSET_ROOT / "policies" / "BEST_alpha_walking.onnx")]) == 1
    out = capsys.readouterr().out.splitlines()
    assert len(out) == 9 + 2  # a line per keyframe, the fall line, the verdict
    assert out[-1].startswith("FAIL: largest head error ")


def test_check_reads_a_troupes_star_alone(tmp_path, capsys):
    """A troupe's other ducks are other policies: the check plays the star alone, on its own keyframes."""
    import json

    from microduck_challenges.got_a_talent import check

    reference = json.loads(TIMELINE.read_text())
    troupe = tmp_path / "timeline.json"
    troupe.write_text(json.dumps({"timeline_version": 2, "duration_s": reference["duration_s"], "ducks": [
        {"policy": "self", "keyframes": reference["keyframes"]},
        {"policy": "local:got_a_talent/first@1", "start": {"x": 0.0, "y": 0.6, "yaw": 0}, "keyframes": [{"t": 0.0}]},
    ]}))
    assert check.main([str(ASSET_ROOT / "policies" / "BEST_alpha_walking.onnx"), "--timeline", str(troupe)]) == 1
    out = capsys.readouterr().out.splitlines()
    assert len(out) == 9 + 2  # the star's keyframes, the fall line, the verdict


def test_check_refuses_a_malformed_timeline(tmp_path, capsys):
    """main() must fail cleanly, like the Arena's own CLI, instead of raising."""
    from microduck_challenges.got_a_talent import check

    bad = tmp_path / "timeline.json"
    bad.write_text("{nope")
    assert check.main(["policy.onnx", "--timeline", str(bad)]) == 1
    err = capsys.readouterr().err
    assert "timeline.json" in err and "not a JSON file" in err


def test_the_manifest_says_what_the_policy_reads():
    """publish writes this into every got-a-talent manifest: the sit flag and the head pose, never 'unused'."""
    from mjlab_microduck import challenge as ch

    import microduck_challenges  # noqa: F401 — registers every challenge's task

    command = ch.for_task("Mjlab-GotATalent-MicroDuck").command
    assert command["twist"].startswith("[sit flag")
    assert "head_yaw" in command["head"]
    assert command["body"].startswith("[bounce flag")

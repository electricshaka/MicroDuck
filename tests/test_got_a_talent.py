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
    """twist.vx is the sit flag, head_pose the 4 head deltas, body[0] the toprock flag:
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
    """The footwork and sweep state ToprockCommand carries (see its __init__), sans the mjlab base."""
    return types.SimpleNamespace(
        command=torch.zeros(n, 6),
        last_foot=-torch.ones(n, dtype=torch.long),
        prev_down=torch.zeros(n, 2, dtype=torch.bool),
        since_paid=torch.full((n,), 10.0),
        since_valid=torch.full((n,), 10.0),
        airborne_run=torch.zeros(n, dtype=torch.long),
        swept_turned=torch.zeros(n),
        swept_frontier=torch.zeros(n),
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
    """body[0] is forced down every step while a sit or a headstand is commanded — the stage
    only ever raises it during the standing opening, and training never shows it elsewhere.
    body[1] (hs12) is its mirror image: it rides ONLY a commanded sit (downrock is floorwork)."""
    from microduck_challenges.got_a_talent import rewards as R

    cmd = R.ToprockCommand.__new__(R.ToprockCommand)
    cmd._flag = torch.ones(3)
    cmd._sweep_flag = torch.ones(3)
    cmd._command = torch.zeros(3, 6)
    twist = torch.tensor([[0.0, 0.0, 0.0],   # stand
                          [1.0, 0.0, 0.0],   # sit
                          [0.0, 1.0, 0.0]])  # headstand
    cmd._env = types.SimpleNamespace(command_manager=types.SimpleNamespace(
        get_command=lambda name: twist))
    R.ToprockCommand._update_command(cmd)
    assert cmd._command[:, 0].tolist() == [1.0, 0.0, 0.0]   # toprock rides the stand
    assert cmd._command[:, 1].tolist() == [0.0, 1.0, 0.0]   # sweep rides the sit
    assert cmd._command[:, 2:].abs().max() == 0.0           # the rest stay zero


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


def test_toprock_pays_more_than_standing_still_and_the_sway_is_seasoning(cfg):
    step, sway = cfg.rewards["toprock_step"], cfg.rewards["toprock_sway"]
    assert step.func.__name__ == "toprock_step" and step.weight > 0.0
    assert sway.weight < step.weight  # a held lean cannot out-earn the steps
    # Standing still must earn LESS than stepping: stillness rests while the flag is up,
    # and its headstand gate (gate_command="twist") survives the second wrapping.
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


def _sweep_env(cmd, z: float = 0.06, yaw: float = 3.0, head_found: float = 0.0):
    """A duck in the crouch on a flat stage, seen by rewards.downrock_*: trunk height, trunk yaw
    rate and the head sensor are set per step by the test (joint_pos for the leg shape)."""
    head = types.SimpleNamespace(data=types.SimpleNamespace(found=torch.full((1, 1), head_found)))
    robot = types.SimpleNamespace(data=types.SimpleNamespace(
        root_link_pos_w=torch.tensor([[0.0, 0.0, z]]),
        root_link_ang_vel_w=torch.tensor([[0.0, 0.0, yaw]]),
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
            get_command=lambda name: cmd.command),
    )


def test_downrock_sweep_pays_frontier_only_in_the_crouch():
    """hs12's pay is the headspin's frontier shape at seated speed: NEW counter-clockwise turn,
    past the furthest so far, capped — so a slow real sweep pays FULL at the curriculum's start
    (rate_max 2.0). Turning back pays nothing, and reclaiming the ground pays nothing until the
    frontier moves again. Standing, head-down or flag-down, the same turning earns zero."""
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
    keyframe with exactly ONE leg swept out, crouch height and a live counter-clockwise trunk
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
        left_out = abs(l_hip + 1.15) < 1e-6 and abs(l_knee - 0.25) < 1e-6
        right_out = abs(r_hip - 1.15) < 1e-6 and abs(r_knee + 0.25) < 1e-6
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
    """hs12's config: frontier sweep pays 6.0 (the headspin's weight), the leg shape 2.0 (must
    not out-earn the turn), the seed event is registered, the sweep flag is drawn, and the
    quarter-turns-first rate_max curriculum starts at a slow turn paying FULL."""
    from microduck_challenges.got_a_talent import rewards as R

    sweep, leg = cfg.rewards["downrock_sweep"], cfg.rewards["downrock_leg"]
    assert sweep.func is R.downrock_sweep and leg.func is R.downrock_leg
    assert sweep.weight == 6.0 and 0.0 < leg.weight < sweep.weight
    assert sweep.params["command_name"] == "body_pose" and sweep.params["rate_max"] == 2.0
    assert cfg.commands["body_pose"].sweep_prob > 0.0
    event = cfg.events["spawn_seated_sweep"]
    assert event.func is R.spawn_seated_sweep and event.params["prob"] > 0.0
    stages = cfg.curriculum["sweep_rate_max"].params["stages"]
    assert stages[0]["rate_max"] == 2.0 and stages[-1]["rate_max"] > stages[0]["rate_max"]


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


TIMELINE = Path(microduck_challenges.__file__).parent / "got_a_talent" / "timeline.json"


def test_the_reference_timeline_plays_on_the_arena():
    timeline = load_troupe(TIMELINE, load_event("got-a-talent")).ducks[0].timeline
    assert timeline.duration_s == 20.0
    assert len(timeline.times) == 9


def test_the_reference_timeline_stays_inside_what_training_saw(cfg):
    """The Arena guardrail, widened to the breakdance envelope the stage verified.

    Pollen's invariants hold as before: flags are only ever 0 or 1, the body slots
    stay zero, and each head angle stays inside the widened curriculum's own range.
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
        assert set(command[:3]) <= {0.0, 1.0} and command[7:] == (0.0,) * 6
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
    assert command["body"].startswith("[toprock flag")

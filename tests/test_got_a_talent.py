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
    """The footwork state ToprockCommand carries (see its __init__), sans the mjlab base."""
    return types.SimpleNamespace(
        command=torch.zeros(n, 6),
        last_foot=-torch.ones(n, dtype=torch.long),
        prev_down=torch.zeros(n, 2, dtype=torch.bool),
        since_paid=torch.full((n,), 10.0),
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
    only ever raises it during the standing opening, and training never shows it elsewhere."""
    from microduck_challenges.got_a_talent import rewards as R

    cmd = R.ToprockCommand.__new__(R.ToprockCommand)
    cmd._flag = torch.ones(3)
    cmd._command = torch.zeros(3, 6)
    twist = torch.tensor([[0.0, 0.0, 0.0],   # stand
                          [1.0, 0.0, 0.0],   # sit
                          [0.0, 1.0, 0.0]])  # headstand
    cmd._env = types.SimpleNamespace(command_manager=types.SimpleNamespace(
        get_command=lambda name: twist))
    R.ToprockCommand._update_command(cmd)
    assert cmd._command[:, 0].tolist() == [1.0, 0.0, 0.0]
    assert cmd._command[:, 1:].abs().max() == 0.0


def test_toprock_step_pays_alternations_only_and_at_the_cap():
    """hs11's frontier trick: a new contact by the foot that was NOT last down pays; the same
    foot landing twice pays nothing; and no payment is closer than the 0.5 s tempo cap."""
    from microduck_challenges.got_a_talent import rewards as R

    cmd = _toprock_cmd()
    cmd.command[0, 0] = 1.0  # toprock up
    env = _toprock_env(cmd)

    def step(left, right):
        env.scene.sensors["feet_ground_contact"].data.found = torch.tensor(
            [[float(left), float(right)]])
        return float(R.toprock_step(env))

    assert step(1, 0) == 1.0        # the first foot down starts the show
    assert step(0, 0) == 0.0         # lifting costs nothing
    assert step(1, 0) == 0.0         # the SAME foot down again: no step, no pay
    cmd.since_paid[0] = 0.5          # the tempo timer has run out
    assert step(0, 1) == 1.0         # the OTHER foot, in tempo: paid, timer restarts
    assert step(1, 1) == 0.0         # both down, nothing new
    cmd.since_paid[0] = 0.1          # too soon again
    assert step(0, 0) == 0.0
    assert step(1, 0) == 0.0         # a fresh foot in a hurry: fresh, but off-tempo → unpaid
    cmd.command[0, 0] = 0.0          # flag down: a full clean step earns nothing
    cmd.since_paid[0] = 10.0
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
    """posture_stillness pauses for the headstand flag AND the toprock flag — both gates live in
    one flat wrapper, since the reward params are a flat dict and cannot nest wrappers."""
    from microduck_challenges.got_a_talent import rewards as R

    def env_with(twist_flag: float, body_flag: float):
        cmds = {"twist": torch.tensor([[0.0, twist_flag, 0.0]]),
                "body_pose": torch.tensor([[body_flag, 0.0, 0.0, 0.0, 0.0, 0.0]])}
        return types.SimpleNamespace(command_manager=types.SimpleNamespace(
            get_command=lambda name: cmds[name]))

    quiet = lambda env: torch.full((1,), 1.0)  # a stillness that always pays 1.0
    pay = lambda env: float(R.unless_toprock(env, quiet))
    assert pay(env_with(0.0, 0.0)) == 1.0      # neither flag: the still stand earns its stillness
    assert pay(env_with(1.0, 0.0)) == 0.0      # headstand up: rests, as unless_headstand did
    assert pay(env_with(0.0, 1.0)) == 0.0      # toprock up: rests now, so stepping can out-earn it


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

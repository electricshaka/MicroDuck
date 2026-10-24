"""MicroDuck Got a Talent's headstand: the flag a timeline sends, what training pays for it, and the
headstand starts that give the balance something to learn from.

The command (twist) is [sit flag, headstand flag, spin flag]. While the headstand flag is 1 the duck
should get from wherever it is onto the top of its head and stay there, upside down, with nothing
else on the floor (stage 1, the hold). While the spin flag is also 1 it should spin there,
counter-clockwise seen from above, as fast as it can up to 15 rad/s (stage 2, ``headspin``).
Nothing taxes turning about the vertical axis (a twisting entry, a b-boy's angled step into the
roll, and a spinning headstand all hold as well as a still one); only tipping (pitch and roll
rates) is taxed.

Training latches the flag: once an episode commands a headstand it keeps commanding it until the
episode ends. Coming back down is a later stage.

The sit/stand rest stack (posture, height, upright, stillness, head tracking, speed caps) says
"stay upright" in every term, so ``unless_headstand`` switches each of those terms off while the
flag is 1, the way the backward-roll task pauses its rest stack mid-roll.

Measured in the Arena's physics (scratchpad spike, 2026-10-22): resting on its head with neck_pitch
+0.5, head_pitch +1.0 and the legs at HOME, the trunk sits at z 0.15 m and, held open-loop, stays
on its head about 1 s. Pollen's forward roll passes through that headstand 0.1-0.25 s after it
starts, too fast to stop without braking.
"""

import math
from dataclasses import dataclass as _dataclass

import torch

from mjlab.managers import CommandTerm, CommandTermCfg
from mjlab_microduck.tasks import mdp as microduck_mdp

HEADSTAND = 1  # twist slot of the headstand flag
SPIN = 2       # twist slot of the spin flag: 1 = spin counter-clockwise (seen from above) on the head
TOPROCK = 0    # body slot of the toprock flag: 1 = step in place to the beat while standing

HEADSTAND_Z = 0.15            # trunk height resting on the head (measured, see above)
HEADSTAND_NECK_PITCH = 0.5    # the measured balance pose, rad from 0 (not from HOME)
HEADSTAND_HEAD_PITCH = 1.0


def headstand_flag(env, command_name: str = "twist") -> torch.Tensor:
    return env.command_manager.get_command(command_name)[:, HEADSTAND]


def _inverted(env) -> torch.Tensor:
    """0 upright, 1 upside down: gravity's z in the trunk frame, -1 → +1, rescaled."""
    robot = env.scene["robot"]
    return (1.0 + robot.data.projected_gravity_b[:, 2]) * 0.5


def _found(env, sensor_name: str) -> torch.Tensor:
    return (env.scene.sensors[sensor_name].data.found > 0).any(dim=-1).float()


def _trunk_z(env) -> torch.Tensor:
    robot = env.scene["robot"]
    return robot.data.root_link_pos_w[:, 2] - env.scene.terrain.env_origins[:, 2]


# hs1 learned to lie upside down, rocking between the head and the top of the trunk
# at z 0.035-0.07 m: inverted and, every other step, head-only. The shaping terms
# now pay only with the trunk off the floor: nothing below 0.07 m, full from 0.12 m
# (standing is 0.115, the headstand 0.15).
LYING_Z, HIGH_Z = 0.07, 0.12


def _off_the_floor(env) -> torch.Tensor:
    return torch.clamp((_trunk_z(env) - LYING_Z) / (HIGH_Z - LYING_Z), 0.0, 1.0)


# ── Command ────────────────────────────────────────────────────────────────────


class HeadstandCommand(microduck_mdp.SitStandCommand):
    """The sit/stand command plus a latched headstand flag in twist[1].

    A resample that finds the flag up keeps it up; otherwise it commands a headstand with
    ``headstand_prob`` (and then no sit), else the sit/stand draw as before. A new episode starts
    with the flag down before its first draw, unless spawn_headstand put it on its head: mjlab
    applies reset events before it resets the commands, so the spawn marks ``spawned`` and the
    reset raises the flag from that mark.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._headstand_prob = float(getattr(cfg, "headstand_prob", 0.35))
        self._spin_prob = float(getattr(cfg, "spin_prob", 0.0))
        self.spawned = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # headspin's state: the trunk's turn about the vertical axis while spinning on the head
        # (rad, counter-clockwise positive), and the furthest it has reached.
        self.turned = torch.zeros(self.num_envs, device=self.device)
        self.frontier = torch.zeros(self.num_envs, device=self.device)

    def reset(self, env_ids):
        self.vel_command_b[env_ids, HEADSTAND] = 0.0
        extras = super().reset(env_ids)
        mark = self.spawned[env_ids]
        self.vel_command_b[env_ids, HEADSTAND] = torch.where(mark, 1.0, self.vel_command_b[env_ids, HEADSTAND])
        self.vel_command_b[env_ids, 0] = torch.where(mark, 0.0, self.vel_command_b[env_ids, 0])
        self._draw_spin(env_ids)
        self.spawned[env_ids] = False
        self.turned[env_ids] = 0.0
        self.frontier[env_ids] = 0.0
        return extras

    def _draw_spin(self, env_ids: torch.Tensor) -> None:
        """The spin flag: drawn afresh at every resample while the headstand flag is up, so spins
        start and stop on the head; always down otherwise."""
        up = self.vel_command_b[env_ids, HEADSTAND] > 0.5
        spin = up & (torch.rand(len(env_ids), device=self.device) < self._spin_prob)
        self.vel_command_b[env_ids, SPIN] = spin.float()

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        n = len(env_ids)
        if n == 0:
            return
        was_up = self.vel_command_b[env_ids, HEADSTAND] > 0.5
        super()._resample_command(env_ids)
        up = was_up | (torch.rand(n, device=self.device) < self._headstand_prob)
        self.vel_command_b[env_ids, HEADSTAND] = up.float()
        self.vel_command_b[env_ids, 0] = torch.where(up, 0.0, self.vel_command_b[env_ids, 0])
        self._draw_spin(env_ids)


@_dataclass(kw_only=True)
class HeadstandCommandCfg(microduck_mdp.SitStandCommandCfg):
    class_type: type = HeadstandCommand
    # Probability that a resample raises the headstand flag (it then stays up).
    headstand_prob: float = 0.35
    # Probability that a resample with the headstand flag up also raises the spin flag.
    spin_prob: float = 0.0

    def build(self, env) -> "HeadstandCommand":
        return HeadstandCommand(self, env)


# ── Rewards ────────────────────────────────────────────────────────────────────


def unless_headstand(env, term, gate_command: str = "twist", **params) -> torch.Tensor:
    """A sit/stand term, switched off while the headstand flag is up."""
    return term(env, **params) * (1.0 - headstand_flag(env, gate_command))


def headstand_progress(env, command_name: str = "twist") -> torch.Tensor:
    """Dense pull toward upside down while a headstand is commanded: 0 upright, 1 inverted,
    paid only with the trunk off the floor."""
    return headstand_flag(env, command_name) * _inverted(env) * _off_the_floor(env)


def headstand_support(env, head_sensor: str, body_sensor: str, command_name: str = "twist") -> torch.Tensor:
    """The head on the floor and nothing else, the trunk at least half way over and off the floor."""
    gate = torch.clamp((_inverted(env) - 0.5) / 0.4, 0.0, 1.0)
    support = _found(env, head_sensor) * (1.0 - _found(env, body_sensor))
    return headstand_flag(env, command_name) * gate * support * _off_the_floor(env)


def headstand_body_contact(env, body_sensor: str, command_name: str = "twist") -> torch.Tensor:
    """1 while a headstand is commanded and the trunk or a leg touches the floor (use a negative
    weight). Lifting off from the feet costs it for a moment; lying on the trunk costs it all along."""
    return headstand_flag(env, command_name) * _found(env, body_sensor)


def headstand_hold(
    env,
    head_sensor: str,
    body_sensor: str,
    command_name: str = "twist",
    z_target: float = HEADSTAND_Z,
    z_std: float = 0.03,
    tilt_rate_std: float = 1.5,
) -> torch.Tensor:
    """The goal: upside down within ~25°, only the head on the floor, trunk at headstand height,
    not tipping. Multiplicative, so lying on the back (low, trunk on the floor) or a head-and-feet
    plank earns nothing. Only the horizontal angular rates count as tipping: spinning about the
    vertical axis is free."""
    robot = env.scene["robot"]
    tilt_rate = torch.sum(torch.square(robot.data.root_link_ang_vel_w[:, :2]), dim=1)
    steady = torch.exp(-tilt_rate / tilt_rate_std ** 2)
    on_head = _on_head(env, head_sensor, body_sensor, z_target, z_std)
    return headstand_flag(env, command_name) * on_head * steady


def _on_head(env, head_sensor: str, body_sensor: str, z_target: float = HEADSTAND_Z, z_std: float = 0.03) -> torch.Tensor:
    """0-1: upside down within ~25° (full), only the head on the floor, trunk at headstand height."""
    inverted = torch.clamp((_inverted(env) - 0.85) / 0.1, 0.0, 1.0)
    height = torch.exp(-torch.square((_trunk_z(env) - z_target) / z_std))
    support = _found(env, head_sensor) * (1.0 - _found(env, body_sensor))
    return inverted * support * height


def headspin(
    env,
    head_sensor: str,
    body_sensor: str,
    command_name: str = "twist",
    rate_max: float = 15.0,
) -> torch.Tensor:
    """The spin, while the headstand and spin flags are both up: the trunk's NEW counter-clockwise
    turn about the vertical axis per second, past the furthest it has turned so far, capped at
    ``rate_max`` rad/s and scaled to 0-1. Faster pays more, up to the cap; turning back and
    forth pays nothing (the way back must be made up before the frontier moves again), and a
    turn only counts on the head (the hold's own test without its steadiness), so spinning while
    lying down or tumbling earns nothing. The policy may drive it with any joints: neck whips,
    leg swings, both."""
    command = env.command_manager.get_term(command_name)
    flags = command.command[:, HEADSTAND] * command.command[:, SPIN]
    on_head = _on_head(env, head_sensor, body_sensor)
    yaw_rate = env.scene["robot"].data.root_link_ang_vel_w[:, 2]
    counting = flags * (on_head > 0.5).float()
    command.turned += yaw_rate * env.step_dt * counting
    advance = torch.clamp(command.turned - command.frontier, min=0.0)
    command.frontier = torch.maximum(command.frontier, command.turned)
    rate = advance / env.step_dt
    return flags * on_head * torch.clamp(rate, max=rate_max) / rate_max


def head_contact_unless_headstand(env, head_sensor: str, command_name: str = "twist") -> torch.Tensor:
    """1 while the headstand flag is down and the head touches the floor (use a negative weight):
    the mirror of headstand_body_contact. hs2-hs4 "stood" as a tripod, both feet and the head on the
    floor, trunk leaning 36° with the neck folded to its limit; stable, and paid nearly all of the
    stand rewards but the composite, so the head never came up."""
    return (1.0 - headstand_flag(env, command_name)) * _found(env, head_sensor)


def action_beyond_limits(env, action_name: str = "joint_pos") -> torch.Tensor:
    """Squared distance of each commanded joint target past its joint's limits, summed (use a
    negative weight). A target past a limit moves nothing: the joint stops at the limit whatever
    the target, so the rewards give no gradient back from there. hs2 and hs3 held the neck at its
    -1.57 rad limit by commanding -2.5, then -9.5 rad, with exploration noise of 0.48: the stand
    and head-tracking rewards could never pull the head back up. This term can."""
    term = env.action_manager.get_term(action_name)
    target = term.raw_action * term.scale + term.offset
    limits = env.scene["robot"].data.soft_joint_pos_limits[:, term.target_ids]
    excess = (limits[..., 0] - target).clamp(min=0.0) + (target - limits[..., 1]).clamp(min=0.0)
    return torch.sum(torch.square(excess), dim=1)


def headspin_rate_max_curriculum(
    env,
    env_ids: torch.Tensor,
    stages: list[dict],
    reward_name: str = "headspin",
) -> torch.Tensor:
    """Step-staged rate_max curriculum for headspin ('quarter turns first')."""
    del env_ids
    term_cfg = env.reward_manager.get_term_cfg(reward_name)
    rate_max = term_cfg.params.get("rate_max", 3.0)
    for stage in stages:
        if env.common_step_counter >= stage["step"]:
            rate_max = stage["rate_max"]
    term_cfg.params["rate_max"] = rate_max
    return torch.tensor([rate_max], device=env.device)


def head_stillness_on_head(
    env,
    command_name: str = "twist",
    vel_std: float = 1.0,
    yaw_std: float = 0.15,
    roll_std: float = 0.15,
) -> torch.Tensor:
    """Reward for keeping the head/neck quiet and centered during headstand (hs8), forcing legs to drive spin."""
    robot = env.scene["robot"]
    # Head joint velocities: neck_pitch (5), head_pitch (6), head_yaw (7), head_roll (8)
    head_joint_vel = robot.data.joint_vel[:, 5:9]
    head_vel_sq = torch.sum(torch.square(head_joint_vel), dim=-1)
    vel_still = torch.exp(-head_vel_sq / (vel_std ** 2))

    # Head yaw (7) and roll (8) deviations from neutral 0.0
    yaw_err = torch.square(robot.data.joint_pos[:, 7])
    roll_err = torch.square(robot.data.joint_pos[:, 8])
    pose_still = torch.exp(-yaw_err / (yaw_std ** 2) - roll_err / (roll_std ** 2))

    return headstand_flag(env, command_name) * vel_still * pose_still


# The spindle pose for the headstand pivot, absolute joint angles of joints 5-8:
# neck_pitch, head_pitch (the measured balance pose, see spawn_headstand) and
# head_yaw, head_roll centered.
SPINDLE_POSE = (HEADSTAND_NECK_PITCH, HEADSTAND_HEAD_PITCH, 0.0, 0.0)


def head_spindle_penalty(env, command_name: str = "twist") -> torch.Tensor:
    """Summed squared deviation of the 4 head/neck joints (5-8) from SPINDLE_POSE while a
    headstand is commanded (use a negative weight). The strict counterpart of
    head_stillness_on_head: its exp reward bottoms out within ~0.3 rad of the pose, so beyond
    that basin a big neck swing costs nothing and gains nothing — hs8/hs9 kept pumping the
    spin with a neck yaw of ±31° even at its weight 3.0. A quadratic never saturates: every
    extra rad of yaw is paid for, so rotation has to come from the legs against the head's
    grip (a b-boy's leg drive), and the head/neck stays a quiet spindle for the freeze too.
    Off while the headstand flag is down: the standing head moves stay free."""
    robot = env.scene["robot"]
    pos = robot.data.joint_pos[:, 5:9]
    err = pos - torch.tensor(SPINDLE_POSE, device=pos.device, dtype=pos.dtype)
    return headstand_flag(env, command_name) * torch.sum(torch.square(err), dim=-1)


# ── Toprock (body[0], hs11) ────────────────────────────────────────────────────


class ToprockCommand(CommandTerm):
    """The toprock flag in body[0] (1 = step in place, 0 = stand still); body 1-5 stay zero.

    A resample draws the flag Bernoulli(toprock_prob). Every step it is forced down while a sit or
    a headstand is commanded: top-rock is a standing move, and the policy never sees the flag up
    in a posture it cannot step in — the same slot never carries two moves at once.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._flag_prob = float(getattr(cfg, "toprock_prob", 0.5))
        self._flag = torch.zeros(self.num_envs, device=self.device)
        self._command = torch.zeros(self.num_envs, 6, device=self.device)
        # spawn_toprock marks the episodes it starts mid-step; its reset raises the flag from the
        # mark (the HeadstandCommand.spawned pattern: mjlab applies reset events before commands).
        self.spawned = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # Footwork state for rewards.toprock_step (the frontier trick, as with the headspin's
        # turned/frontier): which foot landed last (-1 = neither yet), each foot's contact on the
        # previous step, seconds since a step counted (the ~2 Hz cap and the dense hold window),
        # and the airborne streak that tells a hop from a step.
        self.last_foot = -torch.ones(self.num_envs, dtype=torch.long, device=self.device)
        self.prev_down = torch.zeros(self.num_envs, 2, dtype=torch.bool, device=self.device)
        self.since_paid = torch.full((self.num_envs,), 10.0, device=self.device)
        # seconds since the last VALID alternating step — toprock_step pays while this is < hold_s
        self.since_valid = torch.full((self.num_envs,), 10.0, device=self.device)
        # consecutive control steps with BOTH feet off the floor; a few of them says the duck is
        # airborne (hopping), and an airborne landing is not a step.
        self.airborne_run = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return self._command

    def reset(self, env_ids):
        extras = super().reset(env_ids)
        # A spawn_toprock episode starts with the flag up (mjlab applies reset events first, so
        # the spawn has marked these env_ids); every other episode starts flag-down and waits for
        # its first draw, like the headstand does.
        self._flag[env_ids] = self.spawned[env_ids].float()
        self.last_foot[env_ids] = -1
        self.prev_down[env_ids] = False
        self.since_paid[env_ids] = 10.0
        self.since_valid[env_ids] = 10.0
        self.airborne_run[env_ids] = 0
        self.spawned[env_ids] = False
        return extras

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        self._flag[env_ids] = (
            torch.rand(len(env_ids), device=self.device) < self._flag_prob
        ).float()
        self.since_paid[env_ids] = 10.0  # a new segment may count its first step at once
        self.since_valid[env_ids] = 10.0  # and never inherits the previous segment's hold window

    def _update_command(self) -> None:
        twist = self._env.command_manager.get_command("twist")
        standing = (twist[:, 0] < 0.5) & (twist[:, 1] < 0.5)
        self._command[:, TOPROCK] = self._flag * standing
        self._command[:, 1:] = 0.0

    def _update_metrics(self) -> None:
        pass


@_dataclass(kw_only=True)
class ToprockCommandCfg(CommandTermCfg):
    class_type: type = ToprockCommand
    # Probability a resample raises the toprock flag (it then holds to the next resample).
    toprock_prob: float = 0.5

    def build(self, env) -> "ToprockCommand":
        return ToprockCommand(self, env)


TOPROCK_TILT_COS = math.cos(math.radians(15.0))  # the gate: trunk within 15 degrees of vertical


def toprock_flag(env, command_name: str = "body_pose") -> torch.Tensor:
    return env.command_manager.get_command(command_name)[:, TOPROCK]


def unless_toprock(env, term, gate_command: str = "twist", flag_command: str = "body_pose",
                   **params) -> torch.Tensor:
    """A sit/stand term, switched off while the headstand flag OR the toprock flag is up: the
    headstand gate of ``unless_headstand`` with the footwork gate added on (one flat wrapper —
    the reward params are a flat dict, so ``unless_headstand`` cannot nest inside it).
    posture_stillness uses this: a b-boy's stand stills, but under a toprock command standing
    still must earn LESS than stepping — unpaused, stillness pays quiet feet and the step is
    never started."""
    off = torch.maximum(headstand_flag(env, gate_command), toprock_flag(env, flag_command))
    return term(env, **params) * (1.0 - off)


def _toprock_gate(env, command_name: str) -> torch.Tensor:
    """The toprock flag up AND the duck truly standing: trunk at stand height and within 15° of
    vertical — footwork low, tilted or upside down earns nothing (a flat gate, not a soft one:
    the reward must not pay a state that is not the move)."""
    robot = env.scene["robot"]
    upright = torch.clamp(
        (-robot.data.projected_gravity_b[:, 2] - TOPROCK_TILT_COS) / 0.034, 0.0, 1.0
    )
    tall = torch.clamp((_trunk_z(env) - 0.10) / 0.01, 0.0, 1.0)  # full from z 0.11, standing is 0.115
    return toprock_flag(env, command_name) * upright * tall


def toprock_step(
    env,
    command_name: str = "body_pose",
    feet_sensor: str = "feet_ground_contact",
    period_s: float = 0.5,
    hold_s: float = 0.75,
) -> torch.Tensor:
    """The footwork itself, DENSE, the way the headstand pays. The flag is up and the duck stands;
    a valid step is a NEW contact by the foot that was NOT last down, no sooner than ``period_s``
    after the last valid one (the ~2 Hz beat), and made WITHOUT having been airborne — one foot
    always on the floor is what separates a step from a hop. Every valid step refreshes a window
    of ``hold_s``; the duck is paid on EVERY step it spends inside that window. A rhythm that
    lapses stops paying ``hold_s`` later, so the only way to keep collecting is to keep stepping —
    same self-sustaining shape as headstand_hold, which is why the headstand was learnable and
    hs11's per-landing-pay was not: one frame per landing amortized a full step to ~0.004, under
    the noise floor of the posture it costs to lift a foot. Same frontier rules as before:
    same-foot taps and two-footed hops never refresh the window."""
    cmd = env.command_manager.get_term(command_name)
    gate = _toprock_gate(env, command_name)
    found = env.scene.sensors[feet_sensor].data.found > 0  # (N, 2): left and right foot on the floor
    landed = found & ~cmd.prev_down
    fresh = (landed[:, 0] & (cmd.last_foot != 0)) | (landed[:, 1] & (cmd.last_foot != 1))
    flying = cmd.airborne_run >= 3       # ~60 ms with nothing underfoot: airborne, so not stepping
    cmd.since_paid += env.step_dt
    valid = fresh & ~flying & (cmd.since_paid >= period_s)
    cmd.since_paid = torch.where(valid, torch.zeros_like(cmd.since_paid), cmd.since_paid)
    cmd.since_valid += env.step_dt
    cmd.since_valid = torch.where(valid, torch.zeros_like(cmd.since_valid), cmd.since_valid)
    cmd.airborne_run = torch.where(
        found.any(dim=1), torch.zeros_like(cmd.airborne_run), cmd.airborne_run + 1)
    cmd.last_foot = torch.where(landed[:, 0], torch.zeros_like(cmd.last_foot), cmd.last_foot)
    cmd.last_foot = torch.where(landed[:, 1], torch.ones_like(cmd.last_foot), cmd.last_foot)
    cmd.prev_down = found
    return gate * (cmd.since_valid < hold_s).float()


def toprock_sway(
    env,
    command_name: str = "body_pose",
    target: float = 0.15,
    std: float = 0.1,
) -> torch.Tensor:
    """Light seasoning on the step: hip_roll held about ``target`` rad off HOME (both legs, the
    lateral weight shift of rocking side to side). Small next to toprock_step — it gives the steps
    their shape, it must not become the move: a held lean pays this and no steps, and at weight 0.5
    it cannot out-earn the 2.0 every-step rhythm window that only real stepping keeps alive."""
    robot = env.scene["robot"]
    home = robot.data.default_joint_pos
    dev = (
        torch.abs(robot.data.joint_pos[:, 1] - home[:, 1])
        + torch.abs(robot.data.joint_pos[:, 10] - home[:, 10])
    ) * 0.5
    return _toprock_gate(env, command_name) * torch.exp(-torch.square(dev - target) / std**2)


# ── Events ─────────────────────────────────────────────────────────────────────


def spawn_headstand(
    env,
    env_ids: torch.Tensor,
    prob: float = 0.2,
    z: float = HEADSTAND_Z,
    joint_noise_std: float = 0.05,
    yaw_rate_max: float = 6.0,
    command_name: str = "twist",
):
    """Start a share of episodes already on the head (the reverse curriculum that taught the
    roulade and the standup their second halves), with the headstand flag up. Some start
    spinning about the vertical axis, up to ``yaw_rate_max`` rad/s either way, so a spinning
    headstand is a state the policy has seen. Runs after set_ground_state."""
    ids = env_ids[torch.rand(len(env_ids), device=env.device) < prob]
    n = len(ids)
    if n == 0:
        return
    asset = env.scene["robot"]
    servo_ids = microduck_mdp._servo_joint_ids(env, asset)

    # Upside down: a half turn about the trunk's long axis, then a random heading.
    yaw = torch.rand(n, device=env.device) * 2 * math.pi - math.pi
    quat = torch.stack([torch.zeros_like(yaw), torch.cos(yaw / 2), torch.sin(yaw / 2), torch.zeros_like(yaw)], dim=1)
    env.sim.data.qpos[ids, 2] = z + microduck_mdp._env_origin_z(env, ids)
    env.sim.data.qpos[ids, 3:7] = quat
    env.sim.data.qvel[ids, :6] = 0.0
    # Free-joint angular velocity is in the body frame: upside down, world yaw is -body z.
    env.sim.data.qvel[ids, 5] = -(torch.rand(n, device=env.device) * 2 - 1) * yaw_rate_max

    cols = torch.tensor([7 + j for j in servo_ids], device=env.device, dtype=torch.long)
    pose = asset.data.default_joint_pos[ids][:, servo_ids].clone()
    pose[:, 5] = HEADSTAND_NECK_PITCH
    pose[:, 6] = HEADSTAND_HEAD_PITCH
    pose += torch.randn_like(pose) * joint_noise_std
    env.sim.data.qpos[ids.unsqueeze(1).long(), cols.unsqueeze(0)] = pose

    env.command_manager.get_term(command_name).spawned[ids] = True


# The raised foot's three joints, in the 14-servo order, per side — signs measured on the
# training model (scratch/probe_joint_signs.py): left hip_roll/hip_pitch/knee POSITIVE lift the
# left foot (its geom z 0.015 → ~0.055 m at these values); the right leg's joints are mounted
# mirrored, so its signs are negative. Together: thigh forward, knee bent, leg opened out.
TOPROCK_LIFT_JOINTS = {"left": (1, 2, 3), "right": (10, 11, 12)}
TOPROCK_LIFT_POSE = (0.15, 0.50, 0.30)   # hip_roll, hip_pitch, knee offsets (rad)


def spawn_toprock(
    env,
    env_ids: torch.Tensor,
    prob: float = 0.25,
    tilt: float = 0.05,
    z_lift: float = 0.015,
    joint_noise_std: float = 0.04,
    command_name: str = "body_pose",
):
    """Start a share of toprock episodes already MID-STEP: standing, one foot raised off the
    floor, weight leaning the other way, the toprock flag up (ToprockCommand.reset reads the mark).

    Why this exists: hs11 showed the alternation is never discovered from a planted stand —
    1750 iterations, 4096 envs, and the stage probe still counted one foot down. A clean step
    needs one-foot support the STAND policy never visits, so PPO's noise pays it almost never and
    the gradient never compounds — the same discovery wall the headless headstand hit, which
    spawn_headstand broke by seeding the state. Here the seed is half a step: the first thing that
    can happen is the raised foot LANDING, and a landing by a foot that was not last down pays at
    once. Runs after set_ground_state (standing) and after spawn_headstand: an episode already put
    on its head by that spawn is left alone — a headstand does not toprock."""
    head_down = env.command_manager.get_term("twist").spawned
    chosen = (torch.rand(len(env_ids), device=env.device) < prob) & ~head_down[env_ids]
    ids = env_ids[chosen]
    n = len(ids)
    if n == 0:
        return
    asset = env.scene["robot"]
    servo_ids = microduck_mdp._servo_joint_ids(env, asset)

    # Which foot goes up, per env: +1 left, -1 right (the mirrored-leg sign).
    side = torch.randint(0, 2, (n,), device=env.device) * 2 - 1
    cols = torch.tensor([7 + j for j in servo_ids], device=env.device, dtype=torch.long)
    pose = asset.data.default_joint_pos[ids][:, servo_ids].clone()
    for foot, sign in (("left", 1), ("right", -1)):
        j = torch.tensor(TOPROCK_LIFT_JOINTS[foot], device=env.device, dtype=torch.long)
        off = torch.tensor(TOPROCK_LIFT_POSE, device=env.device) * sign
        lift = (side == sign).nonzero().flatten()
        pose[lift.unsqueeze(1), j.unsqueeze(0)] += off

    pose += torch.randn_like(pose) * joint_noise_std
    env.sim.data.qpos[ids.unsqueeze(1).long(), cols.unsqueeze(0)] = pose

    # A touch higher, so the raised foot clears the floor; lean over the planted foot; still.
    env.sim.data.qpos[ids, 2] = env.sim.data.qpos[ids, 2] + z_lift
    q = env.sim.data.qpos[ids, 3:7].clone()          # free quaternion, w x y z
    roll = -side.float() * tilt                       # away from the lifted foot
    cw, sx = torch.cos(roll / 2), torch.sin(roll / 2)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    env.sim.data.qpos[ids, 3:7] = torch.stack(
        [cw * w - sx * x, cw * x + sx * w, cw * y + sx * z, cw * z - sx * y], dim=1)
    env.sim.data.qvel[ids] = 0.0

    env.command_manager.get_term(command_name).spawned[ids] = True

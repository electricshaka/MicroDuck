"""MicroDuck Got a Talent's headstand: the flag a timeline sends, what training pays for it, and the
headstand starts that give the balance something to learn from.

The command (twist) is [sit flag, headstand flag, 0]. While the headstand flag is 1 the duck should
get from wherever it is onto the top of its head and stay there, upside down, with nothing else on
the floor. Stage 1 of the headspin: the hold. The spin comes later, so nothing here taxes turning
about the vertical axis (a twisting entry, a b-boy's angled step into the roll, and a spinning
headstand all score as well as a still one); only tipping (pitch and roll rates) is taxed.

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

from mjlab_microduck.tasks import mdp as microduck_mdp

HEADSTAND = 1  # twist slot of the headstand flag

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
        self.spawned = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def reset(self, env_ids):
        self.vel_command_b[env_ids, HEADSTAND] = 0.0
        extras = super().reset(env_ids)
        mark = self.spawned[env_ids]
        self.vel_command_b[env_ids, HEADSTAND] = torch.where(mark, 1.0, self.vel_command_b[env_ids, HEADSTAND])
        self.vel_command_b[env_ids, 0] = torch.where(mark, 0.0, self.vel_command_b[env_ids, 0])
        self.spawned[env_ids] = False
        return extras

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        n = len(env_ids)
        if n == 0:
            return
        was_up = self.vel_command_b[env_ids, HEADSTAND] > 0.5
        super()._resample_command(env_ids)
        up = was_up | (torch.rand(n, device=self.device) < self._headstand_prob)
        self.vel_command_b[env_ids, HEADSTAND] = up.float()
        self.vel_command_b[env_ids, 0] = torch.where(up, 0.0, self.vel_command_b[env_ids, 0])


@_dataclass(kw_only=True)
class HeadstandCommandCfg(microduck_mdp.SitStandCommandCfg):
    class_type: type = HeadstandCommand
    # Probability that a resample raises the headstand flag (it then stays up).
    headstand_prob: float = 0.35

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
    inverted = torch.clamp((_inverted(env) - 0.85) / 0.1, 0.0, 1.0)  # full past ~25° from vertical
    height = torch.exp(-torch.square((_trunk_z(env) - z_target) / z_std))
    tilt_rate = torch.sum(torch.square(robot.data.root_link_ang_vel_w[:, :2]), dim=1)
    steady = torch.exp(-tilt_rate / tilt_rate_std ** 2)
    support = _found(env, head_sensor) * (1.0 - _found(env, body_sensor))
    return headstand_flag(env, command_name) * inverted * support * height * steady


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

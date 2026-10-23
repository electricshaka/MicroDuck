"""Watch a MicroDuck Got a Talent policy's headstand and headspin live, in a MuJoCo viewer window.

    uv run --frozen --group workshop python -m microduck_challenges.got_a_talent.headspin_view policy.onnx [--slow 1] [--no-spin]

Loops over seeds 1-3 the same test headspin_check judges: stand 2 s, the headstand flag (twist[1])
from 2 s, the spin flag (twist[2]) from 5 s, 15 s in all. The physics is the Arena's stage model
(collision shapes only); the window shows the robot's full visual model in the same pose, frame by
frame. Left-drag orbits, right-drag pans, scroll zooms, space pauses; close the window to stop.
``--slow 4`` plays at quarter speed; ``--no-spin`` leaves the spin flag down (for policies trained
before it existed, which fall when it rises).

Uses the Arena runner's own start (arena.runner._start, the settle included) and step, so what it
shows is what the stage plays.
"""

import argparse
import time
from pathlib import Path

import arena
import mujoco
import mujoco.viewer
import numpy as np
from arena import runner
from arena.constants import CTRL_DT
from arena.course import compile_course
from arena.manifest import load_event
from arena.policy import load_policy

EVENT = "got-a-talent"
HEADSTAND_AT, SPIN_AT, DURATION = 2.0, 5.0, 15.0


def _display_model() -> mujoco.MjModel:
    """The legs robot with its visual meshes, on a checkered floor: same joints and qpos layout."""
    spec = mujoco.MjSpec.from_file(str(Path(arena.__file__).parent / "assets/robot/legs/robot_allcollisions.xml"))
    spec.add_texture(name="grid", type=mujoco.mjtTexture.mjTEXTURE_2D, builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                     rgb1=[0.25, 0.27, 0.3], rgb2=[0.18, 0.2, 0.22], width=256, height=256)
    material = spec.add_material(name="grid", texrepeat=[8, 8], reflectance=0.1)
    material.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "grid"
    spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[3, 3, 0.05], material="grid", contype=0, conaffinity=0)
    spec.worldbody.add_light(pos=[0, 0, 1.5], dir=[0, 0, -1], diffuse=[0.8, 0.8, 0.8])
    return spec.compile()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Watch a Got a Talent headstand/headspin policy live.")
    parser.add_argument("onnx")
    parser.add_argument("--slow", type=float, default=1.0, help="slow-motion factor (4 = quarter speed)")
    parser.add_argument("--no-spin", action="store_true", help="never raise the spin flag")
    args = parser.parse_args(argv)

    event = load_event(EVENT)
    model = compile_course(event)
    policy = load_policy(Path(args.onnx))
    stand = np.zeros(13)
    headstand = stand.copy(); headstand[1] = 1.0
    spin = headstand.copy(); spin[2] = 0.0 if args.no_spin else 1.0

    display = _display_model()
    shown = mujoco.MjData(display)
    with mujoco.viewer.launch_passive(display, shown) as viewer:
        viewer.cam.distance, viewer.cam.elevation, viewer.cam.azimuth = 0.7, -10, 120
        seed = 0
        while viewer.is_running():
            seed = seed % 3 + 1
            sim, _ = runner._start(event, model, [policy], [0.0], seed, [stand])
            data, adr = sim.data, sim.ducks[0].adr
            for k in range(round(DURATION / CTRL_DT)):
                if not viewer.is_running():
                    return 0
                started = time.perf_counter()
                t = (k + 1) * CTRL_DT
                sim.step([spin if t > SPIN_AT else headstand if t > HEADSTAND_AT else stand])
                up_z = float(data.xmat[adr.trunk].reshape(3, 3)[2, 2])
                if t <= HEADSTAND_AT:
                    phase = "standing"
                else:
                    phase = "HEADSTAND" if up_z < -0.85 and data.qpos[2] > 0.10 else "headstand flag on"
                    if t > SPIN_AT and not args.no_spin:
                        phase += "  + spin flag"
                with viewer.lock():
                    shown.qpos[:] = data.qpos
                    mujoco.mj_forward(display, shown)
                    viewer.cam.lookat[:] = [data.qpos[0], data.qpos[1], 0.1]
                    viewer.set_texts([(None, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                                       f"{Path(args.onnx).name}  seed {seed}", f"t {t:5.2f} s   {phase}")])
                viewer.sync()
                time.sleep(max(0.0, CTRL_DT * args.slow - (time.perf_counter() - started)))
            time.sleep(1.0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

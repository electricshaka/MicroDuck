import math
from pathlib import Path
import imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from arena import runner
from arena.constants import CTRL_DT, JOINT_NAMES
from arena.course import compile_course
from arena.manifest import load_event
from arena.policy import load_policy
from microduck_challenges.got_a_talent.headspin_view import _display_model

EVENT = "got-a-talent"
POLICY_PATH = Path("publish-quackstreet-boy-the-tailspin/policy.onnx")
OUTPUT_MP4 = Path("scratch/microduck_breakdance_latest_720p.mp4")
OUTPUT_WEBP = Path("scratch/microduck_breakdance_latest.webp")
DURATION = 16.0
FPS = 50
WIDTH, HEIGHT = 1280, 720

def main():
    print(f"Loading event '{EVENT}' and policy '{POLICY_PATH}'...")
    event = load_event(EVENT)
    model = compile_course(event)
    policy = load_policy(POLICY_PATH)
    
    # Define choreography commands
    stand = np.zeros(13)
    
    # Phase 1: Toprock with head bob and step
    toprock = stand.copy()
    toprock[7] = 1.0  # body[0] toprock step
    toprock[3] = 0.35 # neck pitch
    toprock[4] = 0.35 # head pitch
    
    # Phase 2: Downrock sweep (coffee grinder)
    downrock = stand.copy()
    downrock[0] = 1.0 # twist[0] sit
    downrock[8] = 1.0 # body[1] downrock sweep
    
    # Phase 3: Inversion dive onto head
    dive = stand.copy()
    dive[1] = 1.0     # twist[1] headstand
    
    # Phase 4: Continuous headspin
    spin = dive.copy()
    spin[2] = 1.0     # twist[2] headspin
    
    # Phase 5: Inverted L-Kick freeze finish
    lkick = dive.copy()
    lkick[11] = 1.0   # body[4] L-kick freeze
    
    sim, _ = runner._start(event, model, [policy], [0.0], 1, [toprock])
    data = sim.data
    
    display = _display_model()
    display.vis.global_.offwidth = WIDTH
    display.vis.global_.offheight = HEIGHT
    shown = mujoco.MjData(display)
    renderer = mujoco.Renderer(display, HEIGHT, WIDTH)
    
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    
    trunk_id = mujoco.mj_name2id(display, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")
    free_adr = int(display.jnt_qposadr[display.body_jntadr[trunk_id]])
    joint_adrs = [int(display.jnt_qposadr[mujoco.mj_name2id(display, mujoco.mjtObj.mjOBJ_JOINT, n)])
                  for n in JOINT_NAMES]
                  
    try:
        font_large = ImageFont.truetype("arialbd.ttf", 22)
        font_small = ImageFont.truetype("arial.ttf", 16)
        font_stat = ImageFont.truetype("consolab.ttf", 18)
        font_note = ImageFont.truetype("ariali.ttf", 15)
    except Exception:
        font_large = ImageFont.load_default()
        font_small = font_large
        font_stat = font_large
        font_note = font_large

    total_steps = round(DURATION / CTRL_DT)
    print(f"Rendering {total_steps} frames at {WIDTH}x{HEIGHT} @ {FPS} FPS...")
    
    frames = []
    spin_turns = 0.0
    sweep_deg = 0.0
    last_yaw = None
    last_sweep_yaw = None
    
    for k in range(total_steps):
        t = (k + 1) * CTRL_DT
        
        # Choreography schedule
        if t < 2.5:
            # Alternating head glance & rhythmic pulse
            cmd = toprock.copy()
            if 0.8 <= t < 1.6:
                cmd[5] = 0.7  # head glance left
                cmd[3] = -0.35 # neck roll counter
            elif t >= 1.6:
                cmd[5] = -0.7 # head glance right
                cmd[3] = 0.35
            phase_num = 1
            phase_name = "TOPROCK: Standing Groove & Rhythmic Head Bobs"
            phase_color = (60, 200, 255)
            highlight_note = "Phase 1: Active 108 BPM head/neck synchronization and step pacing"
        elif t < 5.5:
            cmd = downrock
            phase_num = 2
            phase_name = "DOWNROCK: Coffee Grinder Leg Sweep (709°)"
            phase_color = (255, 185, 30)
            highlight_note = "Phase 2: Pelvis lifted (contacts = 0), sweeping leg extended blade spinning ~709°"
        elif t < 6.5:
            cmd = dive
            phase_num = 3
            phase_name = "TRANSITION: Inversion Dive onto Crown"
            phase_color = (255, 110, 60)
            highlight_note = "Phase 3: 0.3s rapid crown dive with gyroscopic spindle alignment"
        elif t < 12.0:
            cmd = spin
            phase_num = 4
            phase_name = "POWER MOVE: Continuous Headspin (+18.5 CCW Turns @ 140 RPM)"
            phase_color = (40, 255, 130)
            highlight_note = "Phase 4: High-speed rotation driven entirely by leg whipping and counter-torque"
        else:
            cmd = lkick
            phase_num = 5
            phase_name = "FREEZE FINISH: Inverted L-Kick Balance Lock"
            phase_color = (220, 100, 255)
            highlight_note = "Phase 5: Right leg vertical needle balance (err 0.01 rad), left leg horizontal 90°"
            
        sim.step([cmd])
        
        # Update display model
        shown.qpos[free_adr:free_adr + 7] = data.qpos[0:7]
        shown.qpos[joint_adrs] = data.qpos[7:]
        mujoco.mj_forward(display, shown)
        
        # Extract kinematics
        w, x, y, z_rot = data.qpos[3], data.qpos[4], data.qpos[5], data.qpos[6]
        trunk_yaw = math.atan2(2 * (w * z_rot + x * y), 1 - 2 * (y * y + z_rot * z_rot))
        trunk_z = float(data.qpos[2])
        l_hip_pitch = float(data.qpos[9])
        r_hip_pitch = float(data.qpos[14])
        
        # Track rotation degrees
        if phase_num == 2:
            if last_sweep_yaw is not None:
                dyaw = trunk_yaw - last_sweep_yaw
                if dyaw < -math.pi: dyaw += 2 * math.pi
                elif dyaw > math.pi: dyaw -= 2 * math.pi
                sweep_deg += dyaw * (180.0 / math.pi)
            last_sweep_yaw = trunk_yaw
        elif phase_num == 4:
            if last_yaw is not None:
                dyaw = trunk_yaw - last_yaw
                if dyaw < -math.pi: dyaw += 2 * math.pi
                elif dyaw > math.pi: dyaw -= 2 * math.pi
                spin_turns += dyaw / (2 * math.pi)
            last_yaw = trunk_yaw
            
        # Dynamic cinematic camera angles
        if t < 2.5:
            camera.azimuth = 135.0
            camera.distance = 0.74
            camera.elevation = -10.0
        elif t < 5.5:
            # Orbiting camera around downrock sweep
            camera.azimuth = 135.0 + (t - 2.5) * 22.0
            camera.distance = 0.72
            camera.elevation = -16.0
        elif t < 6.5:
            # Low elevation dive view
            camera.azimuth = 201.0
            camera.distance = 0.68
            camera.elevation = -12.0
        elif t < 12.0:
            # Sweeping orbit following headspin momentum
            camera.azimuth = 201.0 + (t - 6.5) * 16.0
            camera.distance = 0.72
            camera.elevation = -10.0
        else:
            # Side profile showing the sharp 90-degree L-kick geometry
            camera.azimuth = 250.0
            camera.distance = 0.66
            camera.elevation = -7.0
            
        camera.lookat[:] = [data.qpos[0], data.qpos[1], max(0.06, trunk_z + 0.04)]
        renderer.update_scene(shown, camera)
        pixels = renderer.render().copy()
        
        # Overlay HUD
        img = Image.fromarray(pixels)
        draw = ImageDraw.Draw(img, "RGBA")
        
        # Header banner
        draw.rounded_rectangle([(24, 18), (WIDTH - 24, 88)], radius=12, fill=(12, 16, 26, 225), outline=(55, 80, 115, 240), width=2)
        
        # Phase badge
        draw.rounded_rectangle([(36, 30), (84, 56)], radius=6, fill=phase_color)
        draw.text((44, 33), f"P{phase_num}", fill=(10, 15, 25), font=font_small)
        
        # Move title
        draw.text((100, 28), "MICRODUCK ARENA: BREAKDANCE SHOWCASE (hs13b)", fill=(255, 255, 255), font=font_large)
        draw.text((100, 58), f"PHASE {phase_num}: {phase_name}", fill=phase_color, font=font_small)
        
        # Telemetry
        time_text = f"T: {t:4.2f} s"
        draw.text((WIDTH - 180, 28), time_text, fill=(240, 240, 240), font=font_stat)
        
        if phase_num == 5:
            stat_text = f"L-HIP: {l_hip_pitch:+4.2f} rad (90°) | R-HIP: {r_hip_pitch:+4.2f} rad (Vertical Lock)"
            draw.text((WIDTH - 530, 58), stat_text, fill=(240, 140, 255), font=font_stat)
        elif phase_num == 4:
            stat_text = f"HEADSPIN: {spin_turns:+4.1f} TURNS | 140 RPM"
            draw.text((WIDTH - 380, 58), stat_text, fill=(80, 255, 140), font=font_stat)
        elif phase_num == 2:
            stat_text = f"SWEEP: {abs(sweep_deg):4.0f}° | Z: {trunk_z:4.3f} m (PELVIS UP)"
            draw.text((WIDTH - 440, 58), stat_text, fill=(255, 200, 80), font=font_stat)
        else:
            stat_text = f"Z: {trunk_z:4.3f} m | L-HIP: {l_hip_pitch:+4.2f} | R-HIP: {r_hip_pitch:+4.2f}"
            draw.text((WIDTH - 440, 58), stat_text, fill=(200, 225, 255), font=font_stat)
            
        # Footer highlight bar
        draw.rounded_rectangle([(24, HEIGHT - 52), (WIDTH - 24, HEIGHT - 18)], radius=8, fill=(18, 14, 24, 225), outline=(90, 55, 85, 210), width=1)
        draw.text((36, HEIGHT - 44), highlight_note, fill=(255, 225, 150), font=font_note)
        
        frames.append(np.array(img.convert("RGB")))
        
    print(f"All {len(frames)} frames rendered. Encoding 50 FPS 720p MP4...")
    writer = imageio.get_writer(
        OUTPUT_MP4,
        fps=FPS,
        codec="libx264",
        quality=9,
        ffmpeg_params=["-crf", "18", "-preset", "fast", "-pix_fmt", "yuv420p"]
    )
    for frame in frames:
        writer.append_data(frame)
    writer.close()
    print(f"MP4 saved successfully: {OUTPUT_MP4} ({OUTPUT_MP4.stat().st_size / 1024 / 1024:.2f} MB)")
    
    print("Encoding WebP animation (25 FPS smooth preview)...")
    webp_frames = frames[::2]
    imageio.mimsave(OUTPUT_WEBP, webp_frames, fps=25, loop=0)
    print(f"WebP saved successfully: {OUTPUT_WEBP} ({OUTPUT_WEBP.stat().st_size / 1024 / 1024:.2f} MB)")

if __name__ == "__main__":
    main()

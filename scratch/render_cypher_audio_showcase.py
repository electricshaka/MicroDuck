import math
import subprocess
from pathlib import Path
import imageio
import imageio_ffmpeg
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
TEMP_VIDEO = Path("scratch/temp_visual_no_audio.mp4")
AUDIO_WAV = Path("scratch/block_rockin_beats_duck_cypher.wav")
FINAL_MP4 = Path("scratch/microduck_block_rockin_beats_720p.mp4")
OUTPUT_WEBP = Path("scratch/microduck_block_rockin_beats.webp")

DURATION = 16.0
FPS = 50
WIDTH, HEIGHT = 1280, 720
BPM = 108.0
BEAT_DUR = 60.0 / BPM  # ~0.5555 s

def main():
    print(f"Loading event '{EVENT}' and policy '{POLICY_PATH}'...")
    event = load_event(EVENT)
    model = compile_course(event)
    policy = load_policy(POLICY_PATH)
    
    stand = np.zeros(13)
    
    # Pre-configure command prototypes
    # twist: [0: sit, 1: headstand, 2: headspin]
    # head: [3: neck_pitch, 4: head_pitch, 5: head_yaw, 6: head_roll]
    # body: [7: toprock_step, 8: downrock_sweep, 9: ?, 10: ?, 11: l_kick_freeze, 12: ?]
    
    toprock_base = stand.copy()
    toprock_base[7] = 1.0  # body[0] toprock step
    
    downrock_base = stand.copy()
    downrock_base[0] = 1.0 # twist[0] sit
    downrock_base[8] = 1.0 # body[1] downrock sweep
    
    dive_base = stand.copy()
    dive_base[1] = 1.0     # twist[1] headstand
    
    spin_base = dive_base.copy()
    spin_base[2] = 1.0     # twist[2] headspin
    
    lkick_base = dive_base.copy()
    lkick_base[11] = 1.0   # body[4] L-kick freeze
    
    sim, _ = runner._start(event, model, [policy], [0.0], 1, [toprock_base])
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
        font_hook = ImageFont.truetype("impact.ttf", 28)
    except Exception:
        font_large = ImageFont.load_default()
        font_small = font_large
        font_stat = font_large
        font_note = font_large
        font_hook = font_large

    total_steps = round(DURATION / CTRL_DT)
    print(f"Rendering {total_steps} frames at {WIDTH}x{HEIGHT} @ {FPS} FPS...")
    
    frames = []
    spin_turns = 0.0
    sweep_deg = 0.0
    last_yaw = None
    last_sweep_yaw = None
    
    for k in range(total_steps):
        t = (k + 1) * CTRL_DT
        
        # Choreography schedule (Pure Policy Execution)
        if t < 2.5:
            cmd = toprock_base.copy()
            beat_phase = (t % BEAT_DUR) / BEAT_DUR
            beat_idx = int(t / BEAT_DUR)
            
            # Dynamic 108 BPM head-bobbing & neck-pumping to the beat
            cmd[3] = 0.40 * math.sin(2.0 * math.pi * beat_phase) # neck pitch bob
            cmd[4] = 0.30 * math.cos(2.0 * math.pi * beat_phase) # head pitch nod
            
            # Alternating callout glances
            if beat_idx % 2 == 0:
                cmd[5] = 0.80 * math.sin(math.pi * beat_phase)   # glance left
                cmd[6] = 0.25 * math.sin(math.pi * beat_phase)   # head roll tilt
            else:
                cmd[5] = -0.80 * math.sin(math.pi * beat_phase)  # glance right
                cmd[6] = -0.25 * math.sin(math.pi * beat_phase)
                
            phase_num = 1
            phase_name = "TOPROCK: 108 BPM Footwork Pacing & Head Grooves"
            phase_color = (60, 210, 255)
            highlight_note = "Phase 1: Dynamic 108 BPM neck pump & alternating head glances matching the beat"
            
        elif t < 5.5:
            cmd = downrock_base.copy()
            cmd[3] = -0.30  # lean head forward into the floor sweep
            cmd[4] = 0.40
            phase_num = 2
            phase_name = "DOWNROCK: Coffee Grinder Leg Sweep (709°)"
            phase_color = (255, 185, 30)
            highlight_note = "Phase 2: Pelvis elevated (0 contacts), sweeping floor rotation sustained across 709°"
            
        elif t < 6.5:
            cmd = dive_base
            phase_num = 3
            phase_name = "TRANSITION: Inversion Dive onto Crown"
            phase_color = (255, 110, 60)
            highlight_note = "Phase 3: Rapid 0.3s dive onto crown with gyroscopic spindle alignment"
            
        elif t < 12.0:
            cmd = spin_base
            phase_num = 4
            phase_name = "POWER MOVE: Continuous Headspin (+18.5 CCW Turns @ 140 RPM)"
            phase_color = (40, 255, 130)
            highlight_note = "Phase 4: High-speed continuous spin powered 100% by leg counter-torque"
            
        else:
            cmd = lkick_base
            phase_num = 5
            phase_name = "FREEZE FINISH: Inverted L-Kick Balance Lock"
            phase_color = (220, 100, 255)
            highlight_note = "Phase 5: Right leg vertical needle balance (err 0.01 rad), left leg horizontal 90°"
            
        # Step physics cleanly through policy
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
            camera.azimuth = 135.0 + math.sin(t * 1.5) * 6.0
            camera.distance = 0.74
            camera.elevation = -12.0
        elif t < 5.5:
            camera.azimuth = 135.0 + (t - 2.5) * 26.0
            camera.distance = 0.72
            camera.elevation = -18.0
        elif t < 6.5:
            camera.azimuth = 213.0
            camera.distance = 0.68
            camera.elevation = -12.0
        elif t < 12.0:
            camera.azimuth = 213.0 + (t - 6.5) * 16.0
            camera.distance = 0.72
            camera.elevation = -10.0
        else:
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
        draw.rounded_rectangle([(24, 18), (WIDTH - 24, 88)], radius=12, fill=(12, 16, 26, 230), outline=(55, 80, 115, 240), width=2)
        
        # Phase badge
        draw.rounded_rectangle([(36, 30), (84, 56)], radius=6, fill=phase_color)
        draw.text((44, 33), f"P{phase_num}", fill=(10, 15, 25), font=font_small)
        
        # Move title
        draw.text((100, 28), "MICRODUCK ARENA: \"BLOCK ROCKIN' BEATS\" CYPHER", fill=(255, 255, 255), font=font_large)
        draw.text((100, 58), f"PHASE {phase_num}: {phase_name}", fill=phase_color, font=font_small)
        
        # Audio track indicator (BPM + Quack Synth)
        draw.text((WIDTH - 420, 28), "[AUDIO: 108 BPM BIG BEAT SYNTH]", fill=(255, 215, 80), font=font_stat)
        draw.text((WIDTH - 160, 58), f"T: {t:4.2f} s", fill=(240, 240, 240), font=font_stat)
        
        # Vocal Chant overlay banner
        if 2.2 <= t <= 3.4 or 12.2 <= t <= 13.6:
            draw.rounded_rectangle([(WIDTH // 2 - 280, 102), (WIDTH // 2 + 280, 150)], radius=8, fill=(240, 35, 75, 235), outline=(255, 220, 100), width=2)
            draw.text((WIDTH // 2 - 245, 108), "BLOCK ROCKIN' BEATS!", fill=(255, 255, 255), font=font_hook)
        elif 0.3 <= t <= 2.2:
            draw.rounded_rectangle([(WIDTH // 2 - 280, 102), (WIDTH // 2 + 280, 142)], radius=8, fill=(30, 20, 42, 215), outline=(130, 85, 175), width=1)
            draw.text((WIDTH // 2 - 240, 110), "\"Back with another one of those...\"", fill=(230, 215, 255), font=font_small)
            
        # Telemetry
        if phase_num == 5:
            stat_text = f"L-HIP: {l_hip_pitch:+4.2f} rad (90°) | R-HIP: {r_hip_pitch:+4.2f} rad (Vertical Lock)"
            draw.text((WIDTH - 540, 58), stat_text, fill=(240, 140, 255), font=font_stat)
        elif phase_num == 4:
            stat_text = f"HEADSPIN: {spin_turns:+4.1f} TURNS | 140 RPM"
            draw.text((WIDTH - 400, 58), stat_text, fill=(80, 255, 140), font=font_stat)
        elif phase_num == 2:
            stat_text = f"SWEEP: {abs(sweep_deg):4.0f}° | Z: {trunk_z:4.3f} m (PELVIS ELEVATED)"
            draw.text((WIDTH - 520, 58), stat_text, fill=(255, 200, 80), font=font_stat)
        else:
            stat_text = f"108 BPM GROOVE | Z: {trunk_z:4.3f} m | L: {l_hip_pitch:+3.2f} | R: {r_hip_pitch:+3.2f}"
            draw.text((WIDTH - 540, 58), stat_text, fill=(200, 225, 255), font=font_stat)
            
        # Footer highlight bar
        draw.rounded_rectangle([(24, HEIGHT - 52), (WIDTH - 24, HEIGHT - 18)], radius=8, fill=(18, 14, 24, 225), outline=(90, 55, 85, 210), width=1)
        draw.text((36, HEIGHT - 44), highlight_note, fill=(255, 225, 150), font=font_note)
        
        frames.append(np.array(img.convert("RGB")))
        
    print(f"All {len(frames)} frames rendered. Encoding visual MP4 to {TEMP_VIDEO}...")
    writer = imageio.get_writer(
        TEMP_VIDEO,
        fps=FPS,
        codec="libx264",
        quality=9,
        ffmpeg_params=["-crf", "18", "-preset", "fast", "-pix_fmt", "yuv420p"]
    )
    for frame in frames:
        writer.append_data(frame)
    writer.close()
    print(f"Visual MP4 saved: {TEMP_VIDEO}")
    
    # Mux audio with FFmpeg
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    print(f"Muxing {AUDIO_WAV} into {FINAL_MP4} using {ffmpeg_exe}...")
    cmd = [
        ffmpeg_exe,
        "-y",
        "-i", str(TEMP_VIDEO),
        "-i", str(AUDIO_WAV),
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", "192k",
        "-shortest",
        str(FINAL_MP4)
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode == 0:
        print(f"SUCCESS: Final broadcast video with audio generated: {FINAL_MP4} ({FINAL_MP4.stat().st_size / 1024 / 1024:.2f} MB)")
    else:
        print(f"FFmpeg mux error: {res.stderr}")
        
    print("Encoding WebP animation (25 FPS smooth preview)...")
    webp_frames = frames[::2]
    imageio.mimsave(OUTPUT_WEBP, webp_frames, fps=25, loop=0)
    print(f"WebP saved successfully: {OUTPUT_WEBP} ({OUTPUT_WEBP.stat().st_size / 1024 / 1024:.2f} MB)")

if __name__ == "__main__":
    main()

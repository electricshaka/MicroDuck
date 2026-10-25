import numpy as np
import scipy.io.wavfile as wavfile
from scipy import signal

SAMPLE_RATE = 44100
DURATION = 16.0
BPM = 108.0
BEAT_DUR = 60.0 / BPM  # ~0.5555 s
SIXTEENTH = BEAT_DUR / 4.0  # ~0.1388 s

# MIDI Note frequencies
def midi_to_freq(m):
    return 440.0 * (2.0 ** ((m - 69) / 12.0))

# Notes: D2=38, F2=41, G2=43, G#2=44, A2=45, C3=48, D3=50, F3=53, G3=55, A3=57
D2, F2, G2, Gs2, A2 = 38, 41, 43, 44, 45
C3, D3, F3, G3, A3 = 48, 50, 53, 55, 57
D4, F4, G4, A4 = 62, 65, 67, 69
D5, A5 = 74, 81

total_samples = int(SAMPLE_RATE * DURATION)
t = np.linspace(0, DURATION, total_samples, endpoint=False)

def synth_kick(start_t, length=0.25):
    idx_start = int(start_t * SAMPLE_RATE)
    idx_end = min(total_samples, idx_start + int(length * SAMPLE_RATE))
    n = idx_end - idx_start
    if n <= 0: return np.zeros(0), 0, 0
    sub_t = np.linspace(0, length, n, endpoint=False)
    # Pitch drop from 140Hz to 40Hz
    freq_env = 40.0 + 100.0 * np.exp(-sub_t * 28.0)
    phase = 2.0 * np.pi * np.cumsum(freq_env) / SAMPLE_RATE
    amp_env = np.exp(-sub_t * 12.0)
    # Click transient
    click = np.random.randn(n) * np.exp(-sub_t * 80.0) * 0.3
    kick = (np.sin(phase) + click) * amp_env
    # Soft saturation
    kick = np.tanh(kick * 1.8)
    return kick, idx_start, idx_end

def synth_snare_quack(start_t, length=0.32):
    idx_start = int(start_t * SAMPLE_RATE)
    idx_end = min(total_samples, idx_start + int(length * SAMPLE_RATE))
    n = idx_end - idx_start
    if n <= 0: return np.zeros(0), 0, 0
    sub_t = np.linspace(0, length, n, endpoint=False)
    
    # Dual formant resonant quack (700Hz and 1800Hz)
    body = np.sin(2.0 * np.pi * 320.0 * sub_t) * 0.6 + np.sin(2.0 * np.pi * 720.0 * sub_t) * 0.8
    # Noise snap (beak mechanical crack)
    noise = np.random.randn(n)
    sos = signal.butter(4, [600, 3200], btype='bandpass', fs=SAMPLE_RATE, output='sos')
    filtered_noise = signal.sosfilt(sos, noise)
    
    amp = np.exp(-sub_t * 14.0)
    snap = (body + filtered_noise * 1.2) * amp
    snap = np.tanh(snap * 1.5)
    return snap, idx_start, idx_end

def synth_acid_bass_note(start_t, midi_note, length=0.12):
    idx_start = int(start_t * SAMPLE_RATE)
    idx_end = min(total_samples, idx_start + int(length * SAMPLE_RATE))
    n = idx_end - idx_start
    if n <= 0: return np.zeros(0), 0, 0
    sub_t = np.linspace(0, length, n, endpoint=False)
    f = midi_to_freq(midi_note)
    
    # Sawtooth + sub sine
    saw = 2.0 * (sub_t * f - np.floor(sub_t * f + 0.5))
    sub = np.sin(2.0 * np.pi * (f / 2.0) * sub_t) * 0.6
    raw = saw + sub
    
    # Resonant filter sweep (acid 303 squelch)
    cutoff = 400.0 + 1800.0 * np.exp(-sub_t * 30.0)
    # Fast approximation of envelope-swept lowpass
    env = np.exp(-sub_t * 10.0)
    note = np.tanh(raw * 2.2) * env
    return note, idx_start, idx_end

def synth_laser_siren(start_t=3.8, end_t=5.8):
    idx_start = int(start_t * SAMPLE_RATE)
    idx_end = min(total_samples, int(end_t * SAMPLE_RATE))
    n = idx_end - idx_start
    sub_t = np.linspace(0, end_t - start_t, n, endpoint=False)
    # Glissando D5 (587Hz) -> A5 (880Hz) with 6Hz vibrato
    f_center = np.linspace(587.0, 920.0, n)
    vibrato = np.sin(2.0 * np.pi * 6.5 * sub_t) * 35.0
    f = f_center + vibrato
    phase = 2.0 * np.pi * np.cumsum(f) / SAMPLE_RATE
    siren = np.sin(phase) + 0.3 * np.sin(2.0 * phase)
    # Swell in and fade
    fade = np.minimum(sub_t / 0.4, 1.0) * np.minimum((sub_t[-1] - sub_t) / 0.4, 1.0)
    return siren * fade * 0.45, idx_start, idx_end

def synth_duck_vocal_word(start_t, text, pitch_midi, length):
    idx_start = int(start_t * SAMPLE_RATE)
    idx_end = min(total_samples, idx_start + int(length * SAMPLE_RATE))
    n = idx_end - idx_start
    if n <= 0: return np.zeros(0), 0, 0
    sub_t = np.linspace(0, length, n, endpoint=False)
    f0 = midi_to_freq(pitch_midi)
    
    # Duck vocal buzz (pulse train + harmonics)
    buzz = np.zeros(n)
    for h in range(1, 14):
        h_amp = 1.0 / (h ** 0.8)
        buzz += np.sin(2.0 * np.pi * f0 * h * sub_t) * h_amp
        
    # Formants for duck chorale quack ("ah", "eh", "oh")
    if "BLOCK" in text or "BACK" in text:
        # Formants at 800 Hz and 1300 Hz ("ah")
        formant = np.sin(2.0 * np.pi * 820.0 * sub_t) * 0.9 + np.sin(2.0 * np.pi * 1350.0 * sub_t) * 0.7
    elif "ROCK" in text:
        # Formants at 650 Hz and 1100 Hz ("oh")
        formant = np.sin(2.0 * np.pi * 650.0 * sub_t) * 0.8 + np.sin(2.0 * np.pi * 1100.0 * sub_t) * 0.6
    elif "BEATS" in text:
        # Formants at 450 Hz and 2200 Hz ("ee")
        formant = np.sin(2.0 * np.pi * 480.0 * sub_t) * 0.8 + np.sin(2.0 * np.pi * 2250.0 * sub_t) * 0.9
    else:
        formant = np.sin(2.0 * np.pi * 700.0 * sub_t) * 0.7 + np.sin(2.0 * np.pi * 1500.0 * sub_t) * 0.6
        
    amp = np.sin(np.pi * np.clip(sub_t / length, 0.0, 1.0)) ** 0.5
    vocal = (buzz * 0.4 + formant * 0.7) * amp
    # Duck nasal overdrive
    vocal = np.tanh(vocal * 2.0)
    return vocal, idx_start, idx_end

def generate_track():
    print(f"Generating 108 BPM Big Beat audio score for {DURATION}s...")
    audio = np.zeros(total_samples)
    
    # 1. 808 Acid Bassline & Drums Loop
    # Chemical Brothers Big Beat D-minor pattern across 7 bars (28 beats)
    num_beats = int(DURATION / BEAT_DUR)
    
    for beat in range(num_beats):
        b_time = beat * BEAT_DUR
        
        # Kick on 1 and 3.5
        if beat % 4 == 0 or beat % 4 == 2.5:
            k, s, e = synth_kick(b_time)
            audio[s:e] += k * 0.85
            
        # Beakbox Snare quack on 2 and 4
        if beat % 4 == 1 or beat % 4 == 3:
            sn, s, e = synth_snare_quack(b_time)
            audio[s:e] += sn * 0.95
            
        # 16th-note Acid Bass Riff: D2, D2, F2, G2, G#2, A2
        sub_pat = [
            (0, D2), (1, D2), (2, F2), (3, G2),
            (0, Gs2), (1, A2), (2, G2), (3, F2)
        ]
        sub_step = (beat % 2) * 4
        for step in range(4):
            note_idx = sub_step + step
            n_time = b_time + step * SIXTEENTH
            if n_time + 0.1 < DURATION:
                _, m = sub_pat[note_idx]
                bass, s, e = synth_acid_bass_note(n_time, m, length=SIXTEENTH * 0.85)
                audio[s:e] += bass * 0.75
                
    # 2. Laser Siren build into headspin drop
    siren, s, e = synth_laser_siren(3.6, 5.8)
    audio[s:e] += siren
    
    # Second laser siren during headspin power peak
    siren2, s, e = synth_laser_siren(8.8, 11.2)
    audio[s:e] += siren2 * 0.8
    
    # 3. Duck Vocal Hook Chant: "Back with another one of those... BLOCK ROCKIN' BEATS!"
    # Hook Call 1 (Opening Toprock bar: 0.2s - 2.8s)
    words_hook1 = [
        ("Back", 0.35, D4, 0.22),
        ("with an-", 0.65, F4, 0.24),
        ("oth-er", 0.95, G4, 0.22),
        ("one of", 1.25, F4, 0.20),
        ("those...", 1.50, D4, 0.40),
        ("BLOCK", 2.22, D4, 0.26),
        ("ROCK-IN'", 2.52, F4, 0.26),
        ("BEATS!", 2.82, A4, 0.55),
    ]
    for txt, t_start, pitch, dur in words_hook1:
        voc, s, e = synth_duck_vocal_word(t_start, txt, pitch, dur)
        audio[s:e] += voc * 0.80

    # Hook Call 2 (Freeze Drop: 12.0s - 14.5s)
    words_hook2 = [
        ("BLOCK", 12.22, D4, 0.28),
        ("ROCK-IN'", 12.55, F4, 0.28),
        ("BEATS!", 12.88, A4, 0.70),
    ]
    for txt, t_start, pitch, dur in words_hook2:
        voc, s, e = synth_duck_vocal_word(t_start, txt, pitch, dur)
        audio[s:e] += voc * 0.90
        # Add vocal delay echo
        voc_echo, s_echo, e_echo = synth_duck_vocal_word(t_start + 0.28, txt, pitch, dur)
        if s_echo < total_samples:
            audio[s_echo:e_echo] += voc_echo * 0.35

    # Normalize audio to -1.0 dB
    max_amp = np.max(np.abs(audio))
    if max_amp > 0:
        audio = (audio / max_amp) * 0.92
        
    out_wav = "scratch/block_rockin_beats_duck_cypher.wav"
    wavfile.write(out_wav, SAMPLE_RATE, (audio * 32767).astype(np.int16))
    print(f"Generated soundtrack successfully: {out_wav} ({DURATION}s, {SAMPLE_RATE}Hz)")
    return out_wav

if __name__ == "__main__":
    generate_track()

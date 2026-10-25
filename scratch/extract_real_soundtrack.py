import numpy as np
import scipy.io.wavfile as wavfile

sr, data = wavfile.read('scratch/full_block_rockin_beats.wav')
# Normalize float
if data.dtype == np.int16:
    audio = data.astype(np.float32) / 32768.0
else:
    audio = data.astype(np.float32)

DURATION = 16.0
n_samples = int(DURATION * sr)

def extract_and_fade(start_sec, out_path):
    start_idx = int(start_sec * sr)
    end_idx = start_idx + n_samples
    clip = audio[start_idx:end_idx].copy()
    
    # 0.05s fade in, 0.4s fade out
    fade_in_len = int(0.05 * sr)
    fade_out_len = int(0.40 * sr)
    
    fade_in = np.linspace(0.0, 1.0, fade_in_len)[:, None]
    fade_out = np.linspace(1.0, 0.0, fade_out_len)[:, None]
    
    clip[:fade_in_len] *= fade_in
    clip[-fade_out_len:] *= fade_out
    
    # Max peak normalize to -0.5 dB
    peak = np.max(np.abs(clip))
    if peak > 0:
        clip = (clip / peak) * 0.94
        
    wavfile.write(out_path, sr, (clip * 32767).astype(np.int16))
    print(f"Extracted {out_path}: {start_sec:.2f}s to {start_sec+DURATION:.2f}s")

# Cut Candidate 1: Starts at 26.2s (Vocal hits at ~2.2s into Downrock drop)
extract_and_fade(26.2, 'scratch/block_rockin_beats_real_drop1.wav')

# Cut Candidate 2: Starts at 35.2s (Vocal hits at ~1.5s, massive bassline drops at ~4.5s)
extract_and_fade(35.2, 'scratch/block_rockin_beats_real_drop2.wav')

# Cut Candidate 3: Starts at 27.5s (Vocal "...another one of those block rock n beats" starts right on beat 1)
extract_and_fade(27.5, 'scratch/block_rockin_beats_real_vocal_lead.wav')

from Synth import Synth
from Utils import mel_spectrogram, MSE
from globals import SAMPLE_RATE
import numpy as np

def evaluate_presets(data):
    presets, target_M, duration = data
    
    synth = Synth(
        sample_rate=SAMPLE_RATE,
        duration=duration,
        presets=presets
    )
    audio = synth.process_audio().astype(np.float64)

    current_M = mel_spectrogram(audio, sr=SAMPLE_RATE, n_fft=2048, hop_length=256, n_mels=128)
    return MSE(current_M, target_M)
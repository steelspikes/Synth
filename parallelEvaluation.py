from Synth import Synth
from Utils import mfcc, mel_spectrogram, MSE
from globals import SAMPLE_RATE, DURATION
import numpy as np

def evaluate_presets(data):
    presets, target_C = data
    synth = Synth(
        sample_rate=SAMPLE_RATE,
        duration=DURATION,
        presets=presets
    )
    audio = synth.process_audio()
    # return mfcc(audio, sr=SAMPLE_RATE)
    C1 = mel_spectrogram(audio, sr=SAMPLE_RATE)
    return MSE(C1, target_C)
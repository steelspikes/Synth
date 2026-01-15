from Synth import Synth
from Utils import mfcc, mel_spectrogram, MSE, spectrogram
from globals import SAMPLE_RATE, DURATION
import numpy as np
import librosa

def evaluate_presets(data):
    presets, target_C = data
    target_spec, target_mel, target_audio = target_C
    synth = Synth(
        sample_rate=SAMPLE_RATE,
        duration=DURATION,
        presets=presets
    )
    audio = synth.process_audio().astype(np.float64)
    # return mfcc(audio, sr=SAMPLE_RATE)
    current_spect = spectrogram(audio)
    # current_mel = mel_spectrogram(audio, sr=SAMPLE_RATE)
    # return 0.5*MSE(current_spect, target_spec) + 0.5*

    # env_synth = librosa.feature.rms(y=audio, frame_length=256)
    # env_target = librosa.feature.rms(y=target_audio, frame_length=256)
    # env_loss = np.mean(np.abs(env_synth - env_target))

    return MSE(current_spect, target_spec) #+ 0.01*env_loss
from Synth import Synth
from Utils import mfcc, mel_spectrogram, MSE, spectrogram
from globals import SAMPLE_RATE, DURATION
import numpy as np
import librosa

def evaluate_presets(data):
    presets, target_C, multi = data
    target_A, target_B, target_C = target_C
    synth = Synth(
        sample_rate=SAMPLE_RATE,
        duration=DURATION,
        presets=presets
    )
    audio = synth.process_audio().astype(np.float64)
    # current_mfcc = mfcc(audio, sr=SAMPLE_RATE)
    # current_spect = spectrogram(audio)
    # current_mel = mel_spectrogram(audio, sr=SAMPLE_RATE)
    # return 0.5*MSE(current_spect, target_spec) + 0.5*

    # env_synth = librosa.feature.rms(y=audio, frame_length=256)
    # env_target = librosa.feature.rms(y=target_audio, frame_length=256)
    # env_loss = np.mean(np.abs(env_synth - env_target))

    if multi:
        current_A, current_B, current_C = (
            mel_spectrogram(audio, sr=SAMPLE_RATE, n_fft=1024, hop_length=128, n_mels=64), 
            mel_spectrogram(audio, sr=SAMPLE_RATE, n_fft=2048, hop_length=512, n_mels=128),
            mel_spectrogram(audio, sr=SAMPLE_RATE, n_fft=4096, hop_length=256, n_mels=256)
        )

        return 0.2*MSE(current_A, target_A) + 0.4*MSE(current_B, target_B) + 0.4*MSE(current_C, target_C)
    else:
        current_B = mel_spectrogram(audio, sr=SAMPLE_RATE, n_fft=4096, hop_length=256, n_mels=256)
        return MSE(current_B, target_B)
from Synth import Synth
from Utils import mfcc
from globals import SAMPLE_RATE, DURATION

def evaluate_presets(data):
    presets, normalized = data
    synth = Synth(
        sample_rate=SAMPLE_RATE,
        duration=DURATION,
        presets=presets,
        is_normalized=normalized
    )
    audio = synth.process_audio()
    mfcc_coefs = mfcc(audio, n_mfcc=13, n_mels=26, n_fft=1024, hop_length=256)

    return mfcc_coefs
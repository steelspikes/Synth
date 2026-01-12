from Synth import Synth
from Utils import mfcc, mel_spectrogram
from globals import SAMPLE_RATE, DURATION

def evaluate_presets(presets):
    synth = Synth(
        sample_rate=SAMPLE_RATE,
        duration=DURATION,
        presets=presets
    )
    audio = synth.process_audio()
    return mfcc(audio, sr=SAMPLE_RATE)
    # return mel_spectrogram(audio, sr=SAMPLE_RATE)
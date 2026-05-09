from Synth.main import Synth
from Libs.Utils import mel_spectrogram, MAE
from Libs.globals import SAMPLE_RATE
import numpy as np

# Se usa como worker en multiprocessing; recibe un tuple para poder pasar varios args con pool.map
def evaluate_presets(data):
    presets, target_M, duration = data
    
    synth = Synth(
        sample_rate=SAMPLE_RATE,
        duration=duration,
        presets=presets
    )
    audio = synth.process_audio().astype(np.float64)

    # Comparamos el espectrograma del audio generado contra el objetivo
    current_M = mel_spectrogram(audio, sr=SAMPLE_RATE, n_fft=2048, hop_length=256, n_mels=128)
    return MAE(current_M, target_M)
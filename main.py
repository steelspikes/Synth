import sounddevice as sd
from Utils import load_parameters_file
import time
from Synth import Synth
import librosa

DURATION = 2
SAMPLE_RATE = 44100

def load_synth(presets):
    synth = Synth(
        sample_rate=SAMPLE_RATE,
        duration=DURATION,
        presets=presets
    )
    audio = synth.process_audio()
    mfcc_coefs = mfcc(audio, n_mfcc=13, n_mels=26, n_fft=1024, hop_length=256)

    return mfcc_coefs

def mfcc(audio, sr=16000, n_mfcc=13, n_mels=26, n_fft=2048, hop_length=512):
    """
    Calcula los coeficientes MFCC de un audio.
    
    Args:
        audio (1D np.array): señal de audio.
        sr (int): sample rate.
        n_mfcc (int): número de coeficientes MFCC a retornar.
        n_mels (int): número de filtros Mel.
        n_fft (int): tamaño de la ventana FFT.
        hop_length (int): salto entre frames.
        
    Returns:
        np.array: matriz (n_mfcc x n_frames) de MFCC.
    """
    return librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=n_mfcc,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length
    )

if __name__ == "__main__":
    print('Starting...')
    import time
    from multiprocessing import Pool

    presets = [load_parameters_file(),load_parameters_file(), load_parameters_file(), load_parameters_file(),load_parameters_file()]

    with Pool(5) as p:
        for i in range(10):
            start = time.time()
            r = p.map(load_synth, presets)
            print([batch.shape for batch in r])

            end = time.time()
            print(end - start)

    # from joblib import Parallel, delayed

    # for i in range(10):
    #     start = time.time()
        
    #     # n_jobs=5 → equivalente a Pool(5)
    #     r = Parallel(n_jobs=5)(
    #         delayed(load_synth)(preset) for preset in presets
    #     )
        
    #     print([batch.shape for batch in r])
        
    #     end = time.time()
    #     print(end - start)
    
    # start = time.time()
    # audios = load_synth(load_parameters_file())
    # mfcc_coefs = mfcc(audios)
    # print(mfcc_coefs.shape)

    # end = time.time()
    # print(end - start)


# for i in audios:
#     print(i)

#     sd.play(i, samplerate=44_100)
#     sd.wait()
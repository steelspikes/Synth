import sounddevice as sd
from Utils import load_parameters_file
import time
from Synth import Synth

DURATION = 2
SAMPLE_RATE = 44100

def load_synth(presets):
    synth = Synth(
        sample_rate=SAMPLE_RATE,
        duration=DURATION,
        presets=presets
    )
    return synth.process_audio()


if __name__ == "__main__":
    import time
    from multiprocessing import Pool

    

    presets = [load_parameters_file(), load_parameters_file(), load_parameters_file(), load_parameters_file(), load_parameters_file()]

    start = time.time()

    with Pool(5) as p:
        for i in range(10):
            r = p.map(load_synth, presets)
            print(sum(len(batch) for batch in r))
    
    # load_synth(load_parameters_file())

    end = time.time()

    print(end - start)


# for i in audios:
#     print(i)

#     sd.play(i, samplerate=44_100)
#     sd.wait()
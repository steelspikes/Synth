from Oscillator import Oscillator
import numpy as np
# import matplotlib.pyplot as plt
import sounddevice as sd
from Utils import load_parameters_file, plot_spectrum_linear
from LFO import LFO
from BiquadFilter import BiquadFilter
from NoiseOscillator import NoiseOscillator
from Envelope import Envelope
import time
import json

DURATION = 2
SAMPLE_RATE = 44100

def ejecuta_proceso(params):
    NUM_PRESETS = 100

    (
        osc1_shape,
        osc1_phase,
        osc1_volume,
        osc1_freq,
        osc1_vdepth,
        osc1_pdepth,

        osc2_shape,
        osc2_phase,
        osc2_volume,
        osc2_freq,
        osc2_vdepth,
        osc2_pdepth,

        osc3_shape,
        osc3_phase,
        osc3_volume,
        osc3_freq,
        osc3_vdepth,
        osc3_pdepth,

        osc4_shape,
        osc4_phase,
        osc4_volume,
        osc4_freq,
        osc4_vdepth,
        osc4_pdepth,

        oscnoise_volume,

        base_cutoff_hz,
        base_q,
        filter_type, # 0 = HPF, 1 = BPF, 2 = LPF
        envelope_depth,
        cutoff_mod_depth,

        lfo1_rate,
        lfo1_shape,

        lfo2_rate,
        lfo2_shape,

        envelope_decay,
        envelope_attack,
        envelope_sustain,
        envelope_release,

        filter_envelope_attack,
        filter_envelope_decay,    
        filter_envelope_sustain,
        filter_envelope_release
    ) = load_parameters_file()

    # print('start')

    lfo1 = LFO(rate_hz=lfo1_rate, shape=lfo1_shape)
    lfo2 = LFO(rate_hz=lfo2_rate, shape=lfo2_shape)

    lfo_signal = lfo1.process(int(SAMPLE_RATE * DURATION))

    osc1 = Oscillator(
        shape=osc1_shape, 
        phase=osc1_phase, 
        volume=osc1_volume, 
        initial_freq=osc1_freq, 
        lfo_signal=lfo_signal,
        sample_rate=SAMPLE_RATE, 
        duration=DURATION,
        volume_mod_depth=osc1_vdepth,
        pitch_mod_depth=osc1_pdepth
    )

    osc2 = Oscillator(
        shape=osc2_shape, 
        phase=osc2_phase, 
        volume=osc2_volume, 
        initial_freq=osc2_freq, 
        lfo_signal=lfo_signal,
        sample_rate=SAMPLE_RATE, 
        duration=DURATION,
        volume_mod_depth=osc2_vdepth,
        pitch_mod_depth=osc2_pdepth
    )

    osc3 = Oscillator(
        shape=osc3_shape, 
        phase=osc3_phase, 
        volume=osc3_volume, 
        initial_freq=osc3_freq, 
        lfo_signal=lfo_signal,
        sample_rate=SAMPLE_RATE, 
        duration=DURATION,
        volume_mod_depth=osc3_vdepth,
        pitch_mod_depth=osc3_pdepth
    )

    osc4 = Oscillator(
        shape=osc4_shape, 
        phase=osc4_phase, 
        volume=osc4_volume, 
        initial_freq=osc4_freq, 
        lfo_signal=lfo_signal,
        sample_rate=SAMPLE_RATE, 
        duration=DURATION,
        volume_mod_depth=osc4_vdepth,
        pitch_mod_depth=osc4_pdepth
    )

    oscnoise = NoiseOscillator(
        volume=oscnoise_volume,
        sample_rate=SAMPLE_RATE, 
        duration=DURATION
    )

    envelope_filter = Envelope(
        attack=filter_envelope_attack, 
        decay=filter_envelope_decay,
        sustain=filter_envelope_sustain, 
        release=filter_envelope_release,
        sample_rate=SAMPLE_RATE
    )

    biquad_filter = BiquadFilter(
        base_cutoff_hz=base_cutoff_hz, 
        filter_type=filter_type, 
        base_q=base_q,
        lfo_instance=lfo2,
        envelope=envelope_filter,
        envelope_depth=envelope_depth,
        cutoff_mod_depth=cutoff_mod_depth,
        sample_rate=SAMPLE_RATE
    )

    envelope_amp = Envelope(
        attack=envelope_attack, 
        decay=envelope_decay,
        sustain=envelope_sustain, 
        release=envelope_release,
        sample_rate=SAMPLE_RATE
    )

    for i in range(100):
        out = osc1.process() + osc2.process() + osc3.process() + osc4.process() + oscnoise.process()
        out = biquad_filter.process(out)

        envelope_amp_signal = envelope_amp.process(DURATION)
        out = out * envelope_amp_signal

        # print('End')

        out_np = out

        out_np = out_np / np.max(np.abs(out_np))

    return out_np

if __name__ == "__main__":
    import time
    from multiprocessing import Pool

    start = time.time()

    with Pool(5) as p:
        for i in range(10):
            r = p.map(ejecuta_proceso, np.ones(5))
            print(sum(len(batch) for batch in r))

    # for i in range(5000):
    #     print(len(ejecuta_proceso(1)))


    end = time.time()

    print(end - start)


# for i in out_np:
#     # import matplotlib.pyplot as plt
#     # plt.plot(i)
#     # plt.show()

#     sd.play(i, samplerate=44_100)
#     sd.wait()
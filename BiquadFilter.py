import numpy as np
from numba import jit, prange
# from LFO import LFO
# import matplotlib.pyplot as plt
# from Envelope import Envelope

@jit(nopython=True, parallel=True)
def biquad_process_multi_preset(input_wave, modulated_cutoff, modulated_q, filter_type_code, sample_rate):
    num_presets = modulated_cutoff.shape[0]
    num_samples = input_wave.shape[1]
    output_wave = np.zeros((num_presets, num_samples)).astype(np.float32)
    
    for p in prange(num_presets):  # Loop paralelo sobre presets
        x1, x2, y1, y2 = 0.0, 0.0, 0.0, 0.0
        Q = modulated_q[p]
        filter_code = filter_type_code[p]
        
        for n in range(num_samples):
            cutoff = modulated_cutoff[p, n]
            w0 = 2 * np.pi * cutoff / sample_rate
            cos_w0 = np.cos(w0)
            alpha = np.sin(w0) / (2 * Q)

            # Coeficientes LPF
            b0_lpf = (1 - cos_w0) / 2
            b1_lpf = 1 - cos_w0
            b2_lpf = (1 - cos_w0) / 2

            # Coeficientes HPF
            b0_hpf = (1 + cos_w0) / 2
            b1_hpf = -(1 + cos_w0)
            b2_hpf = (1 + cos_w0) / 2

            # Coeficientes BPF
            b0_bpf = alpha
            b1_bpf = 0
            b2_bpf = -alpha

            # Mezclar según filter_type_code
            if filter_code <= 1.0:
                blend = filter_code
                b0 = (1.0 - blend) * b0_hpf + blend * b0_bpf
                b1 = (1.0 - blend) * b1_hpf + blend * b1_bpf
                b2 = (1.0 - blend) * b2_hpf + blend * b2_bpf
            else:
                blend = filter_code - 1.0
                b0 = (1.0 - blend) * b0_bpf + blend * b0_lpf
                b1 = (1.0 - blend) * b1_bpf + blend * b1_lpf
                b2 = (1.0 - blend) * b2_bpf + blend * b2_lpf

            a0 = 1 + alpha
            a1 = -2 * cos_w0
            a2 = 1 - alpha

            # Normalizar
            b0, b1, b2 = b0 / a0, b1 / a0, b2 / a0
            a1, a2 = a1 / a0, a2 / a0

            x0 = input_wave[p, n]
            y0 = b0*x0 + b1*x1 + b2*x2 - a1*y1 - a2*y2

            output_wave[p, n] = y0

            x2, x1 = x1, x0
            y2, y1 = y1, y0

    return output_wave

class BiquadFilter:
    def __init__(self, sample_rate=44100, base_cutoff_hz=1000, base_q=0.707, filter_type=0, lfo_instance = None, envelope_depth = 0, envelope = None, cutoff_mod_depth = 0):
        self.sample_rate = sample_rate
        self.base_cutoff_hz = base_cutoff_hz.astype(np.float32)
        self.base_q = base_q.astype(np.float32)
        self.filter_type = filter_type.astype(np.float32)
        self.lfo_instance = lfo_instance
        self.envelope = envelope
        self.envelope_depth = envelope_depth.astype(np.float32)
        self.cutoff_mod_depth = cutoff_mod_depth.astype(np.float32)
        
    def process(self, input_wave):
        presets = input_wave.shape[0]
        num_samples = input_wave.shape[1]

        # modulated_cutoff = np.exp(self.base_cutoff_hz.astype(np.float32))
        modulated_cutoff = self.base_cutoff_hz.astype(np.float32)
        modulated_cutoff = np.expand_dims(modulated_cutoff, axis=1)
        modulated_cutoff = np.broadcast_to(modulated_cutoff, (presets, num_samples))

        modulated_cutoff = np.clip(modulated_cutoff, min=20.0, max=self.sample_rate/2 - 1)

        if self.envelope is not None:
            env_signal = self.envelope.process(num_samples / self.sample_rate)
            octave_range = 5.0
            modulated_cutoff = modulated_cutoff * (2 ** (env_signal * octave_range * np.expand_dims(self.envelope_depth, axis=1).astype(np.float32)))

        if self.lfo_instance is not None:
            lfo_signal = self.lfo_instance.process(num_samples) 
            octave_range = np.expand_dims(self.cutoff_mod_depth, axis=1).astype(np.float32) * 1.0
            mod_factor = np.pow(2.0, lfo_signal * octave_range)
            modulated_cutoff = modulated_cutoff * mod_factor

        modulated_cutoff = np.clip(modulated_cutoff, min=20.0, max=20_000)

        # import matplotlib.pyplot as plt
        # plt.plot(modulated_cutoff[0])
        # plt.show()
       
        return biquad_process_multi_preset(input_wave, modulated_cutoff, self.base_q, self.filter_type, self.sample_rate)
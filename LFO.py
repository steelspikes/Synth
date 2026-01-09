import numpy as np
from Utils import create_morphed_wave, denormalize

class LFO:
    def __init__(self, sample_rate=44100, shape=0, rate_hz=0):
        self.sample_rate = sample_rate
        # self.shape = denormalize(shape, 0, 4)
        self.shape = shape
        self.rate = rate_hz
        self._phase = 0
        self.pulse_width = 0

    def process(self, num_samples):
        t = np.arange(0, num_samples) / self.sample_rate + self._phase

        freq_hz = np.exp(self.rate)
        phase = 2 * np.pi * freq_hz * t
        
        self._phase += num_samples / self.sample_rate
        
        return create_morphed_wave(self.shape, phase)
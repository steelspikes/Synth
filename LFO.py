import numpy as np
from scipy import signal
from Utils import create_morphed_wave, denormalize

class LFO:
    def __init__(self, sample_rate=44100, shape='sine', rate_hz=5.0):
        self.sample_rate = sample_rate
        self.shape = denormalize(shape, 0, 4)
        self.rate = rate_hz
        self._phase = 0
        self.pulse_width = 0

    def process(self, num_samples):
        """Genera un bloque de la señal LFO."""
        t = np.linspace(self._phase, self._phase + num_samples / self.sample_rate, num_samples, endpoint=False)
        phase = 2 * np.pi * self.rate * t
        
        # Guardar la fase final para la próxima llamada
        self._phase += num_samples / self.sample_rate
        
        return create_morphed_wave(self.shape, phase)
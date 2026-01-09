import numpy as np
from Utils import denormalize

class NoiseOscillator:
    def __init__(self, sample_rate=44100, duration = 0, volume = 0, shape=0):
        self.sample_rate = sample_rate
        self.volume = volume
        self.duration = duration
    
    def white_noise(self, n_samples):
        return np.random.randn(n_samples)

    def process(self):
        num_samples = int(self.sample_rate * self.duration)
        
        w = self.white_noise(num_samples)
        output = w
        output = output / (np.abs(output).max() + 1e-8)
            
        return output * self.volume
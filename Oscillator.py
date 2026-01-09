# import matplotlib.pyplot as plt
from Utils import create_morphed_wave
import numpy as np

class Oscillator:
    def __init__(self, sample_rate = 44100, duration = 2.0, volume = 0, shape = 0, initial_freq = 440, phase = 0, lfo_signal = None, volume_mod_depth=0, pitch_mod_depth=0):
        self.sample_rate = sample_rate
        self.duration = duration
        self.shape = shape
        self.initial_freq = initial_freq
        self.pulse_width = 0
        self.phase = phase
        self.volume = volume
        self.volume_mod_depth = volume_mod_depth
        self.pitch_mod_depth = pitch_mod_depth

        self.lfo_signal = lfo_signal

    def modulate_freq(self, depth):
        num_samples = int(self.sample_rate * self.duration)

        if self.lfo_signal is not None:
            octave_range = depth * 1.0
            mod_factor = np.pow(2.0, self.lfo_signal * octave_range)

            return mod_factor

        return np.ones(num_samples)
    
    def modulate_volume(self, depth):
        num_samples = int(self.sample_rate * self.duration)

        if self.lfo_signal is not None:
            lfo_unipolar = (self.lfo_signal + 1.0) / 2.0
            return 1.0 - (depth * (1.0 - lfo_unipolar))

        return np.ones(num_samples)

    def create_osc(self, freq):
        t = np.arange(0, int(self.sample_rate * self.duration), dtype=np.float32) / self.sample_rate
        
        phase_offset_radians = np.clip(self.phase) * 2 * np.pi

        freq_hz = np.exp(freq)

        freq_hz = freq_hz * self.modulate_freq(self.pitch_mod_depth)
        dt = t[1] - t[0]

        base_phase = 2 * np.pi * np.cumsum(freq_hz) * dt

        phase = base_phase + phase_offset_radians
    
        return create_morphed_wave(self.shape, phase)
    
    def process(self):
        waveform = self.create_osc(self.initial_freq)
        waveform = waveform * self.modulate_volume(self.volume_mod_depth)
        return waveform * self.volume
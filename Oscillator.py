# import matplotlib.pyplot as plt
from Utils import create_morphed_wave
import numpy as np

class Oscillator:
    def __init__(self, sample_rate = 44100, duration = 2.0, volume = 0, shape = 0, initial_freq = 440, phase = 0, lfo_signal = None, volume_mod_depth=0, pitch_mod_depth=0):
        self.sample_rate = sample_rate
        self.duration = duration
        self.shape = shape.astype(np.float32)
        self.initial_freq = initial_freq.astype(np.float32)
        self.pulse_width = 0
        self.phase = phase.astype(np.float32)
        self.volume = volume.astype(np.float32)
        self.volume_mod_depth = np.zeros(self.shape.shape[0]) #volume_mod_depth.astype(np.float32)
        self.pitch_mod_depth = np.zeros(self.shape.shape[0]) #pitch_mod_depth.astype(np.float32)

        self.lfo_signal = lfo_signal

    def modulate_freq(self, depth):
        presets = self.shape.shape[0]
        num_samples = int(self.sample_rate * self.duration)

        if self.lfo_signal is not None:
            octave_range = depth * 1.0
            mod_factor = np.pow(2.0, self.lfo_signal * octave_range)

            return mod_factor

        result = np.ones(num_samples)
        result = np.expand_dims(result, axis=0)
        return np.broadcast_to(result, (presets, num_samples))
    
    def modulate_volume(self, depth):
        presets = self.shape.shape[0]
        num_samples = int(self.sample_rate * self.duration)

        if self.lfo_signal is not None:
            lfo_unipolar = (self.lfo_signal + 1.0) / 2.0
            return 1.0 - (depth * (1.0 - lfo_unipolar))

        result = np.ones(num_samples)
        result = np.expand_dims(result, axis=0)
        return np.broadcast_to(result, (presets, num_samples))

    def create_osc(self, freq):
        presets = freq.shape[0]

        t = np.arange(0, int(self.sample_rate * self.duration), dtype=np.float32) / self.sample_rate
        t = np.expand_dims(t, axis=0)
        t = np.broadcast_to(t, (presets, t.shape[1]))
        
        phase_offset_radians = np.clip(self.phase) * 2 * np.pi
        phase_offset_radians = np.expand_dims(phase_offset_radians, axis=1)

        # freq_hz = np.exp(freq)
        freq_hz = freq
        freq_hz = np.expand_dims(freq_hz, axis=1)

        freq_hz = freq_hz * self.modulate_freq(np.expand_dims(self.pitch_mod_depth, axis=1))
        dt = t[:,1] - t[:,0]
        dt = np.expand_dims(dt, axis=1)

        base_phase = 2 * np.pi * np.cumsum(freq_hz, axis=1) * dt

        phase = base_phase + phase_offset_radians
    
        return create_morphed_wave(np.expand_dims(self.shape, axis=1), phase)
    
    def process(self):
        waveform = self.create_osc(self.initial_freq)
        waveform = waveform * self.modulate_volume(np.expand_dims(self.volume_mod_depth, axis=1))
        return waveform * np.expand_dims(self.volume, axis=1)
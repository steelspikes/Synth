from Utils import create_morphed_wave
import numpy as np

class Oscillator:
    def __init__(self, sample_rate = 44100, duration = 2.0, volume = 0, shape = 0, initial_freq = 440, phase = 0):
        self.sample_rate = sample_rate
        self.duration = duration
        self.shape = shape.astype(np.float32)
        self.initial_freq = initial_freq.astype(np.float32)
        self.pulse_width = 0
        self.phase = phase.astype(np.float32)
        self.volume = volume.astype(np.float32)

    def create_osc(self, freq):
        presets = freq.shape[0]

        t = np.arange(0, int(self.sample_rate * self.duration), dtype=np.float32) / self.sample_rate
        t = np.expand_dims(t, axis=0)
        t = np.broadcast_to(t, (presets, t.shape[1]))
        
        phase_offset_radians = np.clip(self.phase) * 2 * np.pi
        phase_offset_radians = np.expand_dims(phase_offset_radians, axis=1)

        freq_hz = freq
        freq_hz = np.expand_dims(freq_hz, axis=1)

        base_phase = 2 * np.pi * freq_hz * t

        phase = base_phase + phase_offset_radians
    
        return create_morphed_wave(np.expand_dims(self.shape, axis=1), phase)
    
    def process(self):
        waveform = self.create_osc(self.initial_freq)
        return waveform * np.expand_dims(self.volume, axis=1)
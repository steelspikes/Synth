import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import sounddevice as sd 
from Utils import create_morphed_wave
from Utils import denormalize

class Oscillator:
    def __init__(self, sample_rate = 44100, duration = 2.0, volume = 0, shape = 0, initial_freq = 440, phase = 0, lfo1_instance = 0, lfo2_instance = 0, volume_mod_depth=0, pitch_mod_depth=0, lfo_choose = 0):
        self.sample_rate = sample_rate
        self.duration = duration
        self.shape = denormalize(shape, 0, 4)
        self.initial_freq = initial_freq
        self.pulse_width = 0
        self.phase = phase
        self.lfo_instance = lfo1_instance if lfo_choose == 0 else lfo2_instance
        self.volume = volume
        self.volume_mod_depth = volume_mod_depth
        self.pitch_mod_depth = pitch_mod_depth

    def modulate(self, depth):
        num_samples = int(self.sample_rate * self.duration)

        if self.lfo_instance is not None:
            # 1. Generar la señal del LFO (va de -1 a 1)
            lfo_signal = self.lfo_instance.process(num_samples)
            
            # 2. Re-escalar el LFO para que vaya de 0 a 1
            # Esto es clave para que module la amplitud correctamente
            modulator = (lfo_signal + 1) / 2
            
            # 3. Aplicar la profundidad de la modulación
            # Asumimos que tienes un self.amp_lfo_depth (de 0 a 1)
            # depth = self.lfo_instance.depth
            final_modulator = (modulator * depth) + (1 - depth)

            # 4. El volumen ahora es un array dinámico
            return final_modulator

        return np.ones(num_samples)

    def create_osc(self, freq):
        t = np.linspace(0., self.duration, int(self.sample_rate * self.duration), endpoint=False)
        
        phase_offset_radians = self.phase * 2 * np.pi

        base_phase = 2 * np.pi * freq * t

        # if self.lfo_instance.dest == 'pitch':
        freq = freq * self.modulate(self.pitch_mod_depth)
        dt = t[1] - t[0]
        base_phase = 2 * np.pi * np.cumsum(freq) * dt

        # 2. Aplicar el offset a la fase de la onda
        
        phase = base_phase + phase_offset_radians

        # Usamos la duración definida en el constructor
        t = np.linspace(0., self.duration, int(self.sample_rate * self.duration), endpoint=False)

        # Nos aseguramos que el valor de 'shape' esté en el rango correcto
        shape = np.clip(self.shape, 0, 4)
    
        # --- Transición: Seno -> Triángulo (shape 0 a 1) ---
        return create_morphed_wave(shape, phase)    
    
    def process(self):
        waveform = self.create_osc(self.initial_freq)

        # if self.lfo_instance.dest == 'volume':
        waveform = waveform * self.modulate(self.volume_mod_depth)

        return waveform * self.volume
import numpy as np
from LFO import LFO
import matplotlib.pyplot as plt

class OldOscillator():
    def __init__(self, sample_rate=44_100, duration = 2.0, shape='square', initial_freq = 440.0, volume = 1.0, tune=0, fine=0, phase=0, lfo_param='', lfo_instance: LFO = None):
        self.sample_rate = sample_rate
        self.duration = duration
        self.shape = shape
        self.initial_freq = initial_freq
        self.volume = volume
        self.tune = tune
        self.fine = fine
        self.freq = initial_freq
        # self.unison_voices = unison_voices
        # self.unison_detune = unison_detune
        self.phase = phase
        self.lfo_param = lfo_param
        self.lfo_instance = lfo_instance

    def compute_delta_semitones(self):
        total_semitones = self.tune + (self.fine / 100.0)
        total_factor = 2**(total_semitones / 12.0)
        self.freq = self.initial_freq * total_factor

    def modulate(self):
        num_samples = int(self.sample_rate * self.duration)

        if self.lfo_instance is not None:
            # 1. Generar la señal del LFO (va de -1 a 1)
            lfo_signal = self.lfo_instance.process(num_samples)
            
            # 2. Re-escalar el LFO para que vaya de 0 a 1
            # Esto es clave para que module la amplitud correctamente
            modulator = (lfo_signal + 1) / 2
            
            # 3. Aplicar la profundidad de la modulación
            # Asumimos que tienes un self.amp_lfo_depth (de 0 a 1)
            depth = self.lfo_instance.depth
            final_modulator = (modulator * depth) + (1 - depth)

            # 4. El volumen ahora es un array dinámico
            return final_modulator

        return np.ones(num_samples)

    def create_oscillator(self, freq):
        # Crea un arreglo de tiempo desde 0 hasta la duración
        t = np.linspace(0., self.duration, int(self.sample_rate * self.duration), endpoint=False)

        phase_offset_radians = self.phase * 2 * np.pi

        base_phase = 2 * np.pi * freq * t

        if self.lfo_instance.dest == 'pitch':
            freq = freq * self.modulate()
            dt = t[1] - t[0]
            base_phase = 2 * np.pi * np.cumsum(freq) * dt

        # 2. Aplicar el offset a la fase de la onda
        
        phase = base_phase + phase_offset_radians
        
        # Genera la forma de onda según la selección
        if self.shape == 'sine':
            waveform = np.sin(phase)
        elif self.shape == 'square':
            waveform = np.sign(np.sin(phase))
        elif self.shape == 'saw':
            normalized_phase = phase / (2 * np.pi)
            waveform = 2 * (normalized_phase % 1.0) - 1.0
        elif self.shape == 'triangle':
            waveform = (2 / np.pi) * np.arcsin(np.sin(phase))
        elif self.shape == 'noise':
            waveform = np.random.uniform(-1.0, 1.0, size=int(self.sample_rate * self.duration))
        else:
            raise ValueError("Forma de onda no soportada. Elige entre: 'sine', 'square', 'saw', 'triangle'")
            
        if self.lfo_instance.dest == 'volume':
            waveform = waveform * self.modulate()
    
        return waveform * self.volume
    
    # def create_unison_sound_mono(self):
    #     num_samples = int(self.sample_rate * self.duration)
        
    #     # 1. Crea un buffer MONO vacío (1D)
    #     final_wave = np.zeros(num_samples)
        
    #     # Genera un espaciado lineal solo para la desafinación
    #     detune_amounts = np.linspace(-self.unison_detune, self.unison_detune, self.unison_voices)
        
    #     for i in range(self.unison_voices):
    #         # Calcular la frecuencia de esta voz (esto no cambia)
    #         cents = detune_amounts[i]
    #         freq_ratio = 2**(cents / 1200.0)
    #         voice_freq = self.freq * freq_ratio
            
    #         # Generar la onda para esta voz (esto no cambia)
    #         mono_voice = self.create_oscillator(voice_freq)
            
    #         # 2. Añadir la voz directamente al buffer final (sin panning)
    #         final_wave += mono_voice

    #     # Normalizar la salida para evitar distorsión (esto no cambia)
    #     final_wave /= np.sqrt(self.unison_voices)
        
    #     return final_wave * self.volume
    
    def process(self):
        # self.compute_delta_semitones()
        
        # if self.unison_voices > 1:
        #     return self.create_unison_sound_mono()

        return self.create_oscillator(self.freq)
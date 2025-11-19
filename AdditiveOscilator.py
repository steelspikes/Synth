import numpy as np
import sounddevice as sd

class AdditiveOscillator:
    def __init__(self, sample_rate=44100, base_freq=440, duration=0, partials=[], volume=1, lfo1_instance=None, lfo2_instance=None,pitch_mod_depth=0,volume_mod_depth=0,lfo_choose=None):
        self.sample_rate = sample_rate
        self.base_freq = base_freq
        self.duration = duration
        self.partials = partials
        self.volume = volume
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

    def process(self):
        """
        Genera una onda sumando parciales (intervalos).
        
        Args:
            base_freq (float): La frecuencia fundamental de la nota en Hz.
            duration (float): La duración del sonido en segundos.
            partials (list): Una lista de tuplas (ratio, amplitud).
                             Ej: [(1.0, 1.0), (2.0, 0.5), (3.0, 0.3)]
            volume (float): El volumen final de la onda.
        """
        num_samples = int(self.sample_rate * self.duration)
        final_wave = np.zeros(num_samples, dtype=np.float64)
        t = np.linspace(0., self.duration, num_samples, endpoint=False)

        print(self.partials)
        
        # Iterar sobre la "receta" de parciales
        for ratio, amp in self.partials:
            if amp > 0:
                # Calcular la frecuencia absoluta para este parcial
                partial_freq = self.base_freq * ratio

                partial_freq = partial_freq * self.modulate(self.pitch_mod_depth)
                dt = t[1] - t[0]
                
                # Generar y sumar la onda sinusoidal correspondiente
                # phase = 2 * np.pi * partial_freq * t
                phase = 2 * np.pi * np.cumsum(partial_freq) * dt

                final_wave += amp * np.sin(phase)
                
        # Normalizar para evitar clipping
        max_abs = np.max(np.abs(final_wave))
        if max_abs > 0:
            final_wave /= max_abs

        final_wave = final_wave * self.modulate(self.volume_mod_depth)
            
        return final_wave * self.volume
    
# ao = AdditiveOscillator()
# sd.play(ao.process(440, 2, [(1, 1),(2,1),(4,1),(1,1)], 1))
# sd.wait()
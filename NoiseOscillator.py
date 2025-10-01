import numpy as np
from Utils import denormalize

class NoiseOscillator:
    def __init__(self, sample_rate=44100, duration = 0, volume = 0, shape=0):
        self.sample_rate = sample_rate
        self.shape = denormalize(shape, 0, 2)
        self.volume = volume
        self.duration = duration

    def _generate_pink_noise(self, num_samples):
        """Generador de ruido rosa (método interno)."""
        white_noise = np.random.randn(num_samples)
        X = np.fft.rfft(white_noise)
        S = np.sqrt(np.arange(len(X)) + 1.)
        pink_noise = np.fft.irfft(X / S).real
        return pink_noise

    def process(self):
        """
        Genera un bloque de ruido con morphing.
        'color' es un float de 0.0 (Blanco) a 2.0 (Marrón).
        """
        num_samples = int(self.sample_rate * self.duration)
        
        # --- Lógica de Morphing ---
        if self.shape <= 1.0:
            # Transición de Blanco (0.0) a Rosa (1.0)
            blend = self.shape
            
            # Solo generamos los dos ruidos que necesitamos
            white_noise = np.random.randn(num_samples)
            pink_noise = self._generate_pink_noise(num_samples)

            # Hacemos el crossfade
            waveform = (1.0 - blend) * white_noise + blend * pink_noise
            
        else: # color > 1.0
            # Transición de Rosa (1.0) a Marrón (2.0)
            blend = self.shape - 1.0 # 'blend' ahora va de 0.0 a 1.0
            
            # Solo generamos los dos ruidos que necesitamos
            pink_noise = self._generate_pink_noise(num_samples)
            white_for_brown = np.random.randn(num_samples) # Ruido base para el marrón
            brown_noise = np.cumsum(white_for_brown)

            # Hacemos el crossfade
            waveform = (1.0 - blend) * pink_noise + blend * brown_noise
        
        # Normalizar el resultado final para que esté en el rango [-1, 1]
        max_abs = np.max(np.abs(waveform))
        if max_abs > 0:
            waveform /= max_abs
            
        return waveform * self.volume
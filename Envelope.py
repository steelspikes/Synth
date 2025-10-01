import numpy as np

class Envelope:
    def __init__(self, sample_rate=44100, attack=0.01, decay=0.3, sustain=0.5, release=0.2):
        self.sample_rate = sample_rate
        self.attack = attack
        self.decay = decay
        self.sustain = sustain  # Cambié el valor por defecto a 0.5 para que sea más realista
        self.release = release

    def process(self, total_duration_sec):
        """Genera el array de la envolvente ADSR para una duración fija."""
        # 1. Convertir tiempos a muestras
        total_samples = int(total_duration_sec * self.sample_rate)
        attack_samples = int(self.attack * self.sample_rate)
        decay_samples = int(self.decay * self.sample_rate)
        release_samples = int(self.release * self.sample_rate)

        # Calcular el tiempo disponible para sustain
        ads_samples = attack_samples + decay_samples + release_samples
        
        # Si la duración total es muy corta, ajustar las fases
        if ads_samples >= total_samples:
            # Distribuir proporcionalmente entre attack, decay y release
            total_ads_time = self.attack + self.decay + self.release
            attack_samples = int((self.attack / total_ads_time) * total_samples)
            decay_samples = int((self.decay / total_ads_time) * total_samples)
            release_samples = total_samples - (attack_samples + decay_samples)
            sustain_samples = 0
        else:
            sustain_samples = total_samples - ads_samples

        # 2. Construir las fases de la envolvente
        attack = np.linspace(0, 1.0, attack_samples)
        decay = np.linspace(1.0, self.sustain, decay_samples)
        sustain = np.full(sustain_samples, self.sustain)
        # release = np.linspace(self.sustain, 0, release_samples)

        if release_samples > 0:
            n = np.arange(release_samples)
            tau = release_samples / 5  # Ajusta la velocidad del decaimiento exponencial
            release = self.sustain * np.exp(-n / tau)
        else:
            release = np.array([])

        # 3. Concatenar todas las partes
        envelope = np.concatenate((attack, decay, sustain, release))

        # 4. Asegurar la longitud exacta (por si hay errores de redondeo)
        return envelope[:total_samples]
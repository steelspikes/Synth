import numpy as np

class Envelope:
    def __init__(self, sample_rate=44100, attack=0.01, decay=0.3, sustain=0.5, release=0.2):
        self.sample_rate = sample_rate
        self.attack = attack.astype(np.float32)
        self.decay = decay.astype(np.float32)
        self.sustain = sustain.astype(np.float32)
        self.release = release.astype(np.float32)

    def process(self, total_duration_sec, n_samples):
        """Genera la curva ADSR vectorizada para un batch de presets.
        
        Devuelve un array (presets x n_samples) con valores en [0, 1].
        """
        presets = self.attack.shape[0]
        # Eje de tiempo normalizado [0, 1] compartido por todos los presets
        x = np.linspace(0, 1.0, n_samples, dtype=np.float32)
        x = np.expand_dims(x, axis=0)
        x = np.broadcast_to(x, (presets, n_samples))

        # Convertimos tiempos absolutos a fracción de la duración total
        rel_attack = self.attack / total_duration_sec
        rel_decay = self.decay / total_duration_sec
        rel_release = self.release / total_duration_sec
        sus_level = self.sustain

        rel_attack = np.expand_dims(rel_attack, axis=1).astype(np.float32)
        rel_decay = np.expand_dims(rel_decay, axis=1).astype(np.float32)
        rel_release = np.expand_dims(rel_release, axis=1).astype(np.float32)
        sus_level = np.expand_dims(sus_level, axis=1).astype(np.float32)

        # Sustain ocupa lo que reste después de A, D y R
        rel_sustain = 1.0 - (rel_attack + rel_decay + rel_release)
        rel_note_off = rel_attack + rel_decay + rel_sustain

        eps = 1e-6

        # ATTACK: sube de 0 a 1 linealmente
        attack_curve = x / (rel_attack + eps)
        attack_curve = np.clip(attack_curve, max=1.0)

        # DECAY: baja de 1 al nivel de sustain
        decay_curve = (x - rel_attack) * (sus_level - 1.0) / (rel_decay + eps)
        decay_curve = np.clip(decay_curve, min=(sus_level - 1.0), max=0.0)

        # RELEASE: baja desde el sustain hasta 0
        release_curve = (x - rel_note_off) * (-sus_level / (rel_release + eps))
        release_curve = np.clip(release_curve, min=-sus_level, max=0.0)

        # SUMA de las tres rampas da la envolvente completa
        envelope = attack_curve + decay_curve + release_curve
        envelope = np.clip(envelope, min=0.0, max=1.0)

        return envelope

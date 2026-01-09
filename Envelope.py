import numpy as np

class Envelope:
    def __init__(self, sample_rate=44100, attack=0.01, decay=0.3, sustain=0.5, release=0.2):
        self.sample_rate = sample_rate
        self.attack = attack
        self.decay = decay
        self.sustain = sustain  # Cambié el valor por defecto a 0.5 para que sea más realista
        self.release = release

    def process(self, total_duration_sec):
        n_samples = int(self.sample_rate * total_duration_sec)
        x = np.linspace(0, 1.0, n_samples, dtype=np.float32)

        rel_attack = self.attack / total_duration_sec
        rel_decay = self.decay / total_duration_sec
        rel_release = self.release / total_duration_sec
        sus_level = self.sustain

        # Duración relativa de sustain para llenar lo que queda
        rel_sustain = 1.0 - (rel_attack + rel_decay + rel_release)
        rel_note_off = rel_attack + rel_decay + rel_sustain

        eps = 1e-6

        # ATTACK
        attack_curve = x / (rel_attack + eps)
        attack_curve = np.clip(attack_curve, max=1.0)

        # DECAY
        decay_curve = (x - rel_attack) * (sus_level - 1.0) / (rel_decay + eps)
        decay_curve = np.clip(decay_curve, min=(sus_level - 1.0), max=0.0)

        # RELEASE
        release_curve = (x - rel_note_off) * (-sus_level / (rel_release + eps))
        release_curve = np.clip(release_curve, min=-sus_level, max=0.0)

        # SUMA
        envelope = attack_curve + decay_curve + release_curve
        envelope = np.clip(envelope, min=0.0, max=1.0)

        return envelope

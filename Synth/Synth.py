from BiquadFilter import BiquadFilter
from NoiseOscillator import NoiseOscillator
from Envelope import Envelope
from Oscillator import Oscillator

class Synth:
    def __init__(self, sample_rate=44100, duration=0, presets=None):
        self.sample_rate = sample_rate
        self.presets = presets
        self.duration = duration

    def process_audio(self):
        """Instancia todos los módulos con los presets recibidos y produce el audio final."""
        osc1 = Oscillator(
            shape=self.presets['osc1_shape'], 
            phase=self.presets['osc1_phase'], 
            volume=self.presets['osc1_volume'], 
            initial_freq=self.presets['osc1_freq'], 
            sample_rate=self.sample_rate, 
            duration=self.duration
        )

        osc2 = Oscillator(
            shape=self.presets['osc2_shape'], 
            phase=self.presets['osc2_phase'], 
            volume=self.presets['osc2_volume'], 
            initial_freq=self.presets['osc2_freq'],
            sample_rate=self.sample_rate, 
            duration=self.duration
        )

        osc3 = Oscillator(
            shape=self.presets['osc3_shape'], 
            phase=self.presets['osc3_phase'], 
            volume=self.presets['osc3_volume'], 
            initial_freq=self.presets['osc3_freq'],
            sample_rate=self.sample_rate, 
            duration=self.duration
        )

        osc4 = Oscillator(
            shape=self.presets['osc4_shape'], 
            phase=self.presets['osc4_phase'], 
            volume=self.presets['osc4_volume'], 
            initial_freq=self.presets['osc4_freq'], 
            sample_rate=self.sample_rate, 
            duration=self.duration,
        )

        oscnoise = NoiseOscillator(
            volume=self.presets['oscnoise_volume'],
            sample_rate=self.sample_rate, 
            duration=self.duration
        )

        # Envelope que modula el cutoff del filtro
        envelope_filter = Envelope(
            attack=self.presets['filter_envelope_attack'], 
            decay=self.presets['filter_envelope_decay'],
            sustain=self.presets['filter_envelope_sustain'], 
            release=self.presets['filter_envelope_release'],
            sample_rate=self.sample_rate
        )

        biquad_filter = BiquadFilter(
            base_cutoff_hz=self.presets['base_cutoff_hz'], 
            filter_type=self.presets['filter_type'], 
            base_q=self.presets['base_q'],
            envelope=envelope_filter,
            envelope_depth=self.presets['envelope_depth'],
            sample_rate=self.sample_rate
        )

        # Envelope de amplitud final
        envelope_amp = Envelope(
            attack=self.presets['envelope_attack'], 
            decay=self.presets['envelope_decay'],
            sustain=self.presets['envelope_sustain'], 
            release=self.presets['envelope_release'],
            sample_rate=self.sample_rate
        )

        # Suma de osciladores → filtro → envelope de amplitud
        out = osc1.process() + osc2.process() + osc3.process() + osc4.process() + oscnoise.process()
        out = biquad_filter.process(out)

        envelope_amp_signal = envelope_amp.process(self.duration, out.shape[1])
        out = out * envelope_amp_signal

        return out
from LFO import LFO
from BiquadFilter import BiquadFilter
from NoiseOscillator import NoiseOscillator
from Envelope import Envelope
from Oscillator import Oscillator
import numpy as np
from Utils import denormalize

class Synth:
    def __init__(self, sample_rate=44100, duration=0, presets=None):
        self.sample_rate = sample_rate
        self.presets = presets
        self.duration = duration

    def process_audio(self):
        # lfo1 = LFO(rate_hz=np.zeros_like(self.presets['lfo1_rate']), shape=self.presets['lfo1_shape'])
        # lfo2 = LFO(rate_hz=np.zeros_like(self.presets['lfo2_rate']), shape=self.presets['lfo2_shape'])

        # lfo_signal = lfo1.process(int(self.sample_rate * self.duration))

        osc1 = Oscillator(
            shape=self.presets['osc1_shape'], 
            phase=np.zeros_like(self.presets['osc1_phase']), 
            volume=self.presets['osc1_volume'], 
            initial_freq=self.presets['osc1_freq'], 
            # lfo_signal=lfo_signal,
            sample_rate=self.sample_rate, 
            duration=self.duration,
            # volume_mod_depth=self.presets['osc1_vdepth'],
            # pitch_mod_depth=self.presets['osc1_pdepth']
        )

        osc2 = Oscillator(
            shape=self.presets['osc2_shape'], 
            phase=np.zeros_like(self.presets['osc2_phase']), 
            volume=self.presets['osc2_volume'], 
            initial_freq=self.presets['osc2_freq'], 
            # lfo_signal=lfo_signal,
            sample_rate=self.sample_rate, 
            duration=self.duration,
            # volume_mod_depth=self.presets['osc2_vdepth'],
            # pitch_mod_depth=self.presets['osc2_pdepth']
        )

        osc3 = Oscillator(
            shape=self.presets['osc3_shape'], 
            phase=np.zeros_like(self.presets['osc3_phase']), 
            volume=self.presets['osc3_volume'], 
            initial_freq=self.presets['osc3_freq'], 
            # lfo_signal=lfo_signal,
            sample_rate=self.sample_rate, 
            duration=self.duration,
            # volume_mod_depth=self.presets['osc3_vdepth'],
            # pitch_mod_depth=self.presets['osc3_pdepth']
        )

        osc4 = Oscillator(
            shape=self.presets['osc4_shape'], 
            phase=np.zeros_like(self.presets['osc4_phase']), 
            volume=self.presets['osc4_volume'], 
            initial_freq=self.presets['osc4_freq'], 
            # lfo_signal=lfo_signal,
            sample_rate=self.sample_rate, 
            duration=self.duration,
            # volume_mod_depth=self.presets['osc4_vdepth'],
            # pitch_mod_depth=self.presets['osc4_pdepth']
        )

        oscnoise = NoiseOscillator(
            volume=self.presets['oscnoise_volume'],
            sample_rate=self.sample_rate, 
            duration=self.duration
        )

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
            # lfo_instance=lfo2,
            envelope=envelope_filter,
            envelope_depth=self.presets['envelope_depth'],
            # cutoff_mod_depth=self.presets['cutoff_mod_depth'],
            sample_rate=self.sample_rate
        )

        envelope_amp = Envelope(
            attack=self.presets['envelope_attack'], 
            decay=self.presets['envelope_decay'],
            sustain=self.presets['envelope_sustain'], 
            release=self.presets['envelope_release'],
            sample_rate=self.sample_rate
        )

        out = osc1.process() + osc2.process() + osc3.process() + osc4.process() + oscnoise.process()
        out = biquad_filter.process(out)

        envelope_amp_signal = envelope_amp.process(self.duration)
        out = out * envelope_amp_signal

        # peak = np.max(np.abs(out))
        # if peak > 0:
        #     out = out / peak

        return out
from LFO import LFO
from BiquadFilter import BiquadFilter
from NoiseOscillator import NoiseOscillator
from Envelope import Envelope
from Oscillator import Oscillator
import numpy as np
from Utils import denormalize

class Synth:
    def __init__(self, sample_rate=44100, duration=0, presets=None, is_normalized=False):
        self.sample_rate = sample_rate
        self.presets = presets
        self.duration = duration
        self.is_normalized = is_normalized

    def process_audio(self):
        MIN_FREQ = 1
        MAX_FREQ = self.sample_rate / 2

        lfo1 = LFO(rate_hz=denormalize(self.is_normalized,self.presets['lfo1_rate'], 0, 20), shape=denormalize(self.is_normalized,self.presets['lfo1_shape'],0,4))
        lfo2 = LFO(rate_hz=denormalize(self.is_normalized,self.presets['lfo2_rate'], 0, 20), shape=denormalize(self.is_normalized,self.presets['lfo2_shape'],0,4))

        lfo_signal = lfo1.process(int(self.sample_rate * self.duration))

        osc1 = Oscillator(
            shape=denormalize(self.is_normalized,self.presets['osc1_shape'], 0, 4), 
            phase=denormalize(self.is_normalized,self.presets['osc1_phase'], 0, 1), 
            volume=denormalize(self.is_normalized,self.presets['osc1_volume'], 0, 1), 
            initial_freq=denormalize(self.is_normalized,self.presets['osc1_freq'], MIN_FREQ, MAX_FREQ), 
            lfo_signal=lfo_signal,
            sample_rate=self.sample_rate, 
            duration=self.duration,
            volume_mod_depth=denormalize(self.is_normalized,self.presets['osc1_vdepth'], 0, 1),
            pitch_mod_depth=denormalize(self.is_normalized,self.presets['osc1_pdepth'], 0, 1)
        )

        osc2 = Oscillator(
            shape=denormalize(self.is_normalized,self.presets['osc2_shape'], 0, 4), 
            phase=denormalize(self.is_normalized,self.presets['osc2_phase'], 0, 1), 
            volume=denormalize(self.is_normalized,self.presets['osc2_volume'], 0, 1), 
            initial_freq=denormalize(self.is_normalized,self.presets['osc2_freq'], MIN_FREQ, MAX_FREQ), 
            lfo_signal=lfo_signal,
            sample_rate=self.sample_rate, 
            duration=self.duration,
            volume_mod_depth=denormalize(self.is_normalized,self.presets['osc2_vdepth'], 0, 1),
            pitch_mod_depth=denormalize(self.is_normalized,self.presets['osc2_pdepth'], 0, 1)
        )

        osc3 = Oscillator(
            shape=denormalize(self.is_normalized,self.presets['osc3_shape'], 0, 4), 
            phase=denormalize(self.is_normalized,self.presets['osc3_phase'], 0, 1), 
            volume=denormalize(self.is_normalized,self.presets['osc3_volume'], 0, 1), 
            initial_freq=denormalize(self.is_normalized,self.presets['osc3_freq'], MIN_FREQ, MAX_FREQ), 
            lfo_signal=lfo_signal,
            sample_rate=self.sample_rate, 
            duration=self.duration,
            volume_mod_depth=denormalize(self.is_normalized,self.presets['osc3_vdepth'], 0, 1),
            pitch_mod_depth=denormalize(self.is_normalized,self.presets['osc3_pdepth'], 0, 1)
        )

        osc4 = Oscillator(
            shape=denormalize(self.is_normalized,self.presets['osc4_shape'], 0, 4), 
            phase=denormalize(self.is_normalized,self.presets['osc4_phase'], 0, 1), 
            volume=denormalize(self.is_normalized,self.presets['osc4_volume'], 0, 1), 
            initial_freq=denormalize(self.is_normalized,self.presets['osc4_freq'], MIN_FREQ, MAX_FREQ), 
            lfo_signal=lfo_signal,
            sample_rate=self.sample_rate, 
            duration=self.duration,
            volume_mod_depth=denormalize(self.is_normalized,self.presets['osc4_vdepth'], 0, 1),
            pitch_mod_depth=denormalize(self.is_normalized,self.presets['osc4_pdepth'], 0, 1)
        )

        oscnoise = NoiseOscillator(
            volume=denormalize(self.is_normalized,self.presets['oscnoise_volume'], 0, 1),
            sample_rate=self.sample_rate, 
            duration=self.duration
        )

        envelope_filter = Envelope(
            attack=denormalize(self.is_normalized,self.presets['filter_envelope_attack'], 0, self.duration), 
            decay=denormalize(self.is_normalized,self.presets['filter_envelope_decay'], 0, self.duration),
            sustain=denormalize(self.is_normalized,self.presets['filter_envelope_sustain'], 0, 1), 
            release=denormalize(self.is_normalized,self.presets['filter_envelope_release'], 0, self.duration),
            sample_rate=self.sample_rate
        )

        biquad_filter = BiquadFilter(
            base_cutoff_hz=denormalize(self.is_normalized,self.presets['base_cutoff_hz'], MIN_FREQ, MAX_FREQ), 
            filter_type=denormalize(self.is_normalized,self.presets['filter_type'], 0, 2), 
            base_q=denormalize(self.is_normalized,self.presets['base_q'], 0.707, 20),
            lfo_instance=lfo2,
            envelope=envelope_filter,
            envelope_depth=denormalize(self.is_normalized,self.presets['envelope_depth'], 0, 1),
            cutoff_mod_depth=denormalize(self.is_normalized,self.presets['cutoff_mod_depth'], 0, 1),
            sample_rate=self.sample_rate
        )

        envelope_amp = Envelope(
            attack=denormalize(self.is_normalized,self.presets['envelope_attack'], 0, self.duration), 
            decay=denormalize(self.is_normalized,self.presets['envelope_decay'], 0, self.duration),
            sustain=denormalize(self.is_normalized,self.presets['envelope_sustain'], 0, 1), 
            release=denormalize(self.is_normalized,self.presets['envelope_release'], 0, self.duration),
            sample_rate=self.sample_rate
        )

        print('1shape', denormalize(self.is_normalized,self.presets['osc1_shape'], 0, 4))
        print('1vol', denormalize(self.is_normalized,self.presets['osc1_volume'], 0, 1))
        print('1freq', denormalize(self.is_normalized,self.presets['osc1_freq'], MIN_FREQ, MAX_FREQ))
        # print(denormalize(self.is_normalized,self.presets['osc1_vdepth'], 0, 1))
        # print(denormalize(self.is_normalized,self.presets['osc1_pdepth'], 0, 1))
        print('1phase', denormalize(self.is_normalized,self.presets['osc1_phase'], 0, 1))

        print('2shape', denormalize(self.is_normalized,self.presets['osc2_shape'], 0, 4))
        print('2vol', denormalize(self.is_normalized,self.presets['osc2_volume'], 0, 1))
        print('2freq',denormalize(self.is_normalized,self.presets['osc2_freq'], MIN_FREQ, MAX_FREQ))
        # print(denormalize(self.is_normalized,self.presets['osc2_vdepth'], 0, 1))
        # print(denormalize(self.is_normalized,self.presets['osc2_pdepth'], 0, 1))
        print('2phase', denormalize(self.is_normalized,self.presets['osc2_phase'], 0, 1))

        print('3shape', denormalize(self.is_normalized,self.presets['osc3_shape'], 0, 4))
        print('3vol', denormalize(self.is_normalized,self.presets['osc3_volume'], 0, 1))
        print('3freq', denormalize(self.is_normalized,self.presets['osc3_freq'], MIN_FREQ, MAX_FREQ))
        # print(denormalize(self.is_normalized,self.presets['osc3_vdepth'], 0, 1))
        # print(denormalize(self.is_normalized,self.presets['osc3_pdepth'], 0, 1))
        print('3phase', denormalize(self.is_normalized,self.presets['osc3_phase'], 0, 1))

        print('4shape', denormalize(self.is_normalized,self.presets['osc4_shape'], 0, 4))
        print('4vol', denormalize(self.is_normalized,self.presets['osc4_volume'], 0, 1))
        print('4freq', denormalize(self.is_normalized,self.presets['osc4_freq'], MIN_FREQ, MAX_FREQ))
        # print(denormalize(self.is_normalized,self.presets['osc4_vdepth'], 0, 1))
        # print(denormalize(self.is_normalized,self.presets['osc4_pdepth'], 0, 1))
        print('4phase', denormalize(self.is_normalized,self.presets['osc4_phase'], 0, 1))

        print('noise', denormalize(self.is_normalized,self.presets['oscnoise_volume'], 0, 1))

        print('Cutoff', denormalize(self.is_normalized,self.presets['base_cutoff_hz'], MIN_FREQ, MAX_FREQ))
        print('Filter type', denormalize(self.is_normalized,self.presets['filter_type'], 0, 2))
        print('Q', denormalize(self.is_normalized,self.presets['base_q'], 0.707, 20))
        # print(denormalize(self.is_normalized,self.presets['envelope_depth'], 0, 1))
        # print(denormalize(self.is_normalized,self.presets['cutoff_mod_depth'], 0, 1))

        # print(denormalize(self.is_normalized,self.presets['envelope_attack'], 0, self.duration))
        # print(denormalize(self.is_normalized,self.presets['envelope_decay'], 0, self.duration))
        # print(denormalize(self.is_normalized,self.presets['envelope_sustain'], 0, 1))
        # print(denormalize(self.is_normalized,self.presets['envelope_release'], 0, self.duration))

        # print(denormalize(self.is_normalized,self.presets['envelope_attack'], 0, self.duration))
        # print(denormalize(self.is_normalized,self.presets['envelope_decay'], 0, self.duration))
        # print(denormalize(self.is_normalized,self.presets['envelope_sustain'], 0, 1))
        # print(denormalize(self.is_normalized,self.presets['envelope_release'], 0, self.duration))

        out = osc1.process() + osc2.process() + osc3.process() + osc4.process() + oscnoise.process()
        out = biquad_filter.process(out)

        envelope_amp_signal = envelope_amp.process(self.duration)
        out = out * envelope_amp_signal
        

        peak = np.max(np.abs(out))
        if peak > 0:
            out = out / peak

        return out
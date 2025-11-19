import sounddevice as sd
import matplotlib.pyplot as plt
from Oscillator import Oscillator
from BiquadFilter import BiquadFilter
from LFO import LFO
from Envelope import Envelope
import numpy as np
from scipy.io import wavfile
import json
from NoiseOscillator import NoiseOscillator
from AdditiveOscilator import AdditiveOscillator
import sys
import os
import shutil

def delete_exports():
    folder = "exports"

    for elemento in os.listdir(folder):
        route = os.path.join(folder, elemento)
        if os.path.isfile(route) or os.path.islink(route):
            os.unlink(route)
        else:
            shutil.rmtree(route)

file_name = sys.argv[1]

delete_exports()

with open(f'presets/{file_name}', 'r') as archivo:
    arrParameters = json.load(archivo)

i = 0

for parameters in arrParameters:
    # Catch Parameters ################################
    sample_rate = parameters['general']['sample_rate']
    duration = parameters['general']['duration']

    def get_osc_param(index, param):
        return parameters['osc'+str(index)][param]

    amp_env_params = parameters['amplitude_envelope']
    filter_env_params = parameters['filter_envelope']
    filter_params = parameters['filter']
    lfo1_params = parameters['lfo1']
    lfo2_params = parameters['lfo2']

    osc_noise = parameters['osc_noise']

    add_osc_params = parameters['add_osc']

    #################################################33

    lfo1 = LFO(rate_hz=lfo1_params.get('rate'), shape=lfo1_params.get('shape'))
    lfo2 = LFO(rate_hz=lfo2_params.get('rate'), shape=lfo2_params.get('shape'))

    addosc = AdditiveOscillator( # Agregar LFO y Envelopes
        sample_rate=sample_rate,
        duration=duration,
        base_freq=add_osc_params.get('base_frequency'),
        volume=add_osc_params.get('volume'),
        partials=add_osc_params.get('partials'),
        lfo1_instance=lfo1,
        lfo2_instance=lfo2,
        volume_mod_depth=add_osc_params.get('volume_mod_depth'),
        pitch_mod_depth=add_osc_params.get('pitch_mod_depth'),
        lfo_choose=add_osc_params.get('lfo_choose')
    )
    out_addosc = addosc.process()

    osc1 = Oscillator(
        shape=get_osc_param(1,'shape'), 
        phase=get_osc_param(1,'phase'), 
        volume=get_osc_param(1,'volume'), 
        initial_freq=get_osc_param(1,'frequency'), 
        lfo1_instance=lfo1,
        lfo2_instance=lfo2,
        sample_rate=sample_rate, 
        duration=duration,
        volume_mod_depth=get_osc_param(1,'volume_mod_depth'),
        pitch_mod_depth=get_osc_param(1,'pitch_mod_depth'),
        lfo_choose=get_osc_param(1,'lfo_choose')
    )
    out_osc1 = osc1.process()

    osc2 = Oscillator(
        shape=get_osc_param(2,'shape'), 
        phase=get_osc_param(2,'phase'), 
        volume=get_osc_param(2,'volume'), 
        initial_freq=get_osc_param(2,'frequency'), 
        lfo1_instance=lfo1,
        lfo2_instance=lfo2, 
        sample_rate=sample_rate, 
        duration=duration,
        volume_mod_depth=get_osc_param(2,'volume_mod_depth'),
        pitch_mod_depth=get_osc_param(2,'pitch_mod_depth'),
        lfo_choose=get_osc_param(2,'lfo_choose')
    )
    out_osc2 = osc2.process()

    osc3 = Oscillator(
        shape=get_osc_param(3,'shape'), 
        phase=get_osc_param(3,'phase'), 
        volume=get_osc_param(3,'volume'), 
        initial_freq=get_osc_param(3,'frequency'), 
        lfo1_instance=lfo1,
        lfo2_instance=lfo2, 
        sample_rate=sample_rate, 
        duration=duration,
        volume_mod_depth=get_osc_param(3,'volume_mod_depth'),
        pitch_mod_depth=get_osc_param(3,'pitch_mod_depth'),
        lfo_choose=get_osc_param(3,'lfo_choose')
    )
    out_osc3 = osc3.process()

    osc4 = Oscillator(
        shape=get_osc_param(4,'shape'), 
        phase=get_osc_param(4,'phase'), 
        volume=get_osc_param(4,'volume'), 
        initial_freq=get_osc_param(4,'frequency'), 
        lfo1_instance=lfo1,
        lfo2_instance=lfo2, 
        sample_rate=sample_rate, 
        duration=duration,
        volume_mod_depth=get_osc_param(4,'volume_mod_depth'),
        pitch_mod_depth=get_osc_param(4,'pitch_mod_depth'),
        lfo_choose=get_osc_param(4,'lfo_choose')
    )
    out_osc4 = osc4.process()

    oscnoise = NoiseOscillator(
        shape=osc_noise.get('shape'), 
        volume=osc_noise.get('volume'),
        sample_rate=sample_rate, 
        duration=duration
    )
    out_noise = oscnoise.process()

    out_osc = out_addosc + out_osc1 + out_osc2 + out_osc3 + out_osc4 + out_noise

    envelope = Envelope(attack=filter_env_params.get('attack'), decay=filter_env_params.get('decay'), sustain=filter_env_params.get('sustain'), release=filter_env_params.get('release'))
    envelope_signal = envelope.process(duration)

    filter1 = BiquadFilter(
        base_cutoff_hz=filter_params.get('cutoff'), 
        filter_type=filter_params.get('type'), 
        base_q=filter_params.get('q'),
        lfo1_instance=lfo1,
        lfo2_instance=lfo2,
        lfo_choose=filter_params.get('lfo_choose'),
        envelope=envelope,
        envelope_depth=filter_params.get('envelope_depth'),
        cutoff_mod_depth=filter_params.get('cutoff_mod_depth')
    )
    out = filter1.process(out_osc)

    envelope_amp = Envelope(attack=amp_env_params.get('attack'), decay=amp_env_params.get('decay'), sustain=amp_env_params.get('sustain'), release=amp_env_params.get('release'))
    envelope_amp_signal = envelope_amp.process(duration)

    out = out * envelope_amp_signal

    wavfile.write(f"exports/sound{i}.wav", sample_rate, np.int16(out * 32767))

    i+=1
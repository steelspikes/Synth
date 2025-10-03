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

for pp in ['parameters.json']:
    with open(pp, 'r') as archivo:
        parameters = json.load(archivo)

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

    #################################################33

    lfo1 = LFO(rate_hz=lfo1_params.get('rate'), shape=lfo1_params.get('shape'))
    lfo2 = LFO(rate_hz=lfo2_params.get('rate'), shape=lfo2_params.get('shape'))

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

    out_osc = out_osc1 + out_osc2 + out_osc3 + out_osc4 + out_noise

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
        cutoff_mod_depth=filter_params.get('cutoff_mod_depth')
    )
    out = filter1.process(out_osc)

    envelope_amp = Envelope(attack=amp_env_params.get('attack'), decay=amp_env_params.get('decay'), sustain=amp_env_params.get('sustain'), release=amp_env_params.get('release'))
    envelope_amp_signal = envelope_amp.process(duration)

    out = out * envelope_amp_signal

    # print(len(out) / 44100)

    # reverb = Reverb()
    # out = reverb.process(out)

    # out = out / np.max(np.abs(out))

    # wavfile.write("audio_output.wav", 44100, np.int16(out * 32767))

    # plt.plot(out)
    # plt.show()
    # plt.pause(0.01)
    # plt.cla()

    #################################
    # N = len(out)

    # # Transformada de Fourier
    # fft_data = np.fft.fft(out)
    # fft_freq = np.fft.fftfreq(N, d=1/sample_rate)

    # # Nos quedamos con la mitad positiva
    # mask = fft_freq >= 0
    # fft_data = np.abs(fft_data[mask])
    # fft_freq = fft_freq[mask]

    # # Graficar
    # # plt.figure(figsize=(10, 6))
    # plt.plot(fft_freq, fft_data)
    # # plt.title("Espectro de Fourier del audio")
    # # plt.xlabel("Frecuencia (Hz)")
    # # plt.ylabel("Magnitud")
    # plt.xlim(0, sample_rate/2)  # hasta Nyquist
    # plt.pause(0.04)
    # plt.cla()
    #############################################




    ##############################################

    # Graficar espectrograma
    # plt.figure(figsize=(10,6))
    # plt.specgram(out, NFFT=1024, Fs=sample_rate, noverlap=512, cmap='viridis')
    # plt.xlabel("Tiempo (s)")
    # plt.ylabel("Frecuencia (Hz)")
    # plt.title("Espectrograma")
    # plt.colorbar(label="Magnitud (dB)")
    # plt.show()

    ##############################################

    sd.play(out, sample_rate)
    sd.wait()
import numpy as np
import matplotlib.pyplot as plt
import json

def create_morphed_wave(morph_param, base_phase): 
    num_waves = 5
    centers = np.linspace(0, num_waves - 1, num_waves, dtype=np.float32)

    dists = np.abs(morph_param - centers)

    weights = np.maximum(0, 1.0 - dists)

    morphed_wave = np.zeros_like(base_phase)
    
    w_sine = weights[0]
    
    if np.any(w_sine > 0):
        morphed_wave += w_sine * get_sine_wave(base_phase)

    w_tri = weights[1]
    w_saw = weights[2]

    if np.any(w_tri > 0) or np.any(w_saw > 0):
        saw_raw = get_sawtooth_wave(base_phase)
        tri_raw = get_triangle_wave(base_phase)

        if np.any(w_saw > 0):
            morphed_wave += w_saw * saw_raw
            
        if np.any(w_tri > 0):
            morphed_wave += w_tri * tri_raw   

    w_sq = weights[3]
    if np.any(w_sq > 0):
        morphed_wave += w_sq * get_square_wave(base_phase, duty=0.5)

    sigmoid = lambda x: 1 / (1 + np.exp(-x))
    k = 10
    alpha = sigmoid(k * (morph_param - 3))
    
    if np.any(alpha > 0.001):
        MIDDLE = 0.5
        w_last = weights[:, 4] 
        duty = MIDDLE - (w_last * MIDDLE)
        
        # Generar pulse variable
        pulse_wave = get_square_wave(base_phase, duty=duty)
        
        morphed_wave = alpha * pulse_wave + (1 - alpha) * morphed_wave
    
    return morphed_wave


def get_sine_wave(x):
    return np.sin(x)

def get_triangle_wave(x):
    saw = get_sawtooth_wave(x) 
    return 2 * np.abs(saw) - 1

def get_sawtooth_wave(x):
    normalized = x / (2 * np.pi)
    return 2 * (normalized - np.floor(normalized + 0.5)) # No diferenciable por floor, pero el gradiente fluye por x (normalized)

def get_square_wave(x, duty=0.5):
    duty = 2.0 * duty - 1.0
    return np.tanh(5.0 * (np.sin(x) - duty))

def denormalize(n, v_min, v_max):
    return v_min + n * (v_max - v_min)

def get_freq_log(freq):
    return np.log(freq)

def plot_spectrum_linear(waveform, sample_rate, title="Espectro de Frecuencia"):
    # 1. Convertir a numpy plano
    
    if waveform.ndim > 1:
        waveform = waveform.flatten()

    # 2. FFT (solo parte positiva)
    N = len(waveform)
    fft_data = np.fft.rfft(waveform)
    
    # 3. Magnitud lineal
    magnitude = np.abs(fft_data)
    
    # 4. Eje de frecuencias
    freqs = np.fft.rfftfreq(N, d=1/sample_rate)

    # 5. Graficar
    plt.figure(figsize=(10, 6))
    plt.plot(freqs, magnitude)
    plt.title(title)
    plt.xlabel('Frecuencia (Hz)')
    plt.ylabel('Amplitud')
    plt.grid(True, ls='-', alpha=0.5)
    plt.xlim(0, sample_rate / 2)
    plt.show()

def load_parameters_file():
    with open('presets/parameters.json', 'r', encoding='utf-8') as file:
        parameters_file = json.load(file)

    osc1_shape = []
    osc1_phase = []
    osc1_volume = []
    osc1_freq = []
    osc1_vdepth = []
    osc1_pdepth = []

    osc2_shape = []
    osc2_phase = []
    osc2_volume = []
    osc2_freq = []
    osc2_vdepth = []
    osc2_pdepth = []

    osc3_shape = []
    osc3_phase = []
    osc3_volume = []
    osc3_freq = []
    osc3_vdepth = []
    osc3_pdepth = []

    osc4_shape = []
    osc4_phase = []
    osc4_volume = []
    osc4_freq = []
    osc4_vdepth = []
    osc4_pdepth = []

    oscnoise_volume = []

    base_cutoff_hz = []
    base_q = []
    filter_type = []
    envelope_depth = []
    cutoff_mod_depth = []

    lfo1_rate = []
    lfo1_shape = []

    lfo2_rate = []
    lfo2_shape = []

    envelope_decay = []
    envelope_attack = []
    envelope_sustain = []
    envelope_release = []

    filter_envelope_attack = []
    filter_envelope_decay = []
    filter_envelope_sustain = []
    filter_envelope_release = []

    for preset in parameters_file:
        osc1_shape.append(preset['osc1']['shape'])
        osc1_phase.append(preset['osc1']['phase'])
        osc1_volume.append(preset['osc1']['volume'])
        osc1_freq.append(get_freq_log(preset['osc1']['frequency']))
        osc1_vdepth.append(preset['osc1']['volume_mod_depth'])
        osc1_pdepth.append(preset['osc1']['pitch_mod_depth'])

        osc2_shape.append(preset['osc2']['shape'])
        osc2_phase.append(preset['osc2']['phase'])
        osc2_volume.append(preset['osc2']['volume'])
        osc2_freq.append(get_freq_log(preset['osc2']['frequency']))
        osc2_vdepth.append(preset['osc2']['volume_mod_depth'])
        osc2_pdepth.append(preset['osc2']['pitch_mod_depth'])

        osc3_shape.append(preset['osc3']['shape'])
        osc3_phase.append(preset['osc3']['phase'])
        osc3_volume.append(preset['osc3']['volume'])
        osc3_freq.append(get_freq_log(preset['osc3']['frequency']))
        osc3_vdepth.append(preset['osc3']['volume_mod_depth'])
        osc3_pdepth.append(preset['osc3']['pitch_mod_depth'])

        osc4_shape.append(preset['osc4']['shape'])
        osc4_phase.append(preset['osc4']['phase'])
        osc4_volume.append(preset['osc4']['volume'])
        osc4_freq.append(get_freq_log(preset['osc4']['frequency']))
        osc4_vdepth.append(preset['osc4']['volume_mod_depth'])
        osc4_pdepth.append(preset['osc4']['pitch_mod_depth'])

        oscnoise_volume.append(preset['osc_noise']['volume'])

        base_cutoff_hz.append(preset['filter']['cutoff'])
        base_q.append(preset['filter']['q'])
        filter_type.append(preset['filter']['type'])
        envelope_depth.append(preset['filter']['envelope_depth'])
        cutoff_mod_depth.append(preset['filter']['cutoff_mod_depth'])

        lfo1_rate.append(get_freq_log(preset['lfo1']['rate']))
        lfo1_shape.append(preset['lfo1']['shape'])

        lfo2_rate.append(get_freq_log(preset['lfo2']['rate']))
        lfo2_shape.append(preset['lfo2']['shape'])

        envelope_decay.append(preset['amplitude_envelope']['decay'])
        envelope_attack.append(preset['amplitude_envelope']['attack'])
        envelope_sustain.append(preset['amplitude_envelope']['sustain'])
        envelope_release.append(preset['amplitude_envelope']['release'])

        filter_envelope_attack.append(preset['filter_envelope']['attack'])
        filter_envelope_decay.append(preset['filter_envelope']['decay'])
        filter_envelope_sustain.append(preset['filter_envelope']['sustain'])
        filter_envelope_release.append(preset['filter_envelope']['release'])

    return (
        osc1_shape[0],
        osc1_phase[0],
        osc1_volume[0],
        osc1_freq[0],
        osc1_vdepth[0],
        osc1_pdepth[0],
        osc2_shape[0],
        osc2_phase[0],
        osc2_volume[0],
        osc2_freq[0],
        osc2_vdepth[0],
        osc2_pdepth[0],
        osc3_shape[0],
        osc3_phase[0],
        osc3_volume[0],
        osc3_freq[0],
        osc3_vdepth[0],
        osc3_pdepth[0],
        osc4_shape[0],
        osc4_phase[0],
        osc4_volume[0],
        osc4_freq[0],
        osc4_vdepth[0],
        osc4_pdepth[0],
        oscnoise_volume[0],
        base_cutoff_hz[0],
        base_q[0],
        filter_type[0],
        envelope_depth[0],
        cutoff_mod_depth[0],
        lfo1_rate[0],
        lfo1_shape[0],
        lfo2_rate[0],
        lfo2_shape[0],
        envelope_decay[0],
        envelope_attack[0],
        envelope_sustain[0],
        envelope_release[0],
        filter_envelope_attack[0],
        filter_envelope_decay[0],
        filter_envelope_sustain[0],
        filter_envelope_release[0]
    )
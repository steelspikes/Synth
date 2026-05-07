import numpy as np
import matplotlib.pyplot as plt
import json
import librosa
from globals import SAMPLE_RATE
from scipy import signal
from scipy.signal import find_peaks

NUM_PARAMETERS = 42

PARAM_NAMES = [
    'osc1_shape', 'osc1_phase', 'osc1_volume', 'osc1_freq',

    'osc2_shape', 'osc2_phase', 'osc2_volume', 'osc2_freq',

    'osc3_shape', 'osc3_phase', 'osc3_volume', 'osc3_freq',

    'osc4_shape', 'osc4_phase', 'osc4_volume', 'osc4_freq',

    'oscnoise_volume',

    'base_cutoff_hz', 'base_q', 'filter_type',
    'envelope_depth',

    'envelope_decay', 'envelope_attack',
    'envelope_sustain', 'envelope_release',

    'filter_envelope_attack', 'filter_envelope_decay',
    'filter_envelope_sustain', 'filter_envelope_release'
]


def create_morphed_wave(morph_param, base_phase):
    base_phase = base_phase.astype(np.float32)
    morph_param = morph_param.astype(np.float32)
    
    presets = morph_param.shape[0]
    num_waves = 5
    centers = np.linspace(0, num_waves - 1, num_waves, dtype=np.float32)
    centers = np.expand_dims(centers, axis=0)
    centers = np.broadcast_to(centers, (presets, num_waves))

    dists = np.abs(morph_param - centers)

    weights = np.maximum(0, 1.0 - dists)

    morphed_wave = np.broadcast_to(np.zeros_like(base_phase), (presets, base_phase.shape[1])).copy()
    
    w_sine = np.expand_dims(weights[:, 0], axis=1)
    
    if np.any(w_sine > 0):
        morphed_wave += w_sine * get_sine_wave(base_phase)

    w_tri = np.expand_dims(weights[:, 1], axis=1)
    w_saw = np.expand_dims(weights[:, 2], axis=1)

    if np.any(w_tri > 0) or np.any(w_saw > 0):
        saw_raw = get_sawtooth_wave(base_phase)
        tri_raw = get_triangle_wave(base_phase)

        if np.any(w_saw > 0):
            morphed_wave += w_saw * saw_raw
            
        if np.any(w_tri > 0):
            morphed_wave += w_tri * tri_raw   

    w_sq = np.expand_dims(weights[:, 3], axis=1)
    if np.any(w_sq > 0):
        morphed_wave += w_sq * get_square_wave(base_phase, duty=0.5)

    sigmoid = lambda x: 1 / (1 + np.exp(-x))
    k = 10
    alpha = sigmoid(k * (morph_param - 3))
    
    if np.any(alpha > 0.001):
        MIDDLE = 0.5
        w_last = weights[:, 4] 
        duty = np.expand_dims(MIDDLE - (w_last * MIDDLE), axis=1)
        
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
    return 2 * (normalized - np.floor(normalized + 0.5)) 

def get_square_wave(x, duty=0.5):
    return signal.square(x, duty=duty)

def denormalize(n, v_min, v_max):
    return n * (v_max - v_min) + v_min

def normalize(n, v_min, v_max):
    return (n - v_min) / (v_max - v_min)

def log_normalize(f, f_min, f_max):
    log_f = np.log10(f)        # logaritmo de la frecuencia
    log_min = np.log10(f_min)  # logaritmo del mínimo
    log_max = np.log10(f_max)  # logaritmo del máximo
    return normalize(log_f, log_min, log_max)

def log_denormalize(x_norm, f_min, f_max):
    log_min = np.log10(f_min)
    log_max = np.log10(f_max)
    log_f = denormalize(x_norm, log_min, log_max)
    return 10 ** log_f  # Volvemos a Hz

def get_freq_log(freq):
    freq = np.maximum(freq, 0.1)
    return np.log(freq)

def plot_spectrum_linear(waveform, sample_rate, title="Espectro de Frecuencia"):
    if waveform.ndim > 1:
        waveform = waveform.flatten()

    N = len(waveform)
    fft_data = np.fft.rfft(waveform)
    
    magnitude = np.abs(fft_data)
    
    freqs = np.fft.rfftfreq(N, d=1/sample_rate)

    plt.figure(figsize=(10, 6))
    plt.plot(freqs, magnitude)
    plt.title(title)
    plt.xlabel('Frecuencia (Hz)')
    plt.ylabel('Amplitud')
    plt.grid(True, ls='-', alpha=0.5)
    plt.xlim(0, sample_rate / 2)
    plt.show()

def load_parameters_file(name):
    with open(f'presets/{name}', 'r', encoding='utf-8') as file:
        parameters_file = json.load(file)

    osc1_shape = []
    osc1_phase = []
    osc1_volume = []
    osc1_freq = []

    osc2_shape = []
    osc2_phase = []
    osc2_volume = []
    osc2_freq = []

    osc3_shape = []
    osc3_phase = []
    osc3_volume = []
    osc3_freq = []

    osc4_shape = []
    osc4_phase = []
    osc4_volume = []
    osc4_freq = []

    oscnoise_volume = []

    base_cutoff_hz = []
    base_q = []
    filter_type = []
    envelope_depth = []
    cutoff_mod_depth = []

    lfo1_rate = []

    lfo2_rate = []

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
        osc1_freq.append(preset['osc1']['frequency'])

        osc2_shape.append(preset['osc2']['shape'])
        osc2_phase.append(preset['osc2']['phase'])
        osc2_volume.append(preset['osc2']['volume'])
        osc2_freq.append(preset['osc2']['frequency'])

        osc3_shape.append(preset['osc3']['shape'])
        osc3_phase.append(preset['osc3']['phase'])
        osc3_volume.append(preset['osc3']['volume'])
        osc3_freq.append(preset['osc3']['frequency'])

        osc4_shape.append(preset['osc4']['shape'])
        osc4_phase.append(preset['osc4']['phase'])
        osc4_volume.append(preset['osc4']['volume'])
        osc4_freq.append(preset['osc4']['frequency'])

        oscnoise_volume.append(preset['osc_noise']['volume'])

        base_cutoff_hz.append(preset['filter']['cutoff'])
        base_q.append(preset['filter']['q'])
        filter_type.append(preset['filter']['type'])
        envelope_depth.append(preset['filter']['envelope_depth'])
        cutoff_mod_depth.append(preset['filter']['cutoff_mod_depth'])

        lfo1_rate.append(preset['lfo1']['rate'])

        lfo2_rate.append(preset['lfo2']['rate'])

        envelope_decay.append(preset['amplitude_envelope']['decay'])
        envelope_attack.append(preset['amplitude_envelope']['attack'])
        envelope_sustain.append(preset['amplitude_envelope']['sustain'])
        envelope_release.append(preset['amplitude_envelope']['release'])

        filter_envelope_attack.append(preset['filter_envelope']['attack'])
        filter_envelope_decay.append(preset['filter_envelope']['decay'])
        filter_envelope_sustain.append(preset['filter_envelope']['sustain'])
        filter_envelope_release.append(preset['filter_envelope']['release'])

    return {
        'osc1_shape': np.array(osc1_shape),
        'osc1_phase': np.array(osc1_phase),
        'osc1_volume': np.array(osc1_volume),
        'osc1_freq': np.array(osc1_freq),

        'osc2_shape': np.array(osc2_shape),
        'osc2_phase': np.array(osc2_phase),
        'osc2_volume': np.array(osc2_volume),
        'osc2_freq': np.array(osc2_freq),

        'osc3_shape': np.array(osc3_shape),
        'osc3_phase': np.array(osc3_phase),
        'osc3_volume': np.array(osc3_volume),
        'osc3_freq': np.array(osc3_freq),

        'osc4_shape': np.array(osc4_shape),
        'osc4_phase': np.array(osc4_phase),
        'osc4_volume': np.array(osc4_volume),
        'osc4_freq': np.array(osc4_freq),

        'oscnoise_volume': np.array(oscnoise_volume),

        'base_cutoff_hz': np.array(base_cutoff_hz),
        'base_q': np.array(base_q),
        'filter_type': np.array(filter_type),
        'envelope_depth': np.array(envelope_depth),
        'cutoff_mod_depth': np.array(cutoff_mod_depth),

        'lfo1_rate': np.array(lfo1_rate),

        'lfo2_rate': np.array(lfo2_rate),

        'envelope_decay': np.array(envelope_decay),
        'envelope_attack': np.array(envelope_attack),
        'envelope_sustain': np.array(envelope_sustain),
        'envelope_release': np.array(envelope_release),

        'filter_envelope_attack': np.array(filter_envelope_attack),
        'filter_envelope_decay': np.array(filter_envelope_decay),
        'filter_envelope_sustain': np.array(filter_envelope_sustain),
        'filter_envelope_release': np.array(filter_envelope_release)
    }

def from_matrix_to_preset(matrix_population):
    params = {
        PARAM_NAMES[i]: matrix_population[:, i]
        for i in range(matrix_population.shape[1])
    }

    return params

def from_preset_to_matrix(preset):
    return np.array([preset[name] for name in PARAM_NAMES])

def mel_spectrogram(audios, sr, n_fft=2048, hop_length=512, n_mels=128):

    batch_results = []

    for y in audios:
        S = librosa.feature.melspectrogram(
            y=y,
            sr=sr,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
        )
        
        S_db = librosa.power_to_db(S, ref=1.0, top_db=80)
        S_norm = (S_db + 80) / 80

        batch_results.append(S_norm)
        
    return np.array(batch_results)

def spectrogram(audios, n_fft=2048, hop_length=256):
    batch_results = []

    for y in audios:
        S = librosa.stft(
            y=y,
            n_fft=n_fft,
            hop_length=hop_length,
            window='hann'
        )

        S_db = librosa.amplitude_to_db(np.abs(S), ref=1.0, top_db=80)
        S_norm = (S_db + 80) / 80

        batch_results.append(S_norm)
        
    return np.array(batch_results)

def MAE(predictions, target):
    min_time = min(predictions.shape[-1], target.shape[-1])
    
    preds_cut = predictions[..., :min_time]
    target_cut = target[..., :min_time]
    
    return np.mean(np.abs(preds_cut - target_cut), axis=(1, 2))

def manage_normalization(presets, should_normalize):
    MIN_FREQ = 20
    MAX_FREQ = SAMPLE_RATE / 2
    
    fun = normalize if should_normalize else denormalize 
    funlog = log_normalize if should_normalize else log_denormalize
    return {
        # Osc 1
        "osc1_shape": fun(presets['osc1_shape'], 0, 4),
        "osc1_phase": fun(presets['osc1_phase'], 0, 1),
        "osc1_volume": fun(presets['osc1_volume'], 0, 1),
        "osc1_freq": funlog(presets['osc1_freq'], MIN_FREQ, MAX_FREQ),

        # Osc 2
        "osc2_shape": fun(presets['osc2_shape'], 0, 4),
        "osc2_phase": fun(presets['osc2_phase'], 0, 1),
        "osc2_volume": fun(presets['osc2_volume'], 0, 1),
        "osc2_freq": funlog(presets['osc2_freq'], MIN_FREQ, MAX_FREQ),

        # Osc 3
        "osc3_shape": fun(presets['osc3_shape'], 0, 4),
        "osc3_phase": fun(presets['osc3_phase'], 0, 1),
        "osc3_volume": fun(presets['osc3_volume'], 0, 1),
        "osc3_freq": funlog(presets['osc3_freq'], MIN_FREQ, MAX_FREQ),

        # Osc 4
        "osc4_shape": fun(presets['osc4_shape'], 0, 4),
        "osc4_phase": fun(presets['osc4_phase'], 0, 1),
        "osc4_volume": fun(presets['osc4_volume'], 0, 1),
        "osc4_freq": funlog(presets['osc4_freq'], MIN_FREQ, MAX_FREQ),

        # Osc Noise
        "oscnoise_volume": fun(presets['oscnoise_volume'], 0, 1),

        # Filter envelope
        "filter_envelope_attack": fun(presets['filter_envelope_attack'], 0, 0.5),
        "filter_envelope_decay": fun(presets['filter_envelope_decay'], 0, 1),
        "filter_envelope_sustain": fun(presets['filter_envelope_sustain'], 0, 1),
        "filter_envelope_release": fun(presets['filter_envelope_release'], 0, 0.5),

        # Biquad filter
        "base_cutoff_hz": funlog(presets['base_cutoff_hz'], MIN_FREQ, MAX_FREQ),
        "filter_type": fun(presets['filter_type'], 0, 2),
        "base_q": funlog(presets['base_q'], 0.707, 20),
        "envelope_depth": fun(presets['envelope_depth'], 0, 1),

        # Amplitude envelope
        "envelope_attack": fun(presets['envelope_attack'], 0, 0.2),
        "envelope_decay": fun(presets['envelope_decay'], 0, 0.3),
        "envelope_sustain": fun(presets['envelope_sustain'], 0, 1),
        "envelope_release": fun(presets['envelope_release'], 0, 0.5),
    }


def denormalize_preset(preset):
    return manage_normalization(preset, False)

def normalize_preset(preset):
    return manage_normalization(preset, True)

def pretty_print(obj, indent=0):
    spacing = "  " * indent
    if isinstance(obj, dict):
        print(f"{spacing}{{")
        for k, v in obj.items():
            print(f"{spacing}  {k}: ", end="")
            pretty_print(v, indent + 1)
        print(f"{spacing}}}")
    elif isinstance(obj, list) or isinstance(obj, np.ndarray):
        if isinstance(obj, np.ndarray):
            obj = obj.tolist()
        print(f"{spacing}[")
        for item in obj:
            pretty_print(item, indent + 1)
        print(f"{spacing}]")
    else:
        print(f"{spacing}{obj}")

def get_audio(audio_path, top_db=20):
    y, _sr = librosa.load(audio_path, sr=SAMPLE_RATE)
    y = np.array(y, dtype=np.float64)
    y, _ = librosa.effects.trim(y, top_db=top_db)

    rms_target = 0.1
    rms_actual = np.sqrt(np.mean(y**2))
    scale = rms_target / (rms_actual + 1e-8)
    return y * scale

# Función artesanal anterior
def __split_audio__(audio, prominence=0.01, win_size_ms=0.02):
    win_size = int(win_size_ms * SAMPLE_RATE)

    rectified_signal = np.abs(audio)
    window = np.ones(win_size) / win_size
    envelope = np.convolve(rectified_signal, window, mode='same')

    minimals, _ = find_peaks(-envelope, prominence=prominence, width=10, distance=2000)
    minimals = np.concatenate((np.array([0]), minimals))

    minimals = minimals + (win_size // 2)

    audios = []
    
    for i in range(minimals.shape[0]):
        if i < minimals.shape[0] - 1:
            audios.append(audio[minimals[i] : minimals[i+1]])
        else:
            audios.append(audio[minimals[i]:])

    return audios



def split_audio(audio, sr, backtrack=True, delta=0.1):
    # 1. Detectar los onsets (momentos de inicio de cada nota)
    # units='samples' nos devuelve el índice exacto en el array
    # backtrack=True ajusta el punto un poco hacia atrás para no comerse el ataque
    onsets = librosa.onset.onset_detect(y=audio, sr=sr, units='samples', backtrack=backtrack, delta=delta)
    
    # 2. Asegurarnos de incluir el inicio y el final del audio para no perder trozos
    # Si el primer onset no es 0, lo agregamos
    if len(onsets) == 0 or onsets[0] != 0:
        onsets = np.concatenate(([0], onsets))
    
    # Agregamos el final del audio como último punto de corte
    onsets = np.concatenate((onsets, [len(audio)]))
    
    audios = []
    
    # 3. Recortar el audio nota por nota
    for i in range(len(onsets) - 1):
        inicio = onsets[i]
        fin = onsets[i+1]
        
        # Extraemos el segmento
        segmento = audio[inicio:fin]
        
        # Filtro de seguridad: si el segmento es demasiado corto (ej. ruido), lo ignoramos
        if len(segmento) > (sr * 0.05): # mínimo 50ms
            audios.append(segmento)
            
    return audios
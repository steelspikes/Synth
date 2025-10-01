import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt

# --- Parámetros Globales ---
sample_rate = 44100  # Frecuencia de muestreo en Hz
duration = 2.0       # Duración en segundos
amplitude = 0.5      # Amplitud (volumen)
frequency = 440.0    # Frecuencia en Hz (La central, A4)

def semitones_to_frequency(frequency, n_semitonos):
    factor = 2**(n_semitonos / 12.0)
    return frequency * factor

# --- FUNCIÓN DE UNISON ---
def create_unison_sound(base_freq, num_voices=5, detune_cents=15, shape='sawtooth'):
    num_samples = int(sample_rate * duration)
    
    # Crea un buffer estéreo vacío (Izquierda, Derecha)
    final_wave = np.zeros((num_samples, 2))
    
    # Genera un espaciado lineal para la desafinación y el panorama
    detune_amounts = np.linspace(-detune_cents, detune_cents, num_voices)
    pan_positions = np.linspace(-1.0, 1.0, num_voices) # -1=Izquierda, 0=Centro, 1=Derecha
    
    for i in range(num_voices):
        # 1. Calcular la frecuencia de esta voz
        # La fórmula para convertir cents a un factor de frecuencia es 2^(cents/1200)
        cents = detune_amounts[i]
        freq_ratio = 2**(cents / 1200.0)
        voice_freq = base_freq * freq_ratio
        
        # 2. Generar la onda para esta voz
        mono_voice = create_oscillator(voice_freq, shape=shape)
        
        # 3. Aplicar el panorama estéreo (Constant Power Panning)
        pan = pan_positions[i]
        angle = (pan * 0.5 + 0.5) * (np.pi / 2) # Mapea de [-1, 1] a [0, pi/2]
        gain_left = np.cos(angle)
        gain_right = np.sin(angle)
        
        # Añadir la voz paneada al buffer final
        final_wave[:, 0] += mono_voice * gain_left
        final_wave[:, 1] += mono_voice * gain_right

    # Normalizar la salida para evitar distorsión (clipping)
    # Dividimos por un factor relacionado con el número de voces
    final_wave /= np.sqrt(num_voices) 
    
    return final_wave * amplitude

def create_oscillator(freq, shape='sine', volume=1):
    """Genera una forma de onda para un oscilador."""
    
    # Crea un arreglo de tiempo desde 0 hasta la duración
    t = np.linspace(0., duration, int(sample_rate * duration), endpoint=False)
    
    # Calcula el argumento de la función de onda (fasor)
    phase = 2 * np.pi * freq * t
    
    # Genera la forma de onda según la selección
    if shape == 'sine':
        waveform = amplitude * np.sin(phase)
    elif shape == 'square':
        waveform = amplitude * np.sign(np.sin(phase))
    elif shape == 'sawtooth':
        # t % (1/freq) normaliza el tiempo para cada ciclo
        # (freq * (...)) escala la rampa a la frecuencia deseada
        waveform = amplitude * (2 * (t * freq - np.floor(0.5 + t * freq)))
    elif shape == 'triangle':
        waveform = amplitude * (2 * np.abs(2 * (t * freq - np.floor(0.5 + t * freq))) - 1)
    elif shape == 'noise':
        waveform = np.random.uniform(-1.0, 1.0, size=int(sample_rate * duration))
    else:
        raise ValueError("Forma de onda no soportada. Elige entre: 'sine', 'square', 'sawtooth', 'triangle'")
        
    return waveform * volume

# --- Generar y Reproducir Sonido ---
# Para instalar sounddevice: pip install sounddevice

# 1. Genera una onda sinusoidal
osc1 = create_oscillator(semitones_to_frequency(frequency, -12), shape='sine', volume=0)
osc2 = create_oscillator(frequency, shape='square', volume=0)
osc3 = create_oscillator(frequency, shape='sawtooth', volume=0)
osc4 = create_oscillator(frequency, shape='noise', volume=1)

result_wave = osc1 + osc2 + osc3 + osc4

plt.plot(result_wave[:15000])
plt.show()

# # Reproducir sonido
# print("Reproduciendo onda cuadrada...")
# sd.play(create_unison_sound(frequency), sample_rate)
# sd.wait()
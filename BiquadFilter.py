import numpy as np
from numba import jit
from LFO import LFO
# import matplotlib.pyplot as plt
from Envelope import Envelope

# --- La función JIT ahora acepta un 'filter_type_code' ---
@jit(nopython=True)
def biquad_process_jit(input_wave, modulated_cutoff, modulated_q, sample_rate, filter_type_code):
    num_samples = len(input_wave)
    output_wave = np.zeros(num_samples)
    x1, x2, y1, y2 = 0.0, 0.0, 0.0, 0.0
    
    for n in range(num_samples):
        # --- Recalcular coeficientes comunes para la muestra actual ---
        cutoff = modulated_cutoff[n]
        Q = modulated_q[n]
        w0 = 2 * np.pi * cutoff / sample_rate
        cos_w0 = np.cos(w0)
        alpha = np.sin(w0) / (2 * Q)
        
        # --- SELECCIÓN DE FÓRMULAS BASADA EN EL TIPO DE FILTRO ---
        # if filter_type_code == 0: # LPF (Pasa-Bajas)
        #     b0 = (1 - cos_w0) / 2
        #     b1 = 1 - cos_w0
        #     b2 = (1 - cos_w0) / 2
        # elif filter_type_code == 1: # HPF (Pasa-Altas)
        #     b0 = (1 + cos_w0) / 2
        #     b1 = -(1 + cos_w0)
        #     b2 = (1 + cos_w0) / 2
        # elif filter_type_code == 2: # BPF (Pasa-Banda)
        #     b0 = alpha
        #     b1 = 0
        #     b2 = -alpha
        # else: # Default a LPF si el código es inválido
        #     b0 = (1 - cos_w0) / 2
        #     b1 = 1 - cos_w0
        #     b2 = (1 - cos_w0) / 2

        b0_lpf = (1 - cos_w0) / 2
        b1_lpf = 1 - cos_w0
        b2_lpf = (1 - cos_w0) / 2
        # HPF
        b0_hpf = (1 + cos_w0) / 2
        b1_hpf = -(1 + cos_w0)
        b2_hpf = (1 + cos_w0) / 2
        # BPF
        b0_bpf = alpha
        b1_bpf = 0
        b2_bpf = -alpha

        if filter_type_code <= 1.0:
            # Interpolar entre HPF (0.0) y BPF (1.0)
            blend = filter_type_code
            b0 = (1.0 - blend) * b0_hpf + blend * b0_bpf
            b1 = (1.0 - blend) * b1_hpf + blend * b1_bpf
            b2 = (1.0 - blend) * b2_hpf + blend * b2_bpf
        else:
            # Interpolar entre BPF (1.0) y LPF (2.0)
            blend = filter_type_code - 1.0
            b0 = (1.0 - blend) * b0_bpf + blend * b0_lpf
            b1 = (1.0 - blend) * b1_bpf + blend * b1_lpf
            b2 = (1.0 - blend) * b2_bpf + blend * b2_lpf

        # Coeficientes 'a' (son los mismos para LPF, HPF, BPF)
        a0 = 1 + alpha
        a1 = -2 * cos_w0
        a2 = 1 - alpha
        
        # Normalizar y aplicar la ecuación del filtro
        b0, b1, b2 = b0/a0, b1/a0, b2/a0
        a1, a2 = a1/a0, a2/a0
        
        x0 = input_wave[n]
        y0 = b0*x0 + b1*x1 + b2*x2 - a1*y1 - a2*y2
        
        output_wave[n] = y0
        x2, x1 = x1, x0
        y2, y1 = y1, y0
        
    return output_wave


class BiquadFilter:
    def __init__(self, sample_rate=44100, base_cutoff_hz=1000, base_q=0.707, filter_type=0, lfo1_instance: LFO = None, lfo2_instance: LFO = None, envelope: Envelope = None, cutoff_mod_depth = 0, lfo_choose = 0):
        self.sample_rate = sample_rate
        self.base_cutoff_hz = base_cutoff_hz
        self.base_q = base_q
        self.filter_type = filter_type
        self.lfo_instance = lfo1_instance if lfo_choose == 0 else lfo2_instance
        self.envelope = envelope
        self.cutoff_mod_depth = cutoff_mod_depth
        
        # Mapeo de string a código numérico
        # self.filter_type_map = {'LPF': 0, 'HPF': 1, 'BPF': 2}
        
    def process(self, input_wave):
         # Empezamos con el valor base
        num_samples = len(input_wave)
        modulated_cutoff = np.full(num_samples, self.base_cutoff_hz, dtype=np.float64)
        # modulated_q = np.full(num_samples, self.base_q, dtype=np.float64)
        nyquist = 0.5 * self.sample_rate

        # Rango total del barrido
        rango_hz = nyquist - self.base_cutoff_hz
        rango_q = 1 - self.base_q
        
        # Añadir modulación de la envolvente
        if self.envelope is not None:
            env_signal = self.envelope.process(num_samples / self.sample_rate) # de 0 a 1
            modulated_cutoff += (env_signal * rango_hz * 1)
            
        # Añadir modulación del LFO
        if self.lfo_instance is not None:
            lfo_signal = self.lfo_instance.process(num_samples) # de -1 a 1
            lfo_signal = (lfo_signal + 1) / 2

            # Fórmula de mapeo
            modulated_cutoff += (lfo_signal * rango_hz * self.cutoff_mod_depth)
            # modulated_q += (lfo_signal * rango_q * self.lfo_instance.depth)

        # plt.plot(modulated_cutoff)
        # plt.show()
            
        modulated_cutoff = np.clip(modulated_cutoff, 20, nyquist - 1)
    
        modulated_q = np.full(num_samples, self.base_q)
        
        # Convertir el tipo de filtro a su código numérico
        # filter_code = self.filter_type_map.get(self.filter_type, 0) # Default a 0 (LPF)
        
        # Llamar a la función JIT optimizada con el código del filtro
        return biquad_process_jit(input_wave, modulated_cutoff, modulated_q, self.sample_rate, self.filter_type)
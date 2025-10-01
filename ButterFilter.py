from scipy import signal
import numpy as np
from LFO import LFO
import matplotlib.pyplot as plt
from Envelope import Envelope
from filters.Biquad import BiquadFilter

class ButterFilter:
    def __init__(self, sample_rate=44_100, cutoff_hz=1000, filter_type='LPF', order='12', lfo_instance: LFO =None, envelope_instance: Envelope = None):
        self.sample_rate = sample_rate
        self.cutoff_hz = cutoff_hz
        self.filter_type = filter_type
        self.order = order
        self.lfo_instance = lfo_instance
        # self.lfo_amount_hz = 800
        self.envelope_instace = envelope_instance

    def process(self, input_wave, buffer_size=64):
        """Procesa una onda de audio, aplicando el filtro y sus modulaciones."""
        num_samples = len(input_wave)
        nyquist = 0.5 * self.sample_rate
        
        # Empezamos con el valor base
        modulated_cutoff = np.full(num_samples, self.cutoff_hz, dtype=np.float64)

        # Rango total del barrido
        rango_hz = nyquist - self.cutoff_hz
        
        # Añadir modulación de la envolvente
        if self.envelope_instace is not None:
            env_signal = self.envelope_instace.process(num_samples / self.sample_rate) # de 0 a 1
            modulated_cutoff += (env_signal * rango_hz * 1)
            
        # Añadir modulación del LFO
        if self.lfo_instance is not None:
            lfo_signal = self.lfo_instance.process(num_samples) # de -1 a 1
            lfo_signal = (lfo_signal + 1) / 2

            # Fórmula de mapeo
            modulated_cutoff += (lfo_signal * rango_hz * self.lfo_instance.depth)

        # plt.plot(modulated_cutoff)
        # plt.show()
            
        # Asegurar que el cutoff se mantenga en un rango válido
        modulated_cutoff = np.clip(modulated_cutoff, 20, nyquist - 1)

        # --- 2. PROCESAR LA ONDA EN BLOQUES (BUFFERS) ---
        output_wave = np.zeros_like(input_wave)
        orderN = {'12': 2, '24': 4}[self.order]
        scipy_btype = {'LPF': 'lowpass', 'HPF': 'highpass', 'BPF': 'bandpass'}[self.filter_type]
        
        current_cutoff = modulated_cutoff[0]

        if self.filter_type == 'BPF':
            f_center = current_cutoff
            bandwidth = 2_000  # Hz, ejemplo
            f_low = max(f_center - bandwidth/2, 0)
            f_high = min(f_center + bandwidth/2, nyquist)
            normal_cutoff = [f_low/nyquist, f_high/nyquist]
        else:
            normal_cutoff = current_cutoff / nyquist

        sos = signal.butter(orderN, normal_cutoff, btype=scipy_btype, output='sos')
        zi = signal.sosfilt_zi(sos) * input_wave[0]  # escala con la primera muestra

        for i in range(0, num_samples, buffer_size):
            end = i + buffer_size
            audio_chunk = input_wave[i:end]
            cutoff_chunk = modulated_cutoff[i:end]
            
            current_cutoff = np.mean(cutoff_chunk)

            if self.filter_type == 'BPF':
                f_center = current_cutoff
                bandwidth = 2000  # Hz, ejemplo
                f_low = max(f_center - bandwidth/2, 0)
                f_high = min(f_center + bandwidth/2, nyquist)
                normal_cutoff = [f_low/nyquist, f_high/nyquist]
            else:
                normal_cutoff = current_cutoff / nyquist
            
            normal_cutoff = np.clip(normal_cutoff, 1e-6, 1 - 1e-6)

            # Rediseñamos el filtro solo si cambió mucho
            sos_new = signal.butter(orderN, normal_cutoff, btype=scipy_btype, output='sos')
            
            # Opcional: interpolar entre sos y sos_new si la modulación es rápida
            # Por simplicidad, usamos sos_new directamente
            sos = sos_new
            
            # Filtramos manteniendo el estado
            output_chunk, zi = signal.sosfilt(sos, audio_chunk, zi=zi)
            output_wave[i:end] = output_chunk
            
        return output_wave
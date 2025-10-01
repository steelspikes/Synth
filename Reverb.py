import numpy as np
from scipy import signal
from scipy.io import wavfile
import matplotlib.pyplot as plt

class Reverb:
    def __init__(self, sample_rate = 44100):
        self.sample_rate = sample_rate
        self.reverb_ir = None

        self.wet_level = 1  # (WET) 0.0 a 1.0
        self.size = 1       # (SIZE) 0.1 a 1.0 (como porcentaje del IR original)
        self.pre_delay_ms = 1 # (PRE) en milisegundos
        self.hp_cutoff_hz = 20 # (HP) Frecuencia de corte del Pasa-Altas
        self.lp_cutoff_hz = 20000 # (LP) Frecuencia de corte del Pasa-Bajas

    def _get_ir(self):
        _, ir_data = wavfile.read("reverb_ir.wav")
        ir_data = ir_data.astype(np.float64) / 32767.0
        if ir_data.ndim > 1:
            ir_data = ir_data[:, 0]
        self.reverb_ir = ir_data

    def _get_processed_ir(self):
        """Modifica el IR original según los parámetros actuales."""
        if self.reverb_ir is None:
            return None

        processed_ir = self.reverb_ir.copy()

        # 1. Aplicar SIZE (acortar el IR)
        new_length = int(len(processed_ir) * self.size)
        processed_ir = processed_ir[:new_length]

        # 2. Aplicar filtros HP y LP
        nyquist = 0.5 * self.sample_rate
        if self.lp_cutoff_hz < nyquist - 1:
            sos_lp = signal.butter(4, self.lp_cutoff_hz / nyquist, btype='lowpass', output='sos')
            processed_ir = signal.sosfilt(sos_lp, processed_ir)
        if self.hp_cutoff_hz > 20:
            sos_hp = signal.butter(4, self.hp_cutoff_hz / nyquist, btype='highpass', output='sos')
            processed_ir = signal.sosfilt(sos_hp, processed_ir)
            
        # 3. Aplicar PRE-DELAY (añadir silencio al inicio)
        pre_delay_samples = int(self.pre_delay_ms / 1000 * self.sample_rate)
        if pre_delay_samples > 0:
            silence = np.zeros(pre_delay_samples)
            processed_ir = np.concatenate((silence, processed_ir))
            
        return processed_ir

    def process(self, input_signal):
        self._get_ir()

        wet_signal = signal.fftconvolve(input_signal, self._get_processed_ir(), mode='full')

        dry_padded = np.zeros_like(wet_signal)
        dry_padded[:len(input_signal)] = input_signal
        
        dry_mix = 1.0 - self.wet_level
        wet_mix = self.wet_level
        
        final_output = (dry_padded * dry_mix) + (wet_signal * wet_mix)
        
        # Normalizar para evitar clipping
        max_abs = np.max(np.abs(final_output))
        if max_abs > 0:
            final_output /= max_abs
            
        return final_output
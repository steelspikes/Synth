import numpy as np
from scipy import signal

def create_morphed_wave(shape, phase):
    if shape <= 1.0:
        sine_wave = np.sin(phase)
        triangle_wave = signal.sawtooth(phase, 0.5)
        blend = shape
        return (1 - blend) * sine_wave + blend * triangle_wave

    elif shape <= 2.0:
        blend = shape - 1.0
        width = 0.5 + blend * 0.5
        return signal.sawtooth(phase, width)

    elif shape <= 3.0:
        saw_wave = signal.sawtooth(phase, 1)
        square_wave = signal.square(phase)
        blend = shape - 2.0
        return (1 - blend) * saw_wave + blend * square_wave

    else: # shape > 3.0
        blend = shape - 3.0
        duty_cycle = 0.5 + (0.5 * blend)
        duty_cycle = min(duty_cycle, 1) 
        return signal.square(phase, duty=duty_cycle)
    
def denormalize(n, v_min, v_max):
    return v_min + n * (v_max - v_min)
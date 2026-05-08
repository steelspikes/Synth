
import numpy as np
from Libs.globals import PROCESSORS, SAMPLE_RATE
from Libs.Utils import from_matrix_to_preset, denormalize_preset, PARAM_NAMES
import numpy as np
import cma
from multiprocessing import Pool
from Libs.parallelEvaluation import evaluate_presets
from scipy.optimize import differential_evolution
from Synth.main import Synth
from Libs.Utils import mel_spectrogram

def render_presets(presets, duration=0):
    synth = Synth(
        sample_rate=SAMPLE_RATE,
        duration=duration,
        presets=presets
    )
    return synth.process_audio().astype(np.float64)

def evaluate_target(audio):
    return mel_spectrogram(audio, sr=SAMPLE_RATE, n_fft=2048, hop_length=256, n_mels=128)

def search_with_DE(target_C, duration, maxiter=150, popsize=10, mutation=(0.6, 0.9), recombination=0.8, tol=1e-4, strategy='rand1bin', disp=False, x0=None):
    bounds = [(0, 1)] * len(PARAM_NAMES)

    with Pool(PROCESSORS) as pool:
        episodes = []

        def get_fitness(solutions):
            solutions = np.array(solutions).T
            solutions_splitted = np.array_split(solutions, PROCESSORS)
            presets_splitted = [(denormalize_preset(from_matrix_to_preset(chunk)), target_C, duration) for chunk in solutions_splitted]
            
            solutions_evaluated = pool.map(evaluate_presets, presets_splitted)
            return np.concatenate(solutions_evaluated).tolist()
        
        def callback(xk, convergence):
            error_actual = evaluate_presets((denormalize_preset(from_matrix_to_preset(np.array([xk]))), target_C, duration))
            episodes.append(error_actual[0])
            
        result = differential_evolution(
            get_fitness,
            bounds=bounds,
            popsize=popsize,        # 10 × n_params individuos
            mutation=mutation,
            recombination=recombination,
            tol=tol,
            maxiter=maxiter,
            polish=False,
            seed=None,
            disp=disp,
            workers=1,
            vectorized=True,
            strategy=strategy,
            callback=callback,
            x0=x0
        )

        x_best = result.x

        return x_best, episodes

def search_with_CMA(target_C, duration, x0, repeat_times=3, sigma0=0.08, popsize=30, tolfun=1e-3, tolx=1e-3, tolfunhist=1e-3, disp=False):
    with Pool(PROCESSORS) as pool:
        episodes = []

        def get_fitness(solutions):
            solutions = np.array(solutions)
            solutions_splitted = np.array_split(solutions, PROCESSORS)
            presets_splitted = [(denormalize_preset(from_matrix_to_preset(chunk)), target_C, duration) for chunk in solutions_splitted]
            
            solutions_evaluated = pool.map(evaluate_presets, presets_splitted)
            return np.concatenate(solutions_evaluated)
        
        bestever = cma.optimization_tools.BestSolution()

        for i in range(repeat_times):
            options = {
                'popsize': popsize,
                'bounds': [np.zeros_like(x0), np.ones_like(x0)],
                'tolfun': tolfun,
                'tolx': tolx,
                'tolfunhist': tolfunhist
            }

            es = cma.CMAEvolutionStrategy(x0, sigma0, options)
            gen = 1

            while not es.stop():
                solutions = np.array(es.ask())  # devuelve una lista de individuos

                fitnesses = get_fitness(solutions)

                best_idx = np.argmin(fitnesses)          # índice del mejor fitness
                best_fitness = fitnesses[best_idx]       # fitness correspondiente

                episodes.append(best_fitness)

                if disp and gen % 100 == 0:
                    print("Gen", gen, "Mejor fitness:", best_fitness, "Sigma", es.sigma, "Restart", i + 1)

                es.tell(solutions, fitnesses)  # pasar fitness al algoritmo

                gen += 1

            bestever.update(es.best)
        
        return bestever.get()[0], episodes
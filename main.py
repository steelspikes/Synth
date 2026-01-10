import sounddevice as sd
from Utils import load_parameters_file, mfcc, from_matrix_to_preset, NUM_PARAMETERS, MSE
import time
from Synth import Synth
import numpy as np
import cma

DURATION = 2
SAMPLE_RATE = 44100

def evaluate_presets(presets):
    synth = Synth(
        sample_rate=SAMPLE_RATE,
        duration=DURATION,
        presets=presets
    )
    audio = synth.process_audio()
    mfcc_coefs = mfcc(audio, n_mfcc=13, n_mels=26, n_fft=1024, hop_length=256)

    return mfcc_coefs

def render_presets(presets):
    synth = Synth(
        sample_rate=SAMPLE_RATE,
        duration=DURATION,
        presets=presets
    )
    return synth.process_audio()

if __name__ == "__main__":
    import time
    from multiprocessing import Pool

    print('Starting...')
    
    # gen_presets = np.zeros((100, NUM_PARAMETERS))
    # presets = from_matrix_to_preset(gen_presets)
    # predicted_mfcc = evaluate_presets(load_parameters_file('parameters.json'))

    target_mfcc = evaluate_presets(load_parameters_file('target.json'))

    x0 = np.zeros(NUM_PARAMETERS)
    sigma0 = 0.5

    # Crear estrategia
    es = cma.CMAEvolutionStrategy(x0, sigma0, {'popsize': 100, 'maxiter': 100})

    while not es.stop():
        # 1️⃣ Generar población
        solutions = np.clip(np.array(es.ask()), 0, None)  # devuelve una lista de individuos
        presets = from_matrix_to_preset(solutions)
        fitnesses = MSE(evaluate_presets(presets), target_mfcc)

        best_idx = np.argmin(fitnesses)          # índice del mejor fitness
        best_solution = solutions[best_idx]      # solución correspondiente
        best_fitness = fitnesses[best_idx]       # fitness correspondiente

        print("Mejor fitness:", best_fitness, best_solution)

        es.tell(solutions, fitnesses)  # pasar fitness al algoritmo
        es.logger.add()  # opcional, para logging interno
        # es.disp()        # imprimir estado de iteración

    # # Mejor solución
    # best_solution = es.result.xbest
    # best_fitness = egg_f(best_solution)  # aquí usamos función original
    # print("Mejor individuo:", best_solution)
    # print("Valor minimo:", best_fitness)








    # with Pool(5) as p:
    #     for i in range(10):
    #         start = time.time()
    #         r = p.map(load_synth, presets)
    #         print([batch.shape for batch in r])

    #         end = time.time()
    #         print(end - start)

#     presets = [load_parameters_file(),load_parameters_file(), load_parameters_file(), load_parameters_file(),load_parameters_file()]

#     with Pool(5) as p:
#         for i in range(10):
#             start = time.time()
#             r = p.map(load_synth, presets)
#             print([batch.shape for batch in r])

#             end = time.time()
#             print(end - start)

#     # from joblib import Parallel, delayed

#     # for i in range(10):
#     #     start = time.time()
        
#     #     # n_jobs=5 → equivalente a Pool(5)
#     #     r = Parallel(n_jobs=5)(
#     #         delayed(load_synth)(preset) for preset in presets
#     #     )
        
#     #     print([batch.shape for batch in r])
        
#     #     end = time.time()
#     #     print(end - start)
    
#     # start = time.time()
#     # audios = load_synth(load_parameters_file())
#     # mfcc_coefs = mfcc(audios)
#     # print(mfcc_coefs.shape)

#     # end = time.time()
#     # print(end - start)


# # for i in audios:
# #     print(i)

# #     sd.play(i, samplerate=44_100)
# #     sd.wait()
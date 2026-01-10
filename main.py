# Trash file



# if __name__ == "__main__":
    

    








#     # with Pool(5) as p:
#     #     for i in range(10):
#     #         start = time.time()
#     #         r = p.map(load_synth, presets)
#     #         print([batch.shape for batch in r])

#     #         end = time.time()
#     #         print(end - start)

# #     presets = [load_parameters_file(),load_parameters_file(), load_parameters_file(), load_parameters_file(),load_parameters_file()]

# #     with Pool(5) as p:
# #         for i in range(10):
# #             start = time.time()
# #             r = p.map(load_synth, presets)
# #             print([batch.shape for batch in r])

# #             end = time.time()
# #             print(end - start)

# #     # from joblib import Parallel, delayed

# #     # for i in range(10):
# #     #     start = time.time()
        
# #     #     # n_jobs=5 → equivalente a Pool(5)
# #     #     r = Parallel(n_jobs=5)(
# #     #         delayed(load_synth)(preset) for preset in presets
# #     #     )
        
# #     #     print([batch.shape for batch in r])
        
# #     #     end = time.time()
# #     #     print(end - start)
    
# #     # start = time.time()
# #     # audios = load_synth(load_parameters_file())
# #     # mfcc_coefs = mfcc(audios)
# #     # print(mfcc_coefs.shape)

# #     # end = time.time()
# #     # print(end - start)


# # # for i in audios:
# # #     print(i)

# # #     sd.play(i, samplerate=44_100)
# # #     sd.wait()
import pickle
import glob
import os

benchmark_l = []

for filename in glob.glob('./noisy_benchmark_data/*_uniter_output_*.p'):
    noisy_benchmark = pickle.load(open(filename, 'rb' ))
    benchmark_l.append(noisy_benchmark)
    os.remove(filename)

    filename = filename.split('/')[-1].split('.')[0]
    max_qubit = int(filename.split('_')[3])
    max_clusters = int(filename.split('_')[5])
    num_shots = int(filename.split('_')[7])
    cluster_shots = int(filename.split('_')[8])

filename = filename.replace('uniter_output','plotter_input')
filename = filename.split('_shots_')[0]
pickle.dump(benchmark_l,
open('./noisy_benchmark_data/%s.p'%(filename),'ab'))

for filename in glob.glob('./noisy_benchmark_data/*_evaluator_input*.p'):
    os.remove(filename)
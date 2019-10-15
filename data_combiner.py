import pickle
import glob
import os

benchmark_l = []

for filename in glob.glob("./noisy_benchmark_data/uniter_output_*.p"):
    noisy_benchmark = pickle.load(open(filename, 'rb' ))
    circ, evaluations, searcher_time, classical_time, quantum_time, uniter_time = noisy_benchmark
    benchmark_l.append(noisy_benchmark)
    os.remove(filename)

    filename = filename.split('/')[-1].split('.')[0]
    max_qubit = int(filename.split('_')[2])
    max_clusters = int(filename.split('_')[4])
    num_shots = int(filename.split('_')[6])

pickle.dump(benchmark_l, open('./noisy_benchmark_data/plotter_input_%d_qubits_%d_clusters_%d_shots.p'%(max_qubit,max_clusters,num_shots),'wb'))
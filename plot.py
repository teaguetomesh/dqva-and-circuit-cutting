import pickle
import matplotlib.pyplot as plt
import scipy.optimize as opt
import numpy as np
from scipy.stats import wasserstein_distance

noisy_benchmark = pickle.load(open('./data/noisy_benchmark_10000_shots_5_qubits.p', 'rb' ))
def func(x, a, b):
    return np.exp(a*x)+b

num_qubits,times,sv_noiseless_fc_l,qasm_noiseless_fc_l,qasm_noisy_fc_l,qasm_noisy_na_fc_l,qasm_noisy_na_cutting_l = noisy_benchmark

qasm_distances = [wasserstein_distance(sv_noiseless_fc_l[i],qasm_noiseless_fc_l[i]) for i in range(len(num_qubits))]
qasm_noise_distances = [wasserstein_distance(sv_noiseless_fc_l[i],qasm_noisy_fc_l[i]) for i in range(len(num_qubits))]
qasm_noise_na_distances = [wasserstein_distance(sv_noiseless_fc_l[i],qasm_noisy_na_fc_l[i]) for i in range(len(num_qubits))]
qasm_noise_na_cutting_distances = [wasserstein_distance(sv_noiseless_fc_l[i],qasm_noisy_na_cutting_l[i]) for i in range(len(num_qubits))]

plt.figure(figsize=(10,10))
plt.subplot(221)
plt.plot(num_qubits,times['searcher'],'^',label='cut searcher')
optimizedParameters, pcov = opt.curve_fit(func, np.array(num_qubits), np.array(times['searcher']))
plt.plot(num_qubits, func(np.array(num_qubits), *optimizedParameters), label="fit")
plt.xlabel('supremacy circuit # qubits')
plt.ylabel('runtime (s)')
plt.legend()
plt.subplot(222)
plt.plot(num_qubits,times['evaluator'],'^',label='cluster evaluator')
optimizedParameters, pcov = opt.curve_fit(func, np.array(num_qubits), np.array(times['evaluator']))
plt.plot(num_qubits, func(np.array(num_qubits), *optimizedParameters), label="fit")
plt.xlabel('supremacy circuit # qubits')
plt.legend()
plt.subplot(223)
plt.plot(num_qubits,times['uniter'],'^',label='uniter')
optimizedParameters, pcov = opt.curve_fit(func, np.array(num_qubits), np.array(times['uniter']))
plt.plot(num_qubits, func(np.array(num_qubits), *optimizedParameters), label="fit")
plt.xlabel('supremacy circuit # qubits')
plt.ylabel('runtime (s)')
plt.legend()
plt.subplot(224)
plt.plot(num_qubits,qasm_distances,'o',label='qasm')
plt.plot(num_qubits,qasm_noise_distances,'o',label='qasm+noise')
plt.plot(num_qubits,qasm_noise_na_distances,'o',label='qasm+noise+na')
plt.plot(num_qubits,qasm_noise_na_cutting_distances,'X',label='qasm+noise+na+cutting')
plt.yscale('log')
plt.xlabel('supremacy circuit # qubits')
plt.ylabel('wasserstein distance')
plt.legend()
plt.suptitle('Noisy Circuit Cutting Benchmark, max qubit = 4, max clusters searched = 6, 1e5 shots')
plt.savefig('./plots/noisy_benchmark.png')
import pickle
import matplotlib.pyplot as plt
import scipy.optimize as opt
import numpy as np
from scipy.stats import wasserstein_distance

filename = 'noisy_benchmark_10000_shots_5_qubits_max_6_clusters'
num_shots = int(filename.split('_')[2])
max_qubit = int(filename.split('_')[4])
max_clusters = int(filename.split('_')[7])
noisy_benchmark = pickle.load(open('./noisy_benchmark_data/%s.p'%filename, 'rb' ))
def func(x, a, b):
    return np.exp(a*x)+b

def cross_entropy(d1,d2):
    h = 0
    for p,q in zip(d1,d2):
        if p==0:
            h += 0
        else:
            h+= -p*np.log(q)
    return h

num_qubits,times,sv_noiseless_fc_l,qasm_noiseless_fc_l,qasm_noisy_fc_l,qasm_noisy_na_fc_l,qasm_noisy_na_cutting_l = noisy_benchmark

identical_distance = [wasserstein_distance(sv_noiseless_fc_l[i],sv_noiseless_fc_l[i]) for i in range(len(num_qubits))]
qasm_distances = [wasserstein_distance(sv_noiseless_fc_l[i],qasm_noiseless_fc_l[i]) for i in range(len(num_qubits))]
qasm_noise_distances = [wasserstein_distance(sv_noiseless_fc_l[i],qasm_noisy_fc_l[i]) for i in range(len(num_qubits))]
qasm_noise_na_distances = [wasserstein_distance(sv_noiseless_fc_l[i],qasm_noisy_na_fc_l[i]) for i in range(len(num_qubits))]
qasm_noise_na_cutting_distances = [wasserstein_distance(sv_noiseless_fc_l[i],qasm_noisy_na_cutting_l[i]) for i in range(len(num_qubits))]

identical_distr_ce = [cross_entropy(sv_noiseless_fc_l[i],sv_noiseless_fc_l[i]) for i in range(len(num_qubits))]
qasm_ce = [cross_entropy(qasm_noiseless_fc_l[i],sv_noiseless_fc_l[i]) for i in range(len(num_qubits))]
qasm_noise_ce = [cross_entropy(qasm_noisy_fc_l[i],sv_noiseless_fc_l[i]) for i in range(len(num_qubits))]
qasm_noise_na_ce = [cross_entropy(qasm_noisy_na_fc_l[i],sv_noiseless_fc_l[i]) for i in range(len(num_qubits))]
qasm_noise_na_cutting_ce = [cross_entropy(qasm_noisy_na_cutting_l[i],sv_noiseless_fc_l[i]) for i in range(len(num_qubits))]

plt.figure(figsize=(10,10))
plt.subplot(231)
plt.plot(num_qubits,times['searcher'],'^',label='cut searcher')
# optimizedParameters, pcov = opt.curve_fit(func, np.array(num_qubits), np.array(times['searcher']))
# plt.plot(num_qubits, func(np.array(num_qubits), *optimizedParameters), label="fit")
plt.xlabel('supremacy circuit # qubits')
plt.ylabel('runtime (s)')
plt.legend()
plt.subplot(232)
plt.plot(num_qubits,times['evaluator'],'^',label='cluster evaluator')
# optimizedParameters, pcov = opt.curve_fit(func, np.array(num_qubits), np.array(times['evaluator']))
# plt.plot(num_qubits, func(np.array(num_qubits), *optimizedParameters), label="fit")
plt.xlabel('supremacy circuit # qubits')
plt.legend()
plt.subplot(233)
plt.plot(num_qubits,times['uniter'],'^',label='uniter')
# optimizedParameters, pcov = opt.curve_fit(func, np.array(num_qubits), np.array(times['uniter']))
# plt.plot(num_qubits, func(np.array(num_qubits), *optimizedParameters), label="fit")
plt.xlabel('supremacy circuit # qubits')
plt.ylabel('runtime (s)')
plt.legend()
plt.subplot(234)
plt.plot(num_qubits,identical_distance,'o',label='Identical Distributions')
plt.plot(num_qubits,qasm_distances,'o',label='qasm')
plt.plot(num_qubits,qasm_noise_distances,'o',label='qasm+noise')
plt.plot(num_qubits,qasm_noise_na_distances,'o',label='qasm+noise+na')
plt.plot(num_qubits,qasm_noise_na_cutting_distances,'X',label='qasm+noise+na+cutting')
# plt.yscale('log')
plt.xlabel('supremacy circuit # qubits')
plt.ylabel('wasserstein_distance')
plt.legend()
plt.subplot(235)
plt.plot(num_qubits,identical_distr_ce,'o',label='Identical Distributions')
plt.plot(num_qubits,qasm_ce,'o',label='qasm')
plt.plot(num_qubits,qasm_noise_ce,'o',label='qasm+noise')
plt.plot(num_qubits,qasm_noise_na_ce,'o',label='qasm+noise+na')
plt.plot(num_qubits,qasm_noise_na_cutting_ce,'X',label='qasm+noise+na+cutting')
# plt.yscale('log')
plt.xlabel('supremacy circuit # qubits')
plt.ylabel('cross entropy')
plt.legend()
plt.suptitle('Noisy Circuit Cutting Benchmark, max qubit = %d, max clusters = %d, %.0e shots'%(max_qubit,max_clusters,num_shots))
plt.savefig('./plots/%s.png'%filename)
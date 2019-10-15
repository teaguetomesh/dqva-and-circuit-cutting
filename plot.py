import pickle
import glob
import matplotlib.pyplot as plt
import scipy.optimize as opt
import numpy as np
from scipy.stats import chisquare
from scipy.stats import wasserstein_distance

def func(x, a, b):
    return np.exp(a*x)+b

def cross_entropy(target,obs):
    assert len(target)==len(obs)
    alpha = 1e-4
    if 0 in obs:
        obs = [(x+alpha)/(1+alpha*len(obs)) for x in obs]
    assert abs(sum(obs)-1)<1e-3
    h = 0
    for p,q in zip(target,obs):
        if p==0:
            h += 0
        else:
            assert q>=0
            h += -p*np.log(q)
    return h

ce_metric = {}
for entry in ['sv_noiseless','qasm','qasm+noise','qasm+noise+na','qasm+noise+na+cutting','reduction']:
    ce_metric[entry] = []
times = {'searcher':[],'classical_evaluator':[],'quantum_evaluator':[],'uniter':[]}
num_qubits = []

filename = './noisy_benchmark_data/plotter_input_6_qubits_6_clusters_10000_shots.p'
benchmark_l = pickle.load(open(filename, 'rb' ))
filename = filename.split('/')[-1].split('.')[0]
max_qubit = int(filename.split('_')[2])
max_clusters = int(filename.split('_')[4])
num_shots = int(filename.split('_')[6])

for noisy_benchmark in benchmark_l:
    circ, evaluations, searcher_time, classical_time, quantum_time, uniter_time = noisy_benchmark
    times['searcher'].append(searcher_time)
    times['classical_evaluator'].append(classical_time)
    times['quantum_evaluator'].append(quantum_time)
    times['uniter'].append(uniter_time)

    qubit_count = len(circ.qubits)

    print(max_qubit,max_clusters,num_shots,qubit_count)

    target = evaluations['sv_noiseless']
    for evaluation_method in ['sv_noiseless','qasm','qasm+noise','qasm+noise+na','qasm+noise+na+cutting']:
        ce = cross_entropy(target=target,obs=evaluations[evaluation_method])
        ce_metric[evaluation_method].append(ce)

    num_qubits.append(qubit_count)
    ce_reduction = 100*(ce_metric['qasm+noise+na'][-1]-ce_metric['qasm+noise+na+cutting'][-1])/(ce_metric['qasm+noise+na'][-1]-ce_metric['sv_noiseless'][-1])
    ce_metric['reduction'].append(ce_reduction)

hybrid = sum(times['classical_evaluator'])>0

plt.figure(figsize=(15,10))
plt.subplot(231)
plt.plot(num_qubits,times['searcher'],'^',label='cut searcher')
# optimizedParameters, pcov = opt.curve_fit(func, np.array(num_qubits), np.array(times['searcher']))
# plt.plot(num_qubits, func(np.array(num_qubits), *optimizedParameters), label="fit")
plt.xlabel('supremacy circuit # qubits')
plt.ylabel('runtime (s)')
plt.legend()
plt.subplot(232)
plt.plot(num_qubits,times['quantum_evaluator'],'^',label='quantum evaluator')
plt.plot(num_qubits,times['classical_evaluator'],'^',label='classical evaluator')
# optimizedParameters, pcov = opt.curve_fit(func, np.array(num_qubits), np.array(times['evaluator']))
# plt.plot(num_qubits, func(np.array(num_qubits), *optimizedParameters), label="fit")
plt.xlabel('supremacy circuit # qubits')
plt.legend()
plt.subplot(233)
plt.plot(num_qubits,times['uniter'],'^',label='uniter')
# optimizedParameters, pcov = opt.curve_fit(func, np.array(num_qubits), np.array(times['uniter']))
# plt.plot(num_qubits, func(np.array(num_qubits), *optimizedParameters), label="fit")
plt.xlabel('supremacy circuit # qubits')
plt.legend()
plt.subplot(234)
plt.plot(num_qubits,ce_metric['sv_noiseless'],'*',label='Identical Distributions',markersize=12)
plt.plot(num_qubits,ce_metric['qasm'],'o',label='qasm')
plt.plot(num_qubits,ce_metric['qasm+noise'],'o',label='qasm+noise')
plt.plot(num_qubits,ce_metric['qasm+noise+na'],'o',label='qasm+noise+na')
plt.plot(num_qubits,ce_metric['qasm+noise+na+cutting'],'X',label='qasm+noise+na+cutting')
# plt.yscale('log')
plt.xlabel('supremacy circuit # qubits')
plt.ylabel('Cross Entropy')
plt.legend()
plt.subplot(235)
# plt.plot(num_qubits,chi_square_improvement,'*',label='chi^2')
axes = plt.gca()
axes.set_ylim([0,100])
plt.plot(num_qubits,ce_metric['reduction'],'*',label='cross entropy')
plt.xlabel('supremacy circuit # qubits')
plt.ylabel('% Reduction')
plt.legend()
if hybrid:
    plt.suptitle('Hybrid Benchmark, max qubit = %d, max clusters = %d, %.0e shots'%(max_qubit,max_clusters,num_shots))
    plt.savefig('./plots/hybrid_%d_qubit_%d_clusters_%d_shots.png'%(max_qubit,max_clusters,num_shots))
else:
    plt.suptitle('Quantum Benchmark, max qubit = %d, max clusters = %d, %.0e shots'%(max_qubit,max_clusters,num_shots))
    plt.savefig('./plots/quantum_%d_qubit_%d_clusters_%d_shots.png'%(max_qubit,max_clusters,num_shots))
import pickle
import glob
import matplotlib.pyplot as plt
import scipy.optimize as opt
import numpy as np
from scipy.stats import chisquare

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

for filename in glob.glob("./noisy_benchmark_data/*.p"):
    filename = filename.split('/')[-1].split('.')[0]
    print(filename)

    num_shots = int(filename.split('_')[2])
    max_qubit = int(filename.split('_')[4])
    max_clusters = int(filename.split('_')[7])
    noisy_benchmark = pickle.load(open('./noisy_benchmark_data/%s.p'%filename, 'rb' ))

    num_qubits,times,sv_noiseless_fc_l,qasm_noiseless_fc_l,qasm_noisy_fc_l,qasm_noisy_na_fc_l,qasm_noisy_na_cutting_l = noisy_benchmark

    print(num_qubits)

    identical_ce = [cross_entropy(sv_noiseless_fc_l[i],sv_noiseless_fc_l[i]) for i in range(len(num_qubits))]
    qasm_ce = [cross_entropy(sv_noiseless_fc_l[i],qasm_noiseless_fc_l[i]) for i in range(len(num_qubits))]
    qasm_noise_ce = [cross_entropy(sv_noiseless_fc_l[i],qasm_noisy_fc_l[i]) for i in range(len(num_qubits))]
    qasm_noise_na_ce = [cross_entropy(sv_noiseless_fc_l[i],qasm_noisy_na_fc_l[i]) for i in range(len(num_qubits))]
    qasm_noise_na_cutting_ce = [cross_entropy(sv_noiseless_fc_l[i],qasm_noisy_na_cutting_l[i]) for i in range(len(num_qubits))]
    ce_improvement = [(qasm_noise_na_ce[i]-qasm_noise_na_cutting_ce[i])/(qasm_noise_na_ce[i]-identical_ce[i]) for i in range(len(num_qubits))]

    identical_distr_chi = [chisquare(sv_noiseless_fc_l[i],sv_noiseless_fc_l[i]).statistic for i in range(len(num_qubits))]
    qasm_chi = [chisquare(qasm_noiseless_fc_l[i],sv_noiseless_fc_l[i]).statistic for i in range(len(num_qubits))]
    qasm_noise_chi = [chisquare(qasm_noisy_fc_l[i],sv_noiseless_fc_l[i]).statistic for i in range(len(num_qubits))]
    qasm_noise_na_chi = [chisquare(qasm_noisy_na_fc_l[i],sv_noiseless_fc_l[i]).statistic for i in range(len(num_qubits))]
    qasm_noise_na_cutting_chi = [chisquare(qasm_noisy_na_cutting_l[i],sv_noiseless_fc_l[i]).statistic for i in range(len(num_qubits))]
    chi_square_improvement = [(qasm_noise_na_chi[i]-qasm_noise_na_cutting_chi[i])/(qasm_noise_na_chi[i]-identical_distr_chi[i]) for i in range(len(num_qubits))]

    plt.figure(figsize=(15,10))
    plt.subplot(231)
    plt.plot(num_qubits,times['searcher'],'^',label='cut searcher')
    optimizedParameters, pcov = opt.curve_fit(func, np.array(num_qubits), np.array(times['searcher']))
    plt.plot(num_qubits, func(np.array(num_qubits), *optimizedParameters), label="fit")
    plt.xlabel('supremacy circuit # qubits')
    plt.ylabel('runtime (s)')
    plt.legend()
    plt.subplot(232)
    plt.plot(num_qubits,times['evaluator'],'^',label='cluster evaluator')
    optimizedParameters, pcov = opt.curve_fit(func, np.array(num_qubits), np.array(times['evaluator']))
    plt.plot(num_qubits, func(np.array(num_qubits), *optimizedParameters), label="fit")
    plt.xlabel('supremacy circuit # qubits')
    plt.legend()
    plt.subplot(233)
    plt.plot(num_qubits,times['uniter'],'^',label='uniter')
    optimizedParameters, pcov = opt.curve_fit(func, np.array(num_qubits), np.array(times['uniter']))
    plt.plot(num_qubits, func(np.array(num_qubits), *optimizedParameters), label="fit")
    plt.xlabel('supremacy circuit # qubits')
    plt.legend()
    plt.subplot(234)
    plt.plot(num_qubits,identical_ce,'*',label='Identical Distributions')
    plt.plot(num_qubits,qasm_ce,'o',label='qasm')
    plt.plot(num_qubits,qasm_noise_ce,'o',label='qasm+noise')
    plt.plot(num_qubits,qasm_noise_na_ce,'o',label='qasm+noise+na')
    plt.plot(num_qubits,qasm_noise_na_cutting_ce,'X',label='qasm+noise+na+cutting')
    # plt.yscale('log')
    plt.xlabel('supremacy circuit # qubits')
    plt.ylabel('Cross Entropy')
    plt.legend()
    plt.subplot(235)
    plt.plot(num_qubits,identical_distr_chi,'*',label='Identical Distributions')
    plt.plot(num_qubits,qasm_chi,'o',label='qasm')
    plt.plot(num_qubits,qasm_noise_chi,'o',label='qasm+noise')
    plt.plot(num_qubits,qasm_noise_na_chi,'o',label='qasm+noise+na')
    plt.plot(num_qubits,qasm_noise_na_cutting_chi,'X',label='qasm+noise+na+cutting')
    # plt.yscale('log')
    plt.xlabel('supremacy circuit # qubits')
    plt.ylabel('chi^2')
    plt.legend()
    plt.subplot(236)
    # plt.plot(num_qubits,chi_square_improvement,'*',label='chi^2')
    plt.plot(num_qubits,ce_improvement,'*',label='cross entropy')
    plt.xlabel('supremacy circuit # qubits')
    plt.ylabel('% Improvement')
    plt.legend()
    plt.suptitle('Noisy Circuit Cutting Benchmark, max qubit = %d, max clusters = %d, %.0e shots'%(max_qubit,max_clusters,num_shots))
    plt.savefig('./plots/%s.png'%filename)
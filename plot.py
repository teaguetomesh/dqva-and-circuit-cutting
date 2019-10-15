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
    alpha = 1e-14
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

all_files = glob.glob('./noisy_benchmark_data/plotter_input_*.p')
for filename in all_files:
    f = open(filename, 'rb' )
    benchmarks = []
    while 1:
        try:
            benchmarks.append(pickle.load(f))
        except EOFError:
            break
    filename = filename.split('/')[-1].split('.')[0]
    print('plotting',filename)
    max_qubit = int(filename.split('_')[2])
    max_clusters = int(filename.split('_')[4])
    num_shots = int(filename.split('_')[6])
    repetitions = len(benchmarks)
    num_pts = len(benchmarks[0])

    ce_metric_l = []
    times_l = []
    for i, benchmark in enumerate(benchmarks):
        print('repetition %d'%i)
        ce_metric = {}
        for entry in ['sv_noiseless','qasm','qasm+noise','qasm+noise+cutting','reduction']:
            ce_metric[entry] = []
        times = {'searcher':[],'classical_evaluator':[],'quantum_evaluator':[],'uniter':[]}
        num_qubits = []
        for noisy_benchmark in benchmark:
            circ, evaluations, searcher_time, classical_time, quantum_time, uniter_time = noisy_benchmark
            times['searcher'].append(searcher_time)
            times['classical_evaluator'].append(classical_time)
            times['quantum_evaluator'].append(quantum_time)
            times['uniter'].append(uniter_time)

            qubit_count = len(circ.qubits)

            target = evaluations['sv_noiseless']
            for evaluation_method in ['sv_noiseless','qasm','qasm+noise','qasm+noise+cutting']:
                ce = cross_entropy(target=target,obs=evaluations[evaluation_method])
                ce_metric[evaluation_method].append(ce)

            num_qubits.append(qubit_count)
            ce_reduction = 100*(ce_metric['qasm+noise'][-1]-ce_metric['qasm+noise+cutting'][-1])/(ce_metric['qasm+noise'][-1]-ce_metric['sv_noiseless'][-1])
            ce_metric['reduction'].append(ce_reduction)
            # print(ce_reduction)
        print(ce_metric)
        # print(times)
        ce_metric_l.append(ce_metric)
        times_l.append(times)
        print('-'*50)

    ce_metric_avg = {}
    ce_metric_err = {}
    for evaluation_method in ce_metric_l[0]:
        ce_metric_avg[evaluation_method] = [0 for i in range(num_pts)]
        ce_metric_err[evaluation_method] = [0 for i in range(num_pts)]

    times_avg = {}
    times_err = {}
    for component in times_l[0]:
        times_avg[component] = [0 for i in range(num_pts)]
        times_err[component] = [0 for i in range(num_pts)]

    for evaluation_method in ce_metric_avg:
        print(evaluation_method)
        for i in range(num_pts):
            ce_list = []
            for ce_metric in ce_metric_l:
                ce_list.append(ce_metric[evaluation_method][i])
            avg = np.mean(ce_list)
            std = np.std(ce_list)
            print('%dth term:'%i,ce_list,avg,std)
            ce_metric_avg[evaluation_method][i] = avg
            ce_metric_err[evaluation_method][i] = std
        print('-'*100)

    print('ce avg:', ce_metric_avg)
    print('ce err:', ce_metric_err)

    for component in times_avg:
        # print(component)
        for i in range(num_pts):
            time_list = []
            for times in times_l:
                time_list.append(times[component][i])
            # print('term %d'%i,time_list)
            avg = np.mean(time_list)
            std = np.std(time_list)
            times_avg[component][i] = avg
            times_err[component][i] = std
    # print(times_avg)
    # print(times_err)

    quantum = sum(times_avg['quantum_evaluator'])>0
    classical = sum(times_avg['classical_evaluator'])>0

    plt.figure(figsize=(15,10))
    plt.subplot(231)
    plt.plot(num_qubits,times_avg['searcher'],'^',label='cut searcher')
    # optimizedParameters, pcov = opt.curve_fit(func, np.array(num_qubits), np.array(times['searcher']))
    # plt.plot(num_qubits, func(np.array(num_qubits), *optimizedParameters), label="fit")
    plt.xlabel('supremacy circuit # qubits')
    plt.ylabel('runtime (s)')
    plt.legend()
    plt.subplot(232)
    plt.plot(num_qubits,times_avg['quantum_evaluator'],'^',label='quantum evaluator')
    plt.plot(num_qubits,times_avg['classical_evaluator'],'^',label='classical evaluator')
    # optimizedParameters, pcov = opt.curve_fit(func, np.array(num_qubits), np.array(times['quantum_evaluator']))
    # plt.plot(num_qubits, func(np.array(num_qubits), *optimizedParameters), label="quantum_fit")
    # optimizedParameters, pcov = opt.curve_fit(func, np.array(num_qubits), np.array(times['classical_evaluator']))
    # plt.plot(num_qubits, func(np.array(num_qubits), *optimizedParameters), label="classical_fit")
    plt.xlabel('supremacy circuit # qubits')
    plt.legend()
    plt.subplot(233)
    plt.plot(num_qubits,times_avg['uniter'],'^',label='uniter')
    # optimizedParameters, pcov = opt.curve_fit(func, np.array(num_qubits), np.array(times['uniter']))
    # plt.plot(num_qubits, func(np.array(num_qubits), *optimizedParameters), label="fit")
    plt.xlabel('supremacy circuit # qubits')
    plt.legend()
    plt.subplot(234)
    plt.errorbar(num_qubits,ce_metric_avg['sv_noiseless'],fmt='*',yerr=ce_metric_err['sv_noiseless'],label='Identical Distributions',markersize=12)
    plt.errorbar(num_qubits,ce_metric_avg['qasm'],fmt='o',yerr=ce_metric_err['qasm'],label='qasm')
    plt.errorbar(num_qubits,ce_metric_avg['qasm+noise'],fmt='o',yerr=ce_metric_err['qasm+noise'],label='qasm+noise')
    # plt.errorbar(num_qubits,ce_metric_avg['qasm+noise+na'],fmt='o',yerr=ce_metric_err['qasm+noise+na'],label='qasm+noise+na')
    plt.errorbar(num_qubits,ce_metric_avg['qasm+noise+cutting'],fmt='X',yerr=ce_metric_err['qasm+noise+cutting'],label='qasm+noise+cutting')
    # plt.yscale('log')
    plt.xlabel('supremacy circuit # qubits')
    plt.ylabel('Cross Entropy')
    plt.legend()
    plt.subplot(235)
    axes = plt.gca()
    axes.set_ylim([0,100])
    plt.errorbar(num_qubits,ce_metric_avg['reduction'],fmt='*',yerr=ce_metric_err['reduction'],label='cross entropy')
    plt.xlabel('supremacy circuit # qubits')
    plt.ylabel('% Reduction')
    plt.legend()

    if quantum and classical:
        plt.suptitle('Hybrid Benchmark, max qubit = %d, max clusters = %d, %.0e shots'%(max_qubit,max_clusters,num_shots))
        plt.savefig('./plots/hybrid_%d_qubit_%d_clusters_%d_shots.png'%(max_qubit,max_clusters,num_shots))
    elif quantum and not classical:
        plt.suptitle('Quantum Benchmark, max qubit = %d, max clusters = %d, %.0e shots'%(max_qubit,max_clusters,num_shots))
        plt.savefig('./plots/quantum_%d_qubit_%d_clusters_%d_shots.png'%(max_qubit,max_clusters,num_shots))
    elif classical and not quantum:
        plt.suptitle('Classical Benchmark, max qubit = %d, max clusters = %d, %.0e shots'%(max_qubit,max_clusters,num_shots))
        plt.savefig('./plots/classical_%d_qubit_%d_clusters_%d_shots.png'%(max_qubit,max_clusters,num_shots))
    else:
        raise Exception('evaluator time was not recorded properly')
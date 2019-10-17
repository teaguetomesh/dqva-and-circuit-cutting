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
    obs = [x if x>=0 else 0 for x in obs]
    # print(sum(obs))
    # assert abs(sum(obs)-1)<1e-2
    alpha = 1e-16
    if 0 in obs:
        obs = [(x+alpha)/(1+alpha*len(obs)) for x in obs]
        # print('scaled:', sum(obs))
    # assert abs(sum(obs)-1)<1e-2
    h = 0
    for p,q in zip(target,obs):
        if p==0:
            h += 0
        else:
            assert q>=0
            h += -p*np.log(q)
    return h

if __name__ == '__main__':
    all_files = glob.glob('./benchmark_data/*_plotter_input_*.p')
    for filename in all_files:
        f = open(filename, 'rb' )
        benchmarks = []
        while 1:
            try:
                benchmarks.append(pickle.load(f))
            except EOFError:
                break
        evaluator_type = filename.split('/')[-1].split('_')[0]
        figname = './plots/'+filename.split('/')[-1].replace('_plotter_input','').replace('.p','.png')
        saturated = 'saturated' in figname
        print('plotting',figname)
        filename = filename.split('/')[-1].split('.p')[0]
        max_qubit = int(filename.split('_')[3])
        max_clusters = int(filename.split('_')[5])
        repetitions = len(benchmarks)
        num_pts = len(benchmarks[0])

        ce_metric_l = []
        times_l = []
        for i, benchmark in enumerate(benchmarks):
            num_shots_l = []
            # print('repetition %d'%i)
            ce_metric = {}
            for entry in ['sv_noiseless','qasm','qasm+noise','qasm+noise+cutting','reduction']:
                ce_metric[entry] = []
            times = {'searcher':[],'classical_evaluator':[],'quantum_evaluator':[],'uniter':[]}
            num_qubits = []
            for data in benchmark:
                num_shots,searcher_time,circ,evaluations,classical_time,quantum_time,uniter_time = data
                num_shots_l.append(num_shots)
                times['searcher'].append(searcher_time)
                times['classical_evaluator'].append(classical_time)
                times['quantum_evaluator'].append(quantum_time)
                times['uniter'].append(uniter_time)

                qubit_count = len(circ.qubits)

                target = evaluations['sv_noiseless']
                # print(len(circ.qubits))
                for evaluation_method in ['sv_noiseless','qasm','qasm+noise','qasm+noise+cutting']:
                    # print(evaluation_method)
                    ce = cross_entropy(target=target,obs=evaluations[evaluation_method])
                    ce_metric[evaluation_method].append(ce)

                num_qubits.append(qubit_count)
                ce_reduction = 100*(ce_metric['qasm+noise'][-1]-ce_metric['qasm+noise+cutting'][-1])/(ce_metric['qasm+noise'][-1]-ce_metric['sv_noiseless'][-1])
                ce_metric['reduction'].append(ce_reduction)
                # print(ce_reduction)
            # print(ce_metric)
            # print(times)
            ce_metric_l.append(ce_metric)
            times_l.append(times)
            # print('-'*50)

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
            # print(evaluation_method)
            for i in range(num_pts):
                ce_list = []
                for ce_metric in ce_metric_l:
                    ce_list.append(ce_metric[evaluation_method][i])
                avg = np.mean(ce_list)
                std = np.std(ce_list)
                # print('%dth term:'%i,ce_list,avg,std)
                ce_metric_avg[evaluation_method][i] = avg
                ce_metric_err[evaluation_method][i] = std
            # print('-'*100)

        # print('ce avg:', ce_metric_avg)
        # print('ce err:', ce_metric_err)

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

        figsize_scale = 4.5
        plt.figure(figsize=(3*figsize_scale,2*figsize_scale))
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
        if evaluator_type == 'classical':
            plt.errorbar(num_qubits,ce_metric_avg['qasm+noise+cutting'],fmt='X',yerr=ce_metric_err['qasm+noise+cutting'],label='qasm+cutting')
        else:
            plt.errorbar(num_qubits,ce_metric_avg['qasm+noise+cutting'],fmt='X',yerr=ce_metric_err['qasm+noise+cutting'],label='qasm+noise+cutting')
        # plt.yscale('log')
        plt.xlabel('supremacy circuit # qubits')
        plt.ylabel('Cross Entropy')
        plt.legend()
        plt.subplot(235)
        width = 0.35
        plt.bar(x=np.array(num_qubits),
        height=[x-y for x,y in zip(ce_metric_avg['qasm+noise'],ce_metric_avg['sv_noiseless'])],
        yerr=[np.sqrt(x*x+y*y) for x,y in zip(ce_metric_err['qasm+noise'],ce_metric_err['sv_noiseless'])],
        width=width, color='r',label='qasm+noise')
        plt.bar(x=np.array(num_qubits)+width,
        height=[x-y for x,y in zip(ce_metric_avg['qasm+noise+cutting'],ce_metric_avg['sv_noiseless'])],
        yerr=[np.sqrt(x*x+y*y) for x,y in zip(ce_metric_err['qasm+noise+cutting'],ce_metric_err['sv_noiseless'])],
        width=width, color='b',label='qasm+noise+cutting')
        plt.xticks(np.array(num_qubits) + width / 2,np.array(num_qubits))
        plt.xlabel('supremacy circuit # qubits')
        plt.ylabel('Cross Entropy Loss')
        plt.legend()
        plt.subplot(236)
        axes = plt.gca()
        axes.set_ylim([0,100])
        plt.errorbar(num_qubits,ce_metric_avg['reduction'],fmt='o',yerr=ce_metric_err['reduction'],label='cross entropy')
        plt.xlabel('supremacy circuit # qubits')
        plt.ylabel('% Reduction')
        plt.legend()

        if saturated:
            plt.suptitle('%s Benchmark, max qubit = %d, max clusters = %d, max %.2e fc shots, saturated'%(evaluator_type,max_qubit,max_clusters,max(num_shots_l)))
        else:
            plt.suptitle('%s Benchmark, max qubit = %d, max clusters = %d, max %.2e fc shots, same total'%(evaluator_type,max_qubit,max_clusters,max(num_shots_l)))
        plt.savefig('%s'%figname)
import pickle
import glob
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from utils.helper_fun import cross_entropy, fidelity, get_filename, read_file
import os
from scipy import optimize

def exp_func(x, a, b):
    return a * np.exp(b*x)

if __name__ == '__main__':
    file_names = glob.glob('./large_on_small/benchmark_data/*/*_plotter_input*.p')
    for file_name in file_names:
        dirname = get_filename(experiment_name='large_on_small',circuit_type='bv',device_name=None,field='plotter_output',evaluation_method=None,shots_mode=None)
        plotter_input = read_file(file_name)
        print(file_name)
        fids = []
        fc_sizes = []
        reconstruction_times = []
        for case in plotter_input:
            reconstruction_time = plotter_input[case]['uniter_time']
            fid = plotter_input[case]['cutting'][-1]
            fc_sizes.append(case[1])
            fids.append(fid)
            reconstruction_times.append(reconstruction_time)
        fc_sizes, fids, reconstruction_times = zip(*sorted(zip(fc_sizes, fids, reconstruction_times)))
        fc_sizes = list(fc_sizes)
        fids = list(fids)
        reconstruction_times = list(reconstruction_times)
        
        print(fc_sizes)
        print(fids)
        print(reconstruction_times)

        fid_params, fid_params_covariance = optimize.curve_fit(exp_func, fc_sizes, fids,p0=[1, 1])
        time_params, time_params_covariance = optimize.curve_fit(exp_func, fc_sizes, reconstruction_times,p0=[1, 1])

        evaluation_mode = file_name.split('/')[-1].split('_')[0]

        plt.figure(figsize=(10,5))
        plt.subplot(121)
        plt.xlabel('Number of qubits',size=12)
        plt.plot(fc_sizes, fids, 'r*',label='cutting mode')
        plt.ylabel('Fidelity, higher is better',size=12)
        # plt.axvline(x=20, color='r', linestyle='--',label='Max Device Size')
        xvals = np.arange(min(fc_sizes),max(fc_sizes),0.1)
        plt.plot(xvals, exp_func(np.array(xvals), fid_params[0], fid_params[1]),'b',label='Fidelity Fitted Exp Function')
        plt.xticks([x for x in fc_sizes])
        plt.legend()
        plt.subplot(122)
        plt.plot(fc_sizes, reconstruction_times, 'r*')
        plt.xlabel('Number of qubits',size=12)
        plt.ylabel('Reconstruction time, lower is better (s)',size=12)
        plt.xticks([x for x in fc_sizes])
        xvals = np.arange(min(fc_sizes),max(fc_sizes),0.1)
        plt.plot(xvals, exp_func(np.array(xvals), time_params[0], time_params[1]),'r',label='Classical Overhead Fitted Exp Function')
        plt.legend()
        plt.tight_layout()
        plt.savefig('%s/large_on_small.png'%dirname,dpi=400)
        plt.close()
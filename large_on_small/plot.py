import pickle
import glob
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from helper_fun import cross_entropy, fidelity
import os
from scipy import optimize

def exp_func(x, a, b):
    return a * np.exp(b*x)

if __name__ == '__main__':
    file_names = glob.glob('./benchmark_data/*/*_plotter_input*.p')
    for file_name in file_names:
        print(file_name)
        f = open(file_name,'rb')
        plotter_input = {}
        while 1:
            try:
                plotter_input.update(pickle.load(f))
            except (EOFError):
                break
        f.close()
        fids = []
        fc_sizes = []
        reconstruction_times = []
        for case in plotter_input:
            reconstruction_time = plotter_input[case]['uniter_time']
            fid = plotter_input[case]['cutting_fid']
            fc_sizes.append(case[1])
            fids.append(fid)
            reconstruction_times.append(reconstruction_time)
        fc_sizes, fids, reconstruction_times = zip(*sorted(zip(fc_sizes, fids, reconstruction_times)))
        fc_sizes = list(fc_sizes)
        fids = list(fids)
        reconstruction_times = list(reconstruction_times)
        
        cut_off_idx = fc_sizes.index(20)+1
        fc_sizes_cut = fc_sizes[cut_off_idx:]
        fids_cut = fids[cut_off_idx:]
        reconstruction_times_cut = reconstruction_times[cut_off_idx:]
        print(fc_sizes_cut)
        print(fids_cut)
        print(reconstruction_times_cut)

        params, params_covariance = optimize.curve_fit(exp_func, fc_sizes_cut, reconstruction_times_cut,p0=[1, 1])
        params_2, params_covariance_2 = optimize.curve_fit(exp_func, fc_sizes[:cut_off_idx], fids[:cut_off_idx], p0=[1, 1])

        evaluation_mode = file_name.split('/')[-1].split('_')[0]

        plt.figure(figsize=(10,5))
        plt.subplot(121)
        plt.xlabel('Number of qubits',size=12)
        plt.plot(fc_sizes[:cut_off_idx], fids[:cut_off_idx], 'bX',label='standard mode')
        plt.plot(fc_sizes[cut_off_idx:], fids[cut_off_idx:], 'r*',label='cutting mode')
        plt.ylabel('Fidelity, higher is better',size=12)
        # plt.axvline(x=20, color='r', linestyle='--',label='Max Device Size')
        xvals = np.arange(min(fc_sizes),max(fc_sizes),0.1)
        plt.plot(xvals, exp_func(np.array(xvals), params_2[0], params_2[1]),'b',label='Fidelity Fitted Exp Function')
        plt.xticks([x for x in fc_sizes])
        plt.legend()
        plt.subplot(122)
        plt.plot(fc_sizes_cut, reconstruction_times_cut, 'r*')
        plt.xlabel('Number of qubits',size=12)
        plt.ylabel('Reconstruction time, lower is better (s)',size=12)
        plt.xticks([x for x in fc_sizes_cut])
        xvals = np.arange(21,30,0.1)
        plt.plot(xvals, exp_func(np.array(xvals), params[0], params[1]),'r',label='Classical Overhead Fitted Exp Function')
        plt.legend()
        plt.tight_layout()
        plt.savefig('large_on_small.png',dpi=400)
        plt.close()
import pickle
import glob
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from helper_fun import cross_entropy, fidelity
import os

if __name__ == '__main__':
    file_names = glob.glob('./benchmark_data/*/*_plotter_input*.p')
    for file_name in file_names:
        print(file_name)
        f = open(file_name,'rb')
        plotter_input = pickle.load(f)
        fids = []
        fc_sizes = []
        reconstruction_times = []
        for case in plotter_input:
            reconstruction_time = plotter_input[case]['uniter_time']
            evaluations = plotter_input[case]['evaluations']
            cutting = evaluations['cutting']
            fid = cutting[-1]
            fc_sizes.append(case[1])
            fids.append(fid)
            reconstruction_times.append(reconstruction_time)
        fc_sizes, fids, reconstruction_times = zip(*sorted(zip(fc_sizes, fids, reconstruction_times)))
        print(fc_sizes)
        print(fids)
        print(reconstruction_times)

        evaluation_mode = file_name.split('/')[-1].split('_')[0]

        plt.figure(figsize=(10,5))
        plt.subplot(121)
        plt.xlabel('Number of qubits')
        plt.plot(fc_sizes, fids, 'bX')
        plt.ylabel('Fidelity')
        plt.xticks([x for x in fc_sizes])
        plt.subplot(122)
        plt.plot(fc_sizes, reconstruction_times, 'r*')
        plt.xlabel('Number of qubits')
        plt.ylabel('Reconstruction time (s)')
        plt.xticks([x for x in fc_sizes])
        plt.tight_layout()
        plt.savefig('%s.png'%evaluation_mode,dpi=400)
        plt.close()
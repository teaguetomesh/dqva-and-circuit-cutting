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
        fc_size = []
        for case in plotter_input:
            reconstruction_time = plotter_input[case]['uniter_time']
            evaluations = plotter_input[case]['evaluations']
            cutting = evaluations['cutting']
            fid = cutting[-1]
            fc_size.append(case[1])
            fids.append(fid)
        fc_size, fids = zip(*sorted(zip(fc_size, fids)))
        print(fc_size)
        print(fids)

        evaluation_mode = file_name.split('/')[-1].split('_')[0]
        plt.figure()
        plt.plot(fc_size,fids,'*',label='BV fidelity')
        plt.legend()
        plt.title('%s'%evaluation_mode)
        plt.savefig('%s.png'%evaluation_mode)
        plt.close()
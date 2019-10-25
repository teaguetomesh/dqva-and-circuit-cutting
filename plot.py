import pickle
import glob
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
from helper_fun import cross_entropy

def func(x, a, b):
    return np.exp(a*x)+b

if __name__ == '__main__':
    all_files = glob.glob('./benchmark_data/*_plotter_input_*.p')
    for filename in all_files:
        f = open(filename, 'rb' )
        plotter_input = pickle.load(f)
        evaluator_type = filename.split('/')[-1].split('_')[0]
        figname = './plots/'+filename.split('/')[-1].replace('_plotter_input','')
        print('plotting',figname)

        [print(case, plotter_input[case]['searcher_time']) for case in plotter_input]

        hw_qubits = [case[0] for case in plotter_input]
        fc_qubits = [case[1] for case in plotter_input]
        dx = [0.2 for x in plotter_input]
        dy = [0.2 for x in plotter_input]
        searcher_time = [plotter_input[case]['searcher_time'] for case in plotter_input]
        classical_time = [plotter_input[case]['classical_time'] for case in plotter_input]
        quantum_time = [plotter_input[case]['quantum_time'] for case in plotter_input]
        uniter_time = [plotter_input[case]['uniter_time'] for case in plotter_input]
        identical_distributions = [
            cross_entropy(target=plotter_input[case]['evaluations']['sv_noiseless'],
            obs= plotter_input[case]['evaluations']['sv_noiseless'])for case in plotter_input]
        qasm = [
            cross_entropy(target=plotter_input[case]['evaluations']['sv_noiseless'],
            obs= plotter_input[case]['evaluations']['qasm'])for case in plotter_input]
        qasm_noise = [
            cross_entropy(target=plotter_input[case]['evaluations']['sv_noiseless'],
            obs= plotter_input[case]['evaluations']['qasm+noise'])for case in plotter_input]
        qasm_noise_cutting = [
            cross_entropy(target=plotter_input[case]['evaluations']['sv_noiseless'],
            obs= plotter_input[case]['evaluations']['qasm+noise+cutting'])for case in plotter_input]
        percent_change = [100*(qasm_noise[i] - qasm_noise_cutting[i])/(qasm_noise[i] - identical_distributions[i]) for i in range(len(plotter_input))]

        fig_scale = 4.5
        fig = plt.figure(figsize=(3*fig_scale,2*fig_scale))
        ax1 = fig.add_subplot(231, projection='3d')
        ax1.bar3d(hw_qubits, fc_qubits, np.zeros(len(plotter_input)), dx, dy, searcher_time)
        ax1.set_xlabel('hardware qubits')
        ax1.set_ylabel('full circuit qubits')
        ax1.set_zlabel('searcher time (seconds)')
        ax1 = fig.add_subplot(232, projection='3d')
        ax1.bar3d(hw_qubits, fc_qubits, np.zeros(len(plotter_input)), dx, dy, classical_time)
        ax1.set_zlim3d(0, 1.2*max(classical_time)+1)
        ax1.set_xlabel('hardware qubits')
        ax1.set_ylabel('full circuit qubits')
        ax1.set_zlabel('classical evaluator time (seconds)')
        ax1 = fig.add_subplot(233, projection='3d')
        ax1.bar3d(hw_qubits, fc_qubits, np.zeros(len(plotter_input)), dx, dy, quantum_time)
        ax1.set_zlim3d(0, 1.2*max(quantum_time)+1)
        ax1.set_xlabel('hardware qubits')
        ax1.set_ylabel('full circuit qubits')
        ax1.set_zlabel('quantum evaluator time (seconds)')
        ax1 = fig.add_subplot(234, projection='3d')
        ax1.bar3d(hw_qubits, fc_qubits, np.zeros(len(plotter_input)), dx, dy, uniter_time)
        ax1.set_xlabel('hardware qubits')
        ax1.set_ylabel('full circuit qubits')
        ax1.set_zlabel('reconstructor time (seconds)')
        ax1 = fig.add_subplot(235, projection='3d')
        ax1.bar3d(hw_qubits, fc_qubits, np.zeros(len(plotter_input)), dx, dy, percent_change)
        ax1.set_xlabel('hardware qubits')
        ax1.set_ylabel('full circuit qubits')
        ax1.set_zlabel('cross entropy gap reduction due to cutting (%)')
        pickle.dump(fig,open('%s'%figname, 'wb'))
        plt.savefig('%.png'%figname)
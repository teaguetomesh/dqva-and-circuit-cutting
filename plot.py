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
        plotter_inputs = []
        while 1:
            try:
                plotter_inputs.append(pickle.load(f))
            except EOFError:
                break
        evaluator_type = filename.split('/')[-1].split('_')[0]
        figname = './plots/'+filename.split('/')[-1].replace('_plotter_input','')

        # [print(case, plotter_input[case]['searcher_time']) for case in plotter_input]

        hw_qubits = [case[0] for case in plotter_inputs[0]]
        fc_qubits = [case[1] for case in plotter_inputs[0]]
        dx = [0.2 for x in plotter_inputs[0]]
        dy = [0.2 for x in plotter_inputs[0]]

        searcher_time_avg = np.array([0.0 for case in plotter_inputs[0]])
        classical_time_avg = np.array([0.0 for case in plotter_inputs[0]])
        quantum_time_avg = np.array([0.0 for case in plotter_inputs[0]])
        uniter_time_avg = np.array([0.0 for case in plotter_inputs[0]])
        ground_truth_avg = np.array([0.0 for case in plotter_inputs[0]])
        qasm_avg = np.array([0.0 for case in plotter_inputs[0]])
        qasm_noise_avg = np.array([0.0 for case in plotter_inputs[0]])
        qasm_noise_cutting_avg = np.array([0.0 for case in plotter_inputs[0]])
        percent_change_avg = np.array([0.0 for case in plotter_inputs[0]])

        for plotter_input in plotter_inputs:
            searcher_time = np.array([plotter_input[case]['searcher_time'] for case in plotter_input])
            classical_time = np.array([plotter_input[case]['classical_time'] for case in plotter_input])
            quantum_time = np.array([plotter_input[case]['quantum_time'] for case in plotter_input])
            uniter_time = np.array([plotter_input[case]['uniter_time'] for case in plotter_input])
            ground_truth = np.array([
                cross_entropy(target=plotter_input[case]['evaluations']['sv_noiseless'],
                obs= plotter_input[case]['evaluations']['sv_noiseless'])for case in plotter_input])
            qasm = np.array([
                cross_entropy(target=plotter_input[case]['evaluations']['sv_noiseless'],
                obs= plotter_input[case]['evaluations']['qasm'])for case in plotter_input])
            qasm_noise = np.array([
                cross_entropy(target=plotter_input[case]['evaluations']['sv_noiseless'],
                obs= plotter_input[case]['evaluations']['qasm+noise'])for case in plotter_input])
            qasm_noise_cutting = np.array([
                cross_entropy(target=plotter_input[case]['evaluations']['sv_noiseless'],
                obs= plotter_input[case]['evaluations']['qasm+noise+cutting'])for case in plotter_input])
            percent_change = np.array([100*(qasm_noise[i] - qasm_noise_cutting[i])/(qasm_noise[i] - ground_truth[i]) for i in range(len(plotter_input))])
            searcher_time_avg += searcher_time
            classical_time_avg += classical_time
            quantum_time_avg += quantum_time
            uniter_time_avg += uniter_time
            ground_truth_avg += ground_truth
            qasm_avg += qasm
            qasm_noise_avg += qasm_noise
            qasm_noise_cutting_avg += qasm_noise_cutting
            percent_change_avg += percent_change
        
        searcher_time_avg /= len(plotter_inputs)
        classical_time_avg /= len(plotter_inputs)
        quantum_time_avg /= len(plotter_inputs)
        uniter_time_avg /= len(plotter_inputs)
        ground_truth_avg /= len(plotter_inputs)
        qasm_avg /= len(plotter_inputs)
        qasm_noise_avg /= len(plotter_inputs)
        qasm_noise_cutting_avg /= len(plotter_inputs)
        percent_change_avg /= len(plotter_inputs)
        best_cc = {}
        for i in range(len(plotter_inputs[0])):
            percent = percent_change_avg[i]
            hw = hw_qubits[i]
            fc = fc_qubits[i]
            if (fc in best_cc and percent>best_cc[fc][1]) or (fc not in best_cc):
                best_cc[fc] = (uniter_time_avg[i],percent)
        print(best_cc)

        # Create some mock data
        fig, ax1 = plt.subplots()

        color = 'tab:blue'
        ax1.set_xlabel('Number of qubits')
        ax1.set_ylabel('Cross entropy reduction (%)', color=color)  # we already handled the x-label with ax1
        ax1.plot([fc for fc in best_cc], [best_cc[fc][1] for fc in best_cc], 'X',color=color)
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

        color = 'tab:red'
        ax2.set_ylabel('Reconstruction time (s)', color=color)
        ax2.plot([fc for fc in best_cc], [best_cc[fc][0] for fc in best_cc], '*',color=color)
        ax2.tick_params(axis='y', labelcolor=color)

        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        plt.savefig('%s_tradeoff.png'%figname[:-2])

        print('plotting %s, %d times average'%(figname,len(plotter_inputs)))

        fig_scale = 4.5
        fig = plt.figure(figsize=(3*fig_scale,2*fig_scale))
        ax1 = fig.add_subplot(231, projection='3d')
        ax1.bar3d(hw_qubits, fc_qubits, np.zeros(len(plotter_input)), dx, dy, searcher_time_avg)
        ax1.set_xlabel('hardware qubits')
        ax1.set_ylabel('full circuit qubits')
        ax1.set_zlabel('searcher time (seconds)')
        ax1 = fig.add_subplot(232, projection='3d')
        ax1.bar3d(hw_qubits, fc_qubits, np.zeros(len(plotter_input)), dx, dy, classical_time_avg)
        ax1.set_zlim3d(0, 1.2*max(classical_time_avg)+1)
        ax1.set_xlabel('hardware qubits')
        ax1.set_ylabel('full circuit qubits')
        ax1.set_zlabel('classical evaluator time (seconds)')
        ax1 = fig.add_subplot(233, projection='3d')
        ax1.bar3d(hw_qubits, fc_qubits, np.zeros(len(plotter_input)), dx, dy, quantum_time_avg)
        ax1.set_zlim3d(0, 1.2*max(quantum_time_avg)+1)
        ax1.set_xlabel('hardware qubits')
        ax1.set_ylabel('full circuit qubits')
        ax1.set_zlabel('quantum evaluator time (seconds)')
        ax1 = fig.add_subplot(234, projection='3d')
        ax1.bar3d(hw_qubits, fc_qubits, np.zeros(len(plotter_input)), dx, dy, uniter_time_avg)
        ax1.set_xlabel('hardware qubits')
        ax1.set_ylabel('full circuit qubits')
        ax1.set_zlabel('reconstructor time (seconds)')
        ax1 = fig.add_subplot(235, projection='3d')
        ax1.bar3d(hw_qubits, fc_qubits, np.zeros(len(plotter_input)), dx, dy, percent_change_avg)
        ax1.set_zlim3d(min(0,1.2*min(percent_change_avg)), 1.2*max(percent_change_avg))
        ax1.set_xlabel('hardware qubits')
        ax1.set_ylabel('full circuit qubits')
        ax1.set_zlabel('cross entropy gap reduction due to cutting (%)')
        # pickle.dump(fig,open('%s'%figname, 'wb'))
        plt.savefig('%s.png'%figname[:-2])
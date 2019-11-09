import pickle
import glob
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
from helper_fun import cross_entropy

def initialize_dict(cases):
    empty_dict = {}
    for case in cases:
        empty_dict[case] = 0.0
    return empty_dict

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

        hw_qubits = [case[0] for case in plotter_inputs[0]]
        fc_qubits = [case[1] for case in plotter_inputs[0]]
        dx = [0.2 for x in plotter_inputs[0]]
        dy = [0.2 for x in plotter_inputs[0]]

        searcher_time_avg = initialize_dict(plotter_inputs[0].keys())
        classical_time_avg = initialize_dict(plotter_inputs[0].keys())
        quantum_time_avg = initialize_dict(plotter_inputs[0].keys())
        uniter_time_avg = initialize_dict(plotter_inputs[0].keys())
        ground_truth_avg = initialize_dict(plotter_inputs[0].keys())
        qasm_avg = initialize_dict(plotter_inputs[0].keys())
        qasm_noise_avg = initialize_dict(plotter_inputs[0].keys())
        hw_fc_avg = initialize_dict(plotter_inputs[0].keys())
        cutting_avg = initialize_dict(plotter_inputs[0].keys())
        percent_change_avg = initialize_dict(plotter_inputs[0].keys())

        for i, plotter_input in enumerate(plotter_inputs):
            print('repetition ',i)
            # Iterate over repetitions
            for case in plotter_input:
                searcher_time_avg[case] += plotter_input[case]['searcher_time']
                classical_time_avg[case] += plotter_input[case]['classical_time']
                quantum_time_avg[case] += plotter_input[case]['quantum_time']
                uniter_time_avg[case] += plotter_input[case]['uniter_time']

                case_ground_truth = cross_entropy(target=plotter_input[case]['evaluations']['sv_noiseless'],
                obs= plotter_input[case]['evaluations']['sv_noiseless'])
                
                case_qasm = cross_entropy(target=plotter_input[case]['evaluations']['sv_noiseless'],
                obs= plotter_input[case]['evaluations']['qasm'])
                
                case_qasm_noise = cross_entropy(target=plotter_input[case]['evaluations']['sv_noiseless'],
                obs= plotter_input[case]['evaluations']['qasm+noise'])

                case_hw_fc = cross_entropy(target=plotter_input[case]['evaluations']['sv_noiseless'],
                obs= plotter_input[case]['evaluations']['hw'])

                case_cutting = cross_entropy(target=plotter_input[case]['evaluations']['sv_noiseless'],
                obs= plotter_input[case]['evaluations']['cutting'])

                # FIXME: percent change calculations are wrong
                case_percent_change = 100*(case_hw_fc - case_cutting)/(case_hw_fc - case_ground_truth)
                print('case {}: plotter calculated: {}, uniter calculated: {}, reconstruction time: {:.3e}'.format(case,
                case_percent_change,plotter_input[case]['percent_reduction'],plotter_input[case]['uniter_time']))
                
                ground_truth_avg[case] += case_ground_truth
                qasm_avg[case] += case_qasm
                qasm_noise_avg[case] += case_qasm_noise
                hw_fc_avg[case] += case_hw_fc
                cutting_avg[case] += case_cutting
                percent_change_avg[case] += case_percent_change

                # print('case {} reduction:{},time:{}'.format(case,case_percent_change,plotter_input[case]['uniter_time']))
            print('*'*50)
        
        num_repetitions = len(plotter_inputs)
        for dictionary in (searcher_time_avg,classical_time_avg,quantum_time_avg,uniter_time_avg,ground_truth_avg,qasm_avg,qasm_noise_avg,hw_fc_avg,cutting_avg,percent_change_avg):
            for case in dictionary:
                dictionary[case] = dictionary[case]/num_repetitions

        best_cc = {}
        for case in percent_change_avg:
            percent = percent_change_avg[case]
            uniter_time = uniter_time_avg[case]
            hw, fc = case
            if (fc in best_cc and percent>best_cc[fc][0]) or (fc not in best_cc):
                best_cc[fc] = (percent,uniter_time)
        [print('Full circuit size %d. Cross entropy reduction = %.3f%%. Reconstruction time = %.3e seconds.'%(fc,best_cc[fc][0],best_cc[fc][1])) for fc in best_cc]

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
        plt.savefig('%s_tradeoff.png'%figname[:-2],dpi=400)

        print('plotting %s, %d times average'%(figname,len(plotter_inputs)))

        fig_scale = 4.5
        fig = plt.figure(figsize=(3*fig_scale,2*fig_scale))
        ax1 = fig.add_subplot(231, projection='3d')
        ax1.bar3d(hw_qubits, fc_qubits, np.zeros(len(plotter_input)), dx, dy, [searcher_time_avg[case] for case in searcher_time_avg])
        ax1.set_xlabel('hardware qubits')
        ax1.set_ylabel('full circuit qubits')
        ax1.set_zlabel('searcher time (seconds)')
        ax1 = fig.add_subplot(232, projection='3d')
        ax1.bar3d(hw_qubits, fc_qubits, np.zeros(len(plotter_input)), dx, dy, [classical_time_avg[case] for case in classical_time_avg])
        ax1.set_zlim3d(0, 1.2*max(classical_time_avg.values())+1)
        ax1.set_xlabel('hardware qubits')
        ax1.set_ylabel('full circuit qubits')
        ax1.set_zlabel('classical evaluator time (seconds)')
        ax1 = fig.add_subplot(233, projection='3d')
        ax1.bar3d(hw_qubits, fc_qubits, np.zeros(len(plotter_input)), dx, dy, [quantum_time_avg[case] for case in quantum_time_avg])
        ax1.set_zlim3d(0, 1.2*max(quantum_time_avg.values())+1)
        ax1.set_xlabel('hardware qubits')
        ax1.set_ylabel('full circuit qubits')
        ax1.set_zlabel('quantum evaluator time (seconds)')
        ax1 = fig.add_subplot(234, projection='3d')
        ax1.bar3d(hw_qubits, fc_qubits, np.zeros(len(plotter_input)), dx, dy, [uniter_time_avg[case] for case in uniter_time_avg])
        ax1.set_xlabel('hardware qubits')
        ax1.set_ylabel('full circuit qubits')
        ax1.set_zlabel('reconstructor time (seconds)')
        ax1 = fig.add_subplot(235, projection='3d')
        ax1.bar3d(hw_qubits, fc_qubits, np.zeros(len(plotter_input)), dx, dy, [percent_change_avg[case] for case in percent_change_avg])
        ax1.set_zlim3d(min(0,1.2*min(percent_change_avg.values())), max(0,1.2*max(percent_change_avg.values())))
        ax1.set_xlabel('hardware qubits')
        ax1.set_ylabel('full circuit qubits')
        ax1.set_zlabel('cross entropy gap reduction due to cutting (%)')
        # pickle.dump(fig,open('%s'%figname, 'wb'))
        plt.savefig('%s.png'%figname[:-2],dpi=400)
        print('-'*100)
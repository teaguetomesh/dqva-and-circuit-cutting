import pickle
import argparse
import glob
import numpy as np
from time import time
import itertools
import matplotlib.pyplot as plt
import os

from cutqc.helper_fun import get_dirname
from qiskit_helper_functions.non_ibmq_functions import read_dict, evaluate_circ
from qiskit_helper_functions.metrics import chi2_distance

def verify(full_circuit,ground_truth,unordered,complete_path_map,subcircuits,counter,layer_schedule):
    subcircuit_out_qubits = {}
    for input_qubit in complete_path_map:
        path = complete_path_map[input_qubit]
        output_qubit = path[-1]
        if output_qubit['subcircuit_idx'] in subcircuit_out_qubits:
            subcircuit_out_qubits[output_qubit['subcircuit_idx']].append((output_qubit['subcircuit_qubit'],full_circuit.qubits.index(input_qubit)))
        else:
            subcircuit_out_qubits[output_qubit['subcircuit_idx']] = [(output_qubit['subcircuit_qubit'],full_circuit.qubits.index(input_qubit))]
    # [print('subcircuit {:d} out qubits : {}'.format(x,subcircuit_out_qubits[x])) for x in subcircuit_out_qubits]
    for subcircuit_idx in subcircuit_out_qubits:
        subcircuit_out_qubits[subcircuit_idx] = sorted(subcircuit_out_qubits[subcircuit_idx],
        key=lambda x:subcircuits[subcircuit_idx].qubits.index(x[0]),reverse=True)
        subcircuit_out_qubits[subcircuit_idx] = [x[1] for x in subcircuit_out_qubits[subcircuit_idx]]
    print('subcircuit_out_qubits:',subcircuit_out_qubits,'smart_order:',layer_schedule['smart_order'])

    unordered_qubit_format = []
    unordered_qubit_state = []
    for subcircuit_idx in layer_schedule['smart_order']:
        print('subcircuit %d'%subcircuit_idx,layer_schedule['subcircuit_state'][subcircuit_idx])
        if subcircuit_idx in subcircuit_out_qubits:
            unordered_qubit_format += subcircuit_out_qubits[subcircuit_idx]
            unordered_qubit_state += layer_schedule['subcircuit_state'][subcircuit_idx]

    unordered_qubit_basis = []
    num_active = unordered_qubit_state.count('active')
    for val in unordered_qubit_state:
        if val=='active':
            num_active-=1
            unordered_qubit_basis.append(int(2**num_active))
        else:
            unordered_qubit_basis.append(0)
    # print('unordered_qubit_format:',unordered_qubit_format)
    # print('unordered_qubit_state:',unordered_qubit_state)
    # print('unordered_qubit_basis:',unordered_qubit_basis)

    ordered_qubit_format, ordered_qubit_state, ordered_qubit_basis = zip(*sorted(zip(unordered_qubit_format, unordered_qubit_state, unordered_qubit_basis),reverse=True))
    # print('ordered_qubit_format:',ordered_qubit_format)
    # print('ordered_qubit_state:',ordered_qubit_state)
    # print('ordered_qubit_basis:',ordered_qubit_basis)

    blurred_sv = np.zeros(len(unordered))
    for state_ctr, true_p in enumerate(ground_truth):
        bin_state_ctr = bin(state_ctr)[2:].zfill(len(ordered_qubit_format))
        # print('bin_state_ctr : %s. true_p = %.3f'%(bin_state_ctr,true_p))
        blurred_state = 0
        match_state = True
        for obs_state, expected_state, basis in zip(bin_state_ctr,ordered_qubit_state,ordered_qubit_basis):
            # print('obs_state %s, expected_state %s, basis = %d'%(obs_state,expected_state,basis))
            if expected_state!='merged' and expected_state!='active' and obs_state!=expected_state:
                match_state = False
                break
            else:
                blurred_state += int(obs_state)*basis
        if match_state:
            blurred_sv[blurred_state] += true_p
        #     print('blurred_state : %d'%blurred_state)
        # else:
        #     print('Does not match DD recursion pattern')
    # print('blurred ground_truth:',blurred_sv,np.sum(blurred_sv))
    # print('reconstructed_prob:',unordered,np.sum(unordered))
    blurred_chi2 = chi2_distance(target=blurred_sv,obs=unordered,normalize=False)
    
    ordered = {}
    chi2_bins = []
    num_active = unordered_qubit_state.count('active')
    num_merged = unordered_qubit_state.count('merged')
    merged_states = list(itertools.product(['0','1'],repeat=num_merged))
    for state_ctr, unordered_p in enumerate(unordered):
        bin_state_ctr = bin(state_ctr)[2:].zfill(num_active)
        avg_unordered_p = unordered_p/2**num_merged
        # print('bin_state_ctr : %s. unordered_p = %.3f, average over %d merged states = %.3f'%(
        #     bin_state_ctr,unordered_p,2**num_merged,avg_unordered_p))
        subgroup_chi2 = 0
        for merged_state in merged_states:
            bin_state_ctr = bin(state_ctr)[2:].zfill(num_active)
            bin_full_state = []
            for qubit in unordered_qubit_state:
                if qubit=='active':
                    bin_full_state.append(bin_state_ctr[0])
                    bin_state_ctr = bin_state_ctr[1:]
                elif qubit=='merged':
                    bin_full_state.append(merged_state[0])
                    merged_state = merged_state[1:]
                elif qubit=='0' or qubit =='1':
                    bin_full_state.append(qubit)
            _, bin_full_state = zip(*sorted(zip(unordered_qubit_format, bin_full_state),reverse=True))
            bin_full_state = ''.join(bin_full_state)
            full_state = int(bin_full_state,2)
            ordered_p = ground_truth[full_state]
            if abs(ordered_p-avg_unordered_p)<1e-16:
                chi2_contribution = 0
            else:
                chi2_contribution = np.power(ordered_p-avg_unordered_p,2)/(ordered_p+avg_unordered_p)
            subgroup_chi2 += chi2_contribution
            ordered[full_state] = avg_unordered_p
        #     print('bin_full_state : %s, full_state : %d, ordered_p = %.3f, \N{GREEK SMALL LETTER CHI}^2 contribution = %.3e'%(
        #         bin_full_state,full_state,ordered_p, chi2_contribution))
        # print('subgroup \N{GREEK SMALL LETTER CHI}^2 = %.3e'%subgroup_chi2)
        chi2_bins.append(subgroup_chi2)
    return blurred_chi2, chi2_bins, ordered

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='verify CutQC')
    parser.add_argument('--circuit_name', metavar='S', type=str,help='which circuit input file to run')
    parser.add_argument('--max_subcircuit_qubit', metavar='N',type=int,help='Max subcircuit qubit')
    parser.add_argument('--early_termination', type=int,choices=[0,1],help='use early_termination')
    parser.add_argument('--qubit_limit', type=int,help='Determines number of bins during dynamic definition')
    parser.add_argument('--num_workers', type=int,help='Number of parallel workers for horizontal_collapse and build')
    parser.add_argument('--eval_mode', type=str,help='Evaluation backend mode')
    args = parser.parse_args()

    source_folder = get_dirname(circuit_name=args.circuit_name,max_subcircuit_qubit=args.max_subcircuit_qubit,
    early_termination=None,eval_mode=None,num_workers=None,qubit_limit=None,field='cutter')
    dest_folder = get_dirname(circuit_name=args.circuit_name,max_subcircuit_qubit=args.max_subcircuit_qubit,
    early_termination=args.early_termination,num_workers=args.num_workers,eval_mode=args.eval_mode,qubit_limit=args.qubit_limit,field='build')
    case_dict = read_dict(filename='%s/subcircuits.pckl'%source_folder)
    if len(case_dict)==0:
        exit(0)

    full_circuit = case_dict['circuit']
    subcircuits = case_dict['subcircuits']
    complete_path_map = case_dict['complete_path_map']
    # [print(x,complete_path_map[x]) for x in complete_path_map]
    print('--> Verifying %d-qubit %s <--'%(full_circuit.num_qubits,args.circuit_name))

    sv = evaluate_circ(circuit=full_circuit,backend='statevector_simulator')
    # for state, p in enumerate(sv):
    #     if p>1e-5:
    #         print(bin(state)[2:].zfill(full_circ_size),p)

    x_ticks_to_plot = np.arange(2**full_circuit.num_qubits)
    idx = np.round(np.linspace(0, len(x_ticks_to_plot) - 1, 5)).astype(int)
    x_ticks_to_plot = [x_ticks_to_plot[x] for x in idx]

    meta_data = read_dict(filename='%s/meta_data.pckl'%dest_folder)
    dynamic_definition_folders = glob.glob('%s/dynamic_definition_*'%dest_folder)
    recursion_depth = len(dynamic_definition_folders)
    all_chi2 = {}
    ordered = {}

    # fig, axs = plt.subplots(2, 2)
    # recursion_ax = {0:axs[0,0],1:axs[0,1],2:axs[1,0],3:axs[1,1]}

    for recursion_layer in range(recursion_depth):
        print('Recursion layer %d'%recursion_layer)
        dynamic_definition_folder = '%s/dynamic_definition_%d'%(dest_folder,recursion_layer)
        build_output = read_dict(filename='%s/build_output.pckl'%(dynamic_definition_folder))
        reconstructed_prob = build_output['reconstructed_prob']
    
        # `blurred_chi2` : reconstructed_prob vs blurred ground_truth, over current recursion bin, for checking purposes
        # `chi2_bins` : averaged reconstructed_prob vs sv for the bins in the current recursion
        blurred_chi2, chi2_bins, recursion_ordered = verify(full_circuit=full_circuit,ground_truth=sv,unordered=reconstructed_prob,
        complete_path_map=complete_path_map,subcircuits=subcircuits,counter=meta_data['counter'],
        layer_schedule=meta_data['dynamic_definition_schedule'][recursion_layer])
        for state_idx, chi2_bin in enumerate(chi2_bins):
            all_chi2[(recursion_layer,state_idx)] = chi2_bin
        upper_bin = meta_data['dynamic_definition_schedule'][recursion_layer]['upper_bin']
        if upper_bin in all_chi2:
            del all_chi2[upper_bin]
        ordered.update(recursion_ordered)
        ordered_list = []
        for state in range(2**full_circuit.num_qubits):
            ordered_list.append(ordered[state])
        verify_chi2 = chi2_distance(target=sv,obs=ordered_list,normalize=True)
        cumulative_chi2 = sum(all_chi2.values())

        pickle.dump({'ordered':ordered_list,'sv':sv}, open('%s/DD_%d.pckl'%(dest_folder,recursion_layer),'wb'))
        print('blurred_\N{GREEK SMALL LETTER CHI}^2 = %.3e, cumulative_\N{GREEK SMALL LETTER CHI}^2 = %.3e, verify_\N{GREEK SMALL LETTER CHI}^2 = %.3e'%(
            blurred_chi2,cumulative_chi2,verify_chi2))

        xdata = np.arange(2**full_circuit.num_qubits)
        # recursion_ax[recursion_layer].plot(xdata,sv,'k--',label='Ground Truth')
        # recursion_ax[recursion_layer].bar(xdata, ordered_list, label='Recursion %d'%(recursion_layer+1))
        # recursion_ax[recursion_layer].set_ylim(0,max(sv)*1.1)
        # recursion_ax[recursion_layer].legend()
        # recursion_ax[recursion_layer].set_xticks(x_ticks_to_plot)
        # recursion_ax[recursion_layer].set_xticklabels(x_ticks_to_plot)
        # recursion_ax[recursion_layer].tick_params(axis='y', labelsize=13)
        # recursion_ax[recursion_layer].tick_params(axis='x', labelsize=13)
        # if recursion_layer==0 or recursion_layer==2:
        #     recursion_ax[recursion_layer].set_ylabel('Probability',fontsize=13)
        # if recursion_layer==2 or recursion_layer==3:
        #     recursion_ax[recursion_layer].set_xlabel('Output States',fontsize=13)
        print('-'*50)
    # fig.tight_layout()
    # if not os.path.exists('./paper_plots'):
    #     os.makedirs('./paper_plots')
    # plt.savefig('./paper_plots/DD_%s_example.pdf'%args.circuit_name,dpi=400)
    # plt.close()
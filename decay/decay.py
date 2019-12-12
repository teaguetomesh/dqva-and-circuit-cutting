import numpy as np
import matplotlib.pyplot as plt
from qcg.generators import gen_supremacy, gen_hwea, gen_BV, gen_qft, gen_sycamore
from utils.helper_fun import evaluate_circ, get_evaluator_info, factor_int, cross_entropy
import argparse
import os
import itertools
from mpi4py import MPI

def find_rank_combinations(combinations,rank,size):
    num_workers = size - 1
    rank_combinations = []

    count = int(len(combinations)/num_workers)
    remainder = len(combinations) % num_workers
    if rank<remainder:
        combinations_start = rank * (count + 1)
        combinations_stop = combinations_start + count + 1
    else:
        combinations_start = rank * count + remainder
        combinations_stop = combinations_start + (count - 1) + 1

    rank_combinations = combinations[combinations_start:combinations_stop]
    return rank_combinations

def calculate_delta_H(circ,min_ce,ground_truth,accumulated_prob,counter,shots_increment,evaluation_method):
    if evaluation_method == 'noisy_qasm_simulator':
        qasm_noise_evaluator_info = get_evaluator_info(circ=circ,device_name='ibmq_boeblingen',
        fields=['device','basis_gates','coupling_map','properties','initial_layout','noise_model'])
        qasm_noise_evaluator_info['num_shots'] = shots_increment
        prob_batch = evaluate_circ(circ=circ,backend='noisy_qasm_simulator',evaluator_info=qasm_noise_evaluator_info)
    elif evaluation_method == 'qasm_simulator':
        qasm_evaluator_info = {'num_shots':shots_increment}
        prob_batch = evaluate_circ(circ=circ,backend='noiseless_qasm_simulator',evaluator_info=qasm_evaluator_info)
    else:
        raise Exception('Illegal evaluation method:',evaluation_method)
    accumulated_prob = [(x*(counter-1)+y)/counter for x,y in zip(accumulated_prob,prob_batch)]
    accumulated_ce = cross_entropy(target=ground_truth,obs=accumulated_prob) - min_ce
    return accumulated_ce, accumulated_prob

if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    num_workers = size - 1

    if rank == size-1:
        a = 1e-1
        r = 1e-1
        length = 5
        first_derivatives = [a * r ** (n - 1) for n in range(1, length + 1)]
        a = 1e-1
        r = 1e-1
        length = 10
        second_derivatives = [a * r ** (n - 1) for n in range(1, length + 1)]
        combinations = list(itertools.product(first_derivatives, second_derivatives))
        for i in range(num_workers):
            comm.send(combinations, dest=i)
    else:
        state = MPI.Status()
        combinations = comm.recv(source=size-1,status=state)
        rank_combinations = find_rank_combinations(combinations,rank,size)
        for combination in rank_combinations:
            first_derivative_threshold, second_derivative_threshold = combination
            dirname = './decay/%.1e__%.1e_decays'%(first_derivative_threshold,second_derivative_threshold)
            if not os.path.exists(dirname):
                os.makedirs(dirname)

            for full_circ_size in range(3,19):
                fig_name = '%s/%d_decay.png'%(dirname,full_circ_size)
                if os.path.isfile(fig_name):
                    continue
                print('%d qubit full circuit'%full_circ_size)
                i, j = factor_int(full_circ_size)
                circ = gen_supremacy(i,j,8)
                ground_truth = evaluate_circ(circ=circ,backend='statevector_simulator',evaluator_info=None)
                min_ce = cross_entropy(target=ground_truth,obs=ground_truth)
                shots_increment = 1024

                noiseless_accumulated_prob = [0 for i in range(np.power(2,len(circ.qubits)))]
                noisy_accumulated_prob = [0 for i in range(np.power(2,len(circ.qubits)))]
                noiseless_delta_H_l = []
                noisy_delta_H_l = []
                counter = 1
                max_counter = max(20,int(20*np.power(2,full_circ_size)/shots_increment))
                cutoff = max_counter
                found_saturation = False
                while 1:
                    if found_saturation and counter>cutoff+10:
                        break
                    elif not found_saturation and counter>max_counter:
                        break
                    print('Counter %d, shots = %d'%(counter,counter*shots_increment))
                    noiseless_accumulated_ce, noiseless_accumulated_prob = calculate_delta_H(circ=circ,min_ce=min_ce,ground_truth=ground_truth,
                    accumulated_prob=noiseless_accumulated_prob,counter=counter,shots_increment=shots_increment,evaluation_method='qasm_simulator')
                    noiseless_delta_H_l.append(noiseless_accumulated_ce)
                    
                    noisy_accumulated_ce, noisy_accumulated_prob = calculate_delta_H(circ=circ,min_ce=min_ce,ground_truth=ground_truth,
                    accumulated_prob=noisy_accumulated_prob,counter=counter,shots_increment=shots_increment,evaluation_method='noisy_qasm_simulator')
                    noisy_delta_H_l.append(noisy_accumulated_ce)
                    
                    if len(noiseless_delta_H_l)>=3:
                        first_derivative = (noiseless_delta_H_l[-1]+noiseless_delta_H_l[-3])/(2*shots_increment)
                        second_derivative = (noiseless_delta_H_l[-1]+noiseless_delta_H_l[-3]-2*noiseless_delta_H_l[-2])/(np.power(shots_increment,2))
                        print('noiseless \u0394H = %.3f, first derivative = %.3e, second derivative = %.3e'%(noiseless_accumulated_ce,first_derivative,second_derivative))

                        first_derivative = (noisy_delta_H_l[-1]+noisy_delta_H_l[-3])/(2*shots_increment)
                        second_derivative = (noisy_delta_H_l[-1]+noisy_delta_H_l[-3]-2*noisy_delta_H_l[-2])/(np.power(shots_increment,2))
                        print('noisy \u0394H = %.3f, first derivative = %.3e, second derivative = %.3e'%(noisy_accumulated_ce,first_derivative,second_derivative))

                        if abs(first_derivative)<first_derivative_threshold and abs(second_derivative)<second_derivative_threshold and noiseless_accumulated_ce<noisy_accumulated_ce and not found_saturation:
                            print('*'*50,'SATURATED','*'*50)
                            cutoff = counter
                            found_saturation = True
                    print('-'*50)
                    counter += 1
                
                plot_start = max(1,cutoff-20)
                xvals = range(plot_start,len(noisy_delta_H_l)+1)
                shots = [x*shots_increment for x in xvals]
                plt.figure()
                plt.axvline(x=cutoff,label='saturated cutoff',color='k',linestyle='--')
                plt.plot(xvals,noiseless_delta_H_l[plot_start-1:],label='noiseless')
                plt.plot(xvals,noisy_delta_H_l[plot_start-1:],label='noisy')
                plt.xticks(ticks=xvals,labels=xvals)
                plt.ylabel('\u0394H')
                plt.xlabel('shots [*1024]')
                plt.title('%d qubit full circuit'%full_circ_size)
                plt.legend()
                plt.savefig(fig_name,dpi=400)
                plt.close()
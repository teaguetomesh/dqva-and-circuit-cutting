import numpy as np
from qcg.generators import gen_supremacy, gen_hwea, gen_BV, gen_qft, gen_sycamore
from utils.helper_fun import evaluate_circ, factor_int, cross_entropy, read_file
import os
from mpi4py import MPI
import pickle

def find_rank_tasks(tasks,rank,num_workers):
    rank_tasks = []

    count = int(len(tasks)/num_workers)
    remainder = len(tasks) % num_workers
    if rank<remainder:
        tasks_start = rank * (count + 1)
        tasks_stop = tasks_start + count + 1
    else:
        tasks_start = rank * count + remainder
        tasks_stop = tasks_start + (count - 1) + 1

    rank_tasks = tasks[tasks_start:tasks_stop]
    return rank_tasks

def calculate_delta_H(circ,ground_truth,accumulated_prob,counter,shots_increment,evaluation_method):
    if evaluation_method == 'qasm_simulator':
        qasm_evaluator_info = {'num_shots':shots_increment}
        prob_batch = evaluate_circ(circ=circ,backend='noiseless_qasm_simulator',evaluator_info=qasm_evaluator_info)
    else:
        raise Exception('Illegal evaluation method:',evaluation_method)
    accumulated_prob = [(x*(counter-1)+y)/counter for x,y in zip(accumulated_prob,prob_batch)]
    assert abs(sum(accumulated_prob)-1)<1e-10
    accumulated_ce = cross_entropy(target=ground_truth,obs=accumulated_prob)
    return accumulated_ce, accumulated_prob

def noiseless_decay(circuit,shots_increment):
    full_circ_size = len(circuit.qubits)
    # print('%d qubit full circuit, shots increment = %d'%(full_circ_size,shots_increment))
    ground_truth = evaluate_circ(circ=circuit,backend='statevector_simulator',evaluator_info=None)
    noiseless_accumulated_prob = [0 for i in range(np.power(2,full_circ_size))]
    noiseless_delta_H_l = []
    max_counter = max(20,int(20*np.power(2,full_circ_size)/shots_increment))
    for counter in range(1,max_counter+1):
        # print('Counter %d, shots = %d'%(counter,counter*shots_increment))
        noiseless_accumulated_ce, noiseless_accumulated_prob = calculate_delta_H(circ=circuit,ground_truth=ground_truth,
        accumulated_prob=noiseless_accumulated_prob,counter=counter,shots_increment=shots_increment,evaluation_method='qasm_simulator')
        noiseless_delta_H_l.append(noiseless_accumulated_ce)
        
    return noiseless_delta_H_l

if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    num_workers = size - 1

    if rank == size-1:
        full_circ_sizes = []
        decay_dict = read_file(filename='./decay/decay.pickle')
        for full_circ_size in range(3,21):
            if full_circ_size not in decay_dict:
                full_circ_sizes.append(full_circ_size)
        rearranged_fc_sizes = []
        while len(full_circ_sizes)>0:
            if len(full_circ_sizes)>=2:
                head = full_circ_sizes.pop(0)
                tail = full_circ_sizes.pop(-1)
                rearranged_fc_sizes.append(head)
                rearranged_fc_sizes.append(tail)
            else:
                rearranged_fc_sizes.append(full_circ_sizes.pop(0))
        for i in range(num_workers):
            rank_tasks = find_rank_tasks(tasks=rearranged_fc_sizes,rank=i,num_workers=num_workers)
            comm.send(rank_tasks, dest=i)
        for i in range(num_workers):
            state = MPI.Status()
            rank_decay_dict = comm.recv(source=i,status=state)
            decay_dict.update(rank_decay_dict)
        print('Decay results have :',decay_dict.keys())
        pickle.dump(decay_dict,open('./decay/decay.pickle','wb'))
    else:
        state = MPI.Status()
        full_circ_sizes = comm.recv(source=size-1,status=state)
        print('Rank %d runs :'%rank,full_circ_sizes,flush=True)
        rank_decay_dict = {}
        for full_circ_size in full_circ_sizes:
            i, j = factor_int(full_circ_size)
            circ = gen_supremacy(i,j,8)

            shots_increment = 1024
        
            noiseless_delta_H_l = noiseless_decay(circuit=circ,shots_increment=shots_increment)
            rank_decay_dict[full_circ_size] = {'ce_l':noiseless_delta_H_l,'shots_increment':shots_increment}
        comm.send(rank_decay_dict, dest=size-1)
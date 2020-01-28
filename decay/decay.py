import numpy as np
from qcg.generators import gen_supremacy, gen_hwea, gen_BV, gen_qft, gen_sycamore
from utils.helper_fun import evaluate_circ, factor_int, read_file, get_evaluator_info
from utils.conversions import dict_to_array
from utils.metrics import chi2_distance
from scipy.stats import wasserstein_distance
import os
from mpi4py import MPI
import pickle
from time import time

def find_rank_tasks(tasks,rank,num_workers):
    rank_tasks = []
    rank_ptr = 0
    task_idx = 0
    decreasing = False
    while task_idx<len(tasks):
        if rank_ptr==rank:
            rank_tasks.append(tasks[task_idx])
        task_idx += 1
        if decreasing and rank_ptr==0:
            decreasing = False
        elif not decreasing and rank_ptr==num_workers-1:
            decreasing = True
        elif decreasing:
            rank_ptr -= 1
        elif not decreasing:
            rank_ptr += 1
        else:
            print('Should not reach here')
    return rank_tasks

def accumulate_batch(circ,accumulated_prob,counter,shots_increment,evaluation_method):
    if evaluation_method == 'qasm_simulator':
        qasm_evaluator_info = {'num_shots':shots_increment}
        prob_batch = evaluate_circ(circ=circ,backend='noiseless_qasm_simulator',evaluator_info=qasm_evaluator_info,force_prob=True)
        prob_batch = dict_to_array(distribution_dict=prob_batch,force_prob=True)
    else:
        raise Exception('Illegal evaluation method:',evaluation_method)
    accumulated_prob = ((counter-1)*accumulated_prob+prob_batch)/counter
    assert abs(sum(accumulated_prob)-1)<1e-10
    return accumulated_prob

def noiseless_decay(circuit,shots_increment,device_max_experiments):
    decay_begin = time()
    ground_truth = evaluate_circ(circ=circuit,backend='statevector_simulator',evaluator_info=None,force_prob=True)
    ground_truth = dict_to_array(distribution_dict=ground_truth,force_prob=True)
    full_circ_size = len(circuit.qubits)
    # print('%d qubit full circuit, shots increment = %d'%(full_circ_size,shots_increment))
    noiseless_accumulated_prob = np.zeros(2**full_circ_size,dtype=float)
    chi2_l = []
    distance_l = []
    max_counter = max(20,int(20*np.power(2,full_circ_size)/shots_increment))
    max_counter = min(max_counter,device_max_experiments)
    for counter in range(1,max_counter+1):
        noiseless_accumulated_prob = accumulate_batch(circ=circuit,accumulated_prob=noiseless_accumulated_prob,
        counter=counter,shots_increment=shots_increment,evaluation_method='qasm_simulator')
        chi2 = chi2_distance(target=ground_truth,obs=noiseless_accumulated_prob)
        distance = wasserstein_distance(u_values=ground_truth,v_values=noiseless_accumulated_prob)
        chi2_l.append(chi2)
        distance_l.append(distance)
        if full_circ_size>=15 and counter%50==0:
            time_elapsed = time()-decay_begin
            eta = time_elapsed/counter*max_counter-time_elapsed
            print('%d qubit circuit, counter %d/%d, ETA = %.1e'%(full_circ_size,counter,max_counter,eta),flush=True)
    return chi2_l, distance_l

if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    num_workers = size

    full_circ_sizes = []
    decay_dict = read_file(filename='./decay/decay.pickle')
    for full_circ_size in range(3,21):
        if full_circ_size not in decay_dict:
            full_circ_sizes.append(full_circ_size)
        
    evaluator_info = get_evaluator_info(circ=None,device_name='ibmq_boeblingen',fields=['properties','device'])
    device_max_shots = evaluator_info['device'].configuration().max_shots
    device_max_experiments = evaluator_info['device'].configuration().max_experiments
    
    rank_tasks = find_rank_tasks(tasks=full_circ_sizes,rank=rank,num_workers=num_workers)
    print('Rank %d runs :'%rank,rank_tasks,flush=True)
    rank_decay_dict = {}
    for full_circ_size in rank_tasks:
        i, j = factor_int(full_circ_size)
        circ = gen_supremacy(i,j,8)

        shots_increment = device_max_shots
    
        chi2_l, distance_l = noiseless_decay(circuit=circ,shots_increment=shots_increment,device_max_experiments=device_max_experiments)
        rank_decay_dict[full_circ_size] = {'circ':circ,'chi2_l':chi2_l,'distance':distance_l,'shots_increment':shots_increment}
    
    if rank == size - 1:
        decay_dict.update(rank_decay_dict)
        print('Decay results have :',decay_dict.keys())
        pickle.dump(decay_dict,open('./decay/decay.pickle','wb'))
        for i in range(num_workers-1):
            state = MPI.Status()
            rank_decay_dict = comm.recv(source=i,status=state)
            decay_dict.update(rank_decay_dict)
            print('Decay results have :',decay_dict.keys())
            pickle.dump(decay_dict,open('./decay/decay.pickle','wb'))
    else:
        comm.send(rank_decay_dict, dest=size-1)
from qcg.generators import gen_hwea
import numpy as np
from helper_fun import evaluate_circ, find_saturated_shots, get_circ_saturated_shots, fidelity, cross_entropy
from evaluator_prob import find_rank_combinations, get_evaluator_info
import MIQCP_searcher as searcher
import cutter
from uniter_prob import reconstruct
import argparse
from time import time
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.extensions.standard import HGate, SGate, SdgGate, XGate
from scipy.stats import wasserstein_distance
import matplotlib.pyplot as plt

def evaluate_cluster(complete_path_map, cluster_circ, combinations, backend, evaluator_info):
    cluster_prob = {}
    for _, combination in enumerate(combinations):
        cluster_dag = circuit_to_dag(cluster_circ)
        inits, meas = combination
        for i,x in enumerate(inits):
            q = cluster_circ.qubits[i]
            if x == 'zero':
                continue
            elif x == 'one':
                cluster_dag.apply_operation_front(op=XGate(),qargs=[q],cargs=[])
            elif x == 'plus':
                cluster_dag.apply_operation_front(op=HGate(),qargs=[q],cargs=[])
            elif x == 'minus':
                cluster_dag.apply_operation_front(op=HGate(),qargs=[q],cargs=[])
                cluster_dag.apply_operation_front(op=XGate(),qargs=[q],cargs=[])
            elif x == 'plus_i':
                cluster_dag.apply_operation_front(op=SGate(),qargs=[q],cargs=[])
                cluster_dag.apply_operation_front(op=HGate(),qargs=[q],cargs=[])
            elif x == 'minus_i':
                cluster_dag.apply_operation_front(op=SGate(),qargs=[q],cargs=[])
                cluster_dag.apply_operation_front(op=HGate(),qargs=[q],cargs=[])
                cluster_dag.apply_operation_front(op=XGate(),qargs=[q],cargs=[])
            else:
                raise Exception('Illegal initialization : ',x)
        for i,x in enumerate(meas):
            q = cluster_circ.qubits[i]
            if x == 'I':
                continue
            elif x == 'X':
                cluster_dag.apply_operation_back(op=HGate(),qargs=[q],cargs=[])
            elif x == 'Y':
                cluster_dag.apply_operation_back(op=SdgGate(),qargs=[q],cargs=[])
                cluster_dag.apply_operation_back(op=HGate(),qargs=[q],cargs=[])
            else:
                raise Exception('Illegal measurement basis:',x)
        cluster_circ_inst = dag_to_circuit(cluster_dag)
        if backend=='statevector_simulator':
            cluster_inst_prob = evaluate_circ(circ=cluster_circ_inst,backend=backend,evaluator_info=None)
            cluster_prob[(tuple(inits),tuple(meas))] = cluster_inst_prob
        elif backend=='noisy_qasm_simulator':
            cluster_inst_prob = evaluate_circ(circ=cluster_circ_inst,backend=backend,evaluator_info=evaluator_info)
            cluster_prob[(tuple(inits),tuple(meas))] = cluster_inst_prob
        elif backend=='hardware':
            cluster_prob[(tuple(inits),tuple(meas))] = cluster_circ_inst
        else:
            raise Exception('Illegal backend:',backend)
    return cluster_prob

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Large quantum circuits on small machine.')
    parser.add_argument('--device-name', metavar='S', type=str,help='which evaluator device input file to run')
    parser.add_argument('--evaluation-method', metavar='S', type=str,help='which evaluator backend to use')
    parser.add_argument('--saturated-shots',action="store_true",help='run saturated number of cluster shots')
    args = parser.parse_args()

    device_name = args.device_name
    device_properties = get_evaluator_info(circ=None,device_name=device_name,fields=['properties'])
    device_size = len(device_properties['properties'].qubits)

    cluster_max_qubits = np.arange(3,7)
    max_clusters = 3
    fc_sizes = np.arange(10,11)
    rank_classical_time = {}
    rank_quantum_time = {}
    rank_results = {}
    results = {}
    for cluster_max_qubit in cluster_max_qubits:
        for fc_size in fc_sizes:
            if fc_size<=cluster_max_qubit or fc_size>device_size or (cluster_max_qubit-1)*max_clusters<fc_size:
                continue
            case = (cluster_max_qubit,fc_size)
            print('case',case)
            rank_results[case] = {}
            results[case] = {}
            rank_quantum_time[case] = 0
            rank_classical_time[case] = 0
            full_circ = gen_hwea(fc_size,1)
            
            hardness, positions, ancilla, d, num_cluster, m = searcher.find_cuts(circ=full_circ,num_clusters=range(2,max_clusters+1),hw_max_qubit=cluster_max_qubit,evaluator_weight=1)
            if m == None:
                print('Case {} not feasible'.format(case))
                print('-'*100)
                continue
            m.print_stat()
            clusters, complete_path_map, K, d = cutter.cut_circuit(full_circ, positions)
            total_shots = find_saturated_shots(clusters=clusters,complete_path_map=complete_path_map,accuracy=1e-1)
            evaluator_input = {case:{'clusters':clusters,'complete_path_map':complete_path_map,'total_shots':total_shots}}
            rank_combinations = find_rank_combinations(evaluator_input,rank=0,size=2)

            ground_truth = evaluate_circ(circ=full_circ,backend='statevector_simulator',evaluator_info=None)
            results[case]['ground_truth'] = ground_truth
            evaluator_info = get_evaluator_info(circ=full_circ,device_name=args.device_name,
            fields=['device','basis_gates','coupling_map','properties','initial_layout','noise_model'])
            evaluator_info['num_shots'] = total_shots
            qasm_noisy_fc = evaluate_circ(circ=full_circ,backend='noisy_qasm_simulator',evaluator_info=evaluator_info)
            results[case]['vanilla'] = qasm_noisy_fc
            
            clusters = evaluator_input[case]['clusters']
            complete_path_map = evaluator_input[case]['complete_path_map']
            total_shots = evaluator_input[case]['total_shots']
            for cluster_idx in range(len(rank_combinations[case])):
                if args.evaluation_method == 'statevector_simulator':
                    classical_evaluator_begin = time()
                    cluster_prob = evaluate_cluster(complete_path_map=complete_path_map,
                    cluster_circ=clusters[cluster_idx],
                    combinations=rank_combinations[case][cluster_idx],
                    backend='statevector_simulator',evaluator_info=None)
                    elapsed_time = time()-classical_evaluator_begin
                    rank_classical_time[case] += elapsed_time
                    print('runs case {}, cluster_{} {}_qubits * {}_instances on CLASSICAL, classical time = {:.3e}'.format(
                        case,cluster_idx,len(clusters[cluster_idx].qubits),
                        len(rank_combinations[case][cluster_idx]),elapsed_time))
                elif args.evaluation_method == 'noisy_qasm_simulator':
                    evaluator_info = get_evaluator_info(circ=clusters[cluster_idx],device_name=args.device_name,
                    fields=['device','basis_gates','coupling_map','properties','initial_layout','noise_model'])
                    quantum_evaluator_begin = time()
                    if args.saturated_shots:
                        evaluator_info['num_shots'] = get_circ_saturated_shots(circ=clusters[cluster_idx],accuracy=1e-3)
                    else:
                        evaluator_info['num_shots'] = int(total_shots/len(rank_combinations[case][cluster_idx]))+1
                    cluster_prob = evaluate_cluster(complete_path_map=complete_path_map,
                    cluster_circ=clusters[cluster_idx],
                    combinations=rank_combinations[case][cluster_idx],
                    backend='noisy_qasm_simulator',evaluator_info=evaluator_info)
                    elapsed_time = time()-quantum_evaluator_begin
                    rank_quantum_time[case] += elapsed_time
                    print('runs case {}, cluster_{} {}_qubits * {}_instances on {} QUANTUM SIMULATOR, {} shots = {}, quantum time  = {:.3e}'.format(
                            case,cluster_idx,len(clusters[cluster_idx].qubits),
                            len(rank_combinations[case][cluster_idx]),args.device_name,'saturated' if args.saturated_shots else 'same_total',evaluator_info['num_shots'], elapsed_time))
                elif args.evaluation_method == 'hardware':
                    quantum_evaluator_begin = time()
                    cluster_prob = evaluate_cluster(complete_path_map=complete_path_map,
                    cluster_circ=clusters[cluster_idx],
                    combinations=rank_combinations[case][cluster_idx],
                    backend='hardware',evaluator_info=None)
                    elapsed_time = time()-quantum_evaluator_begin
                    rank_quantum_time[case] += elapsed_time
                    print('case {}, cluster_{} {}_qubits * {}_instances on {} QUANTUM HARDWARE, {} shots'.format(
                            case,cluster_idx,len(clusters[cluster_idx].qubits),
                            len(rank_combinations[case][cluster_idx]),args.device_name,'saturated' if args.saturated_shots else 'same_total'))
                else:
                    raise Exception('Illegal evaluation method:',args.evaluation_method)
                rank_results[case][cluster_idx] = cluster_prob
            all_cluster_prob = rank_results[case]
            reconstructed_prob = reconstruct(complete_path_map=complete_path_map, full_circ=full_circ, cluster_circs=clusters, cluster_sim_probs=all_cluster_prob)
            results[case]['cutting'] = reconstructed_prob
            vanilla_fid = fidelity(target=ground_truth,obs=qasm_noisy_fc)
            cutting_fid = fidelity(target=ground_truth,obs=reconstructed_prob)
            vanilla_ce = cross_entropy(target=ground_truth,obs=qasm_noisy_fc)
            cutting_ce = cross_entropy(target=ground_truth,obs=reconstructed_prob)
            print('vanilla fidelity = %.3f. cutting fidelity = %.3f'%(vanilla_fid,cutting_fid))
            print('vanilla ce = %.3f. cutting ce = %.3f'%(vanilla_ce,cutting_ce))
            print('-'*100)

    d1 = results[(5,10)]['ground_truth']
    d2 = results[(5,10)]['vanilla']
    d3 = [abs(x) for x in results[(5,10)]['cutting']]
    
    x = np.arange(len(d1))

    plt.figure()
    plt.bar(x,height=d1,label='ground truth, %.3e'%cross_entropy(d1,d1))
    plt.xlabel('quantum state')
    plt.ylabel('probability')
    plt.legend()
    plt.savefig('ground_truth_eg.png',dpi=400)
    plt.close()

    plt.figure()
    plt.bar(x,height=d2,label='vanilla execution, %.3e'%cross_entropy(d1,d2))
    plt.ylim(0,0.5)
    plt.xlabel('quantum state')
    plt.ylabel('probability')
    plt.legend()
    plt.savefig('vanilla_execution_eg.png',dpi=400)
    plt.close()

    plt.figure()
    plt.bar(x,height=d3,label='cutting, %.3e'%cross_entropy(d1,d3))
    plt.ylim(0,0.5)
    plt.xlabel('quantum state')
    plt.ylabel('probability')
    plt.legend()
    plt.savefig('cutting_eg.png',dpi=400)
    plt.close()
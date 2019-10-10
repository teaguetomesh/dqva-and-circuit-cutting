import pickle
import os
import subprocess
from time import time
import numpy as np
from qcg.generators import gen_supremacy, gen_hwea
import MIQCP_searcher as searcher
import cutter
import evaluator_prob as evaluator
import uniter_prob as uniter
from scipy.stats import wasserstein_distance
from qiskit.quantum_info.states.measures import state_fidelity
from qiskit import Aer, IBMQ, execute
from qiskit.providers.aer import noise

times = {'searcher':[],'evaluator':[],'uniter':[]}
num_qubits = []
qasm_distances = []
qasm_cutting_distances = []
max_qubit = 10
dirname = './data'
if not os.path.exists(dirname):
    os.mkdir(dirname)

for dimension in [[3,4]]:
    i,j = dimension
    if i*j<=24 and i*j not in num_qubits:
        print('-'*200)
        num_shots = int(1e4)
        print('%d * %d supremacy circuit, %d shots'%(i,j,num_shots))

        # Generate a circuit
        circ = gen_supremacy(i,j,8,order='75601234')
        # print(circ)

        # Looking for a cut
        searcher_begin = time()
        hardness, positions, ancilla, d, num_cluster, m = searcher.find_cuts(circ,num_clusters=range(1,4),hw_max_qubit=max_qubit,evaluator_weight=1)
        searcher_time = time() - searcher_begin
        m.print_stat()

        if len(positions)>0:
            clusters, complete_path_map, K, d = cutter.cut_circuit(circ, positions)
            print('Complete path map:')
            [print(x,complete_path_map[x]) for x in complete_path_map]

            pickle.dump([clusters,complete_path_map,None], open('%s/evaluator_input.p'%dirname,'wb'))

            # Simulate the clusters
            evaluator_begin = time()
            for cluster_idx in range(len(clusters)):
                print('MPI evaluator on cluster %d'%cluster_idx)
                # print(clusters[cluster_idx])
                subprocess.call(['mpiexec','-n','5','python','evaluator_prob.py','--cluster-idx','%d'%cluster_idx,'--backend','qasm_simulator','--shots','%d'%num_shots])
            evaluator_time = time()-evaluator_begin

            all_cluster_prob = []
            for cluster_idx in range(len(clusters)):
                cluster_prob = pickle.load( open('%s/cluster_%d_prob.p'%(dirname,cluster_idx), 'rb' ))
                all_cluster_prob.append(cluster_prob)

            # Reconstruct the circuit
            uniter_begin = time()
            qasm_cutting_noiseless = uniter.reconstruct(complete_path_map, circ, clusters, all_cluster_prob)
            uniter_time = time()-uniter_begin
        
        else:
            qasm_cutting_noiseless = evaluator.simulate_circ(circ=circ, simulator='qasm_simulator', noisy=False, provider_info=None, num_shots=num_shots)
            evaluator_time = 0
            uniter_time = 0

        print('Running full circuit')
        sv_fc_noiseless = evaluator.simulate_circ(circ=circ,simulator='statevector_simulator',noisy=False, provider_info=None, num_shots=None)
        qasm_fc_noiseless = evaluator.simulate_circ(circ=circ, simulator='qasm_simulator', noisy=False, provider_info=None, num_shots=num_shots)
        # qasm_fc_noisy = evaluator.simulate_circ(circ=circ, simulator='qasm_simulator', noisy=True, provider_info=provider_info, output_format='prob', num_shots=num_shots)
        
        qasm_distance = wasserstein_distance(sv_fc_noiseless,qasm_fc_noiseless)
        qasm_cutting_distance = wasserstein_distance(sv_fc_noiseless,qasm_cutting_noiseless)
        qasm_cutting_fidelity = state_fidelity(sv_fc_noiseless,qasm_cutting_noiseless)
        # no_cutting_distance = wasserstein_distance(qasm_fc_noiseless,qasm_fc_noisy)
        
        qasm_distances.append(qasm_distance)
        qasm_cutting_distances.append(qasm_cutting_distance)
        
        times['searcher'].append(searcher_time)
        times['evaluator'].append(evaluator_time)
        times['uniter'].append(uniter_time)
        num_qubits.append(i*j)
        print('distance due to qasm = ',qasm_distance)
        print('distance due to qasm+cutting =',qasm_cutting_distance)
        print('searcher time = %.3f seconds'%searcher_time)
        print('evaluator time = %.3f seconds'%evaluator_time)
        print('uniter time = %.3f seconds'%uniter_time)
        print('-'*200)
print('*'*200)
print(times)
print('num qubits:',num_qubits)
print('distances due to qasm :',qasm_distances)
print('distances due to qasm, cutting :',qasm_cutting_distances)

# pickle.dump([num_qubits,times,qasm_distances,qasm_cutting_distances], open('%s/noiseless_fidelity_benchmark.p'%dirname,'wb'))
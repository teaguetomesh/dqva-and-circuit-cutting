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
from qiskit import Aer, IBMQ, execute
from qiskit.providers.aer import noise

provider = IBMQ.load_account()
device = provider.get_backend('ibmq_16_melbourne')
properties = device.properties()
coupling_map = device.configuration().coupling_map

gate_times = [
('u1', None, 0), ('u2', None, 100), ('u3', None, 200),
('cx', [1, 0], 678), ('cx', [1, 2], 547), ('cx', [2, 3], 721),
('cx', [4, 3], 733), ('cx', [4, 10], 721), ('cx', [5, 4], 800),
('cx', [5, 6], 800), ('cx', [5, 9], 895), ('cx', [6, 8], 895),
('cx', [7, 8], 640), ('cx', [9, 8], 895), ('cx', [9, 10], 800),
('cx', [11, 10], 721), ('cx', [11, 3], 634), ('cx', [12, 2], 773),
('cx', [13, 1], 2286), ('cx', [13, 12], 1504), ('cx', [], 800)]

noise_model = noise.device.basic_device_noise_model(properties, gate_times=gate_times)
basis_gates = noise_model.basis_gates
provider_info=(provider,noise_model,coupling_map,basis_gates)

times = {'searcher':[],'evaluator':[],'uniter':[]}
num_qubits = []
noiseless_reconstruction_distance = []
noisy_reconstruction_distance = []
full_circ_noisy_noisless_distance = []
max_qubit = 10
dirname = './data'
if not os.path.exists(dirname):
    os.mkdir(dirname)

for dimension in [[3,4],[2,7],[4,4],[3,6],[4,5]]:
    i,j = dimension
    if i*j<=24 and i*j not in num_qubits:
        print('-'*200)
        print('%d * %d supremacy circuit'%(i,j))

        # Generate a circuit
        circ = gen_supremacy(i,j,8,order='75601234')
        # print(circ)

        # Looking for a cut
        searcher_begin = time()
        hardness, positions, ancilla, d, num_cluster, m = searcher.find_cuts(circ,num_clusters=range(1,4),hw_max_qubit=max_qubit,evaluator_weight=0)
        searcher_time = time() - searcher_begin
        m.print_stat()

        if len(positions)>0:
            clusters, complete_path_map, K, d = cutter.cut_circuit(circ, positions)
            print('Complete path map:')
            [print(x,complete_path_map[x]) for x in complete_path_map]

            pickle.dump([clusters,complete_path_map,provider_info], open('%s/evaluator_input.p'%dirname,'wb'))

            # Simulate the clusters
            evaluator_begin = time()
            for cluster_idx in range(len(clusters)):
                print('MPI evaluator on cluster %d'%cluster_idx)
                subprocess.call(['mpiexec','-n','5','python','evaluator_prob.py','--cluster-idx','%d'%cluster_idx,'--backend','statevector_simulator'])
            evaluator_time = time()-evaluator_begin

            all_cluster_prob = []
            for cluster_idx in range(len(clusters)):
                cluster_prob = pickle.load( open('%s/cluster_%d_prob.p'%(dirname,cluster_idx), 'rb' ))
                all_cluster_prob.append(cluster_prob)

            # Reconstruct the circuit
            uniter_begin = time()
            reconstructed_prob = uniter.reconstruct(complete_path_map, circ, clusters, all_cluster_prob)
            uniter_time = time()-uniter_begin
        
        else:
            reconstructed_prob = evaluator.simulate_circ(circ=circ, simulator='qasm_simulator', noisy=True, provider_info=provider_info, output_format='prob',num_shots=int(2*np.power(2,i*j)))
            evaluator_time = 0
            uniter_time = 0

        full_circ_noiseless_prob = evaluator.simulate_circ(circ=circ,simulator='statevector_simulator',output_format='prob')
        noiseless_distance = wasserstein_distance(full_circ_noiseless_prob,reconstructed_prob)
        # full_circ_noisy_prob = evaluator.simulate_circ(circ=circ, simulator='qasm_simulator', noisy=True, provider_info=provider_info, output_format='prob', num_shots=int(2*np.power(2,i*j)))
        # noisy_distance = wasserstein_distance(full_circ_noisy_prob,reconstructed_prob)
        # full_circ_distance = wasserstein_distance(full_circ_noisy_prob,full_circ_noiseless_prob)
        
        noiseless_reconstruction_distance.append(noiseless_distance)
        # noisy_reconstruction_distance.append(noisy_distance)
        # full_circ_noisy_noisless_distance.append(full_circ_distance)
        times['searcher'].append(searcher_time)
        times['evaluator'].append(evaluator_time)
        times['uniter'].append(uniter_time)
        num_qubits.append(i*j)
        print('wasserstein distance to noiseless full circ = ',noiseless_distance)
        # print('wasserstein distance to noisy full circ = ',noisy_distance)
        # print('wasserstein distance between noisy and noiseless full circ = ',full_circ_distance)
        print('searcher time = %.3f seconds'%searcher_time)
        print('evaluator time = %.3f seconds'%evaluator_time)
        print('uniter time = %.3f seconds'%uniter_time)
        print('-'*200)
print('*'*200)
print(times)
print('num qubits:',num_qubits)
print('wasserstein distance to noiseless full circ :',noiseless_reconstruction_distance)
print('wasserstein distance to noisy full circ :',noisy_reconstruction_distance)
print('wasserstein distance between noisy and noiseless full circ = ',full_circ_noisy_noisless_distance)

pickle.dump([num_qubits,times,noiseless_reconstruction_distance,noisy_reconstruction_distance,full_circ_noisy_noisless_distance], open( '%s/full_stack_benchmark.p'%dirname,'wb'))
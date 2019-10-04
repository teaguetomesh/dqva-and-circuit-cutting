import pickle
from time import time
from qcg.generators import gen_supremacy, gen_hwea
import MIQCP_searcher as searcher
import cutter
import simulator_prob as simulator
import uniter_prob as uniter
from scipy.stats import wasserstein_distance

times = {'searcher':[],'simulator':[],'uniter':[]}
num_qubits = []
reconstruction_distance = []
max_qubit = 16
for dimension in [[3,3],[3,4],[3,5],[4,4]]:
    i,j = dimension
    if i*j<=24 and i*j not in num_qubits:
        print('-'*200)
        print('%d * %d supremacy circuit'%(i,j))

        # Generate a circuit
        circ = gen_supremacy(i,j,8,order='75601234')
        # print(circ)

        # Looking for a cut
        searcher_begin = time()
        hardness, positions, ancilla, d, num_cluster, m = searcher.find_cuts(circ,num_clusters=range(1,5),hw_max_qubit=max_qubit,alpha=0)
        searcher_end = time()
        m.print_stat()

        clusters, complete_path_map, K, d = cutter.cut_circuit(circ, positions)
        print('Complete path map:')
        [print(x,complete_path_map[x]) for x in complete_path_map]

        # Simulate the clusters
        simulator_begin = time()
        all_cluster_prob = simulator.simulate_clusters(complete_path_map,clusters,simulator_backend='statevector_simulator')
        simulator_end = time()

        # Reconstruct the circuit
        uniter_begin = time()
        reconstructed_prob = uniter.reconstruct(complete_path_map, circ, clusters, all_cluster_prob)
        uniter_end = time()

        full_circ_sim_prob = simulator.simulate_circ(circ=circ,simulator='statevector_simulator',output_format='prob')
        distance = wasserstein_distance(full_circ_sim_prob,reconstructed_prob)
        
        reconstruction_distance.append(distance)
        times['searcher'].append(searcher_end-searcher_begin)
        times['simulator'].append(simulator_end-simulator_begin)
        times['uniter'].append(uniter_end-uniter_begin)
        num_qubits.append(i*j)
        # print('probability reconstruction distance = ',distance)
        # print('searcher time = %.3f seconds'%(searcher_end-searcher_begin))
        print('simulator time = %.3f seconds'%(simulator_end-simulator_begin))
        print('uniter time = %.3f seconds'%(uniter_end-uniter_begin))
        print('-'*200)
print('*'*200)
print(times)
print('num qubits:',num_qubits)
print('reconstruction distance:',reconstruction_distance)

pickle.dump([num_qubits,times,reconstruction_distance], open( 'full_stack_benchmark.p','wb'))
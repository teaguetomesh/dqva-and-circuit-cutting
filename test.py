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
from qiskit.quantum_info.states.measures import state_fidelity

dirname = './tmp'
if not os.path.exists(dirname):
    os.mkdir(dirname)

provider_info = None

circ = gen_supremacy(3,3,8,order='75601234')
hardness, positions, ancilla, d, num_cluster, m = searcher.find_cuts(circ,num_clusters=range(1,4),hw_max_qubit=6,evaluator_weight=1)
m.print_stat()
clusters, complete_path_map, K, d = cutter.cut_circuit(circ, positions)
pickle.dump([clusters,complete_path_map,provider_info], open('%s/evaluator_input.p'%dirname,'wb'))
for cluster_idx in range(len(clusters)):
    print('MPI evaluator on cluster %d'%cluster_idx)
    subprocess.call(['mpiexec','-n','2','python','evaluator_prob.py','--cluster-idx','%d'%cluster_idx,'--backend','statevector_simulator','--dirname','%s'%dirname])

all_cluster_prob = []
for cluster_idx in range(len(clusters)):
    cluster_prob = pickle.load( open('%s/cluster_%d_prob.p'%(dirname,cluster_idx), 'rb' ))
    all_cluster_prob.append(cluster_prob)

sv_cutting_noiseless = uniter.reconstruct(complete_path_map, circ, clusters, all_cluster_prob)
sv_fc_noiseless = evaluator.simulate_circ(circ, 'statevector_simulator', noisy=False, provider_info=None, output_format='sv', num_shots=1024)
print('fidelity = ',state_fidelity(sv_cutting_noiseless,sv_fc_noiseless))
print('first element comparison:', sv_cutting_noiseless[0],sv_fc_noiseless[0])
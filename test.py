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
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.extensions.standard import HGate, SGate, SdgGate, XGate

def cross_entropy(d1,d2):
    h = 0
    for p,q in zip(d1,d2):
        if p==0:
            h += 0
        else:
            h+= -p*np.log(q)
    return h

dirname = './debug'
if not os.path.exists(dirname):
    os.mkdir(dirname)

provider_info = None

circ = gen_supremacy(1,3,8,order='75601234')
print('Original circuit')
print(circ)
hardness, positions, ancilla, d, num_cluster, m = searcher.find_cuts(circ,num_clusters=range(2,4),hw_max_qubit=2,evaluator_weight=1)
clusters, complete_path_map, K, d = cutter.cut_circuit(circ, positions)
print('cutting positions :',positions)
print('Complete path map:')
[print(x,complete_path_map[x]) for x in complete_path_map]
for i,cluster in enumerate(clusters):
    print('cluster %d'%i)
    print(cluster)
print('*'*100)
# pickle.dump([clusters,complete_path_map,provider_info], open('%s/evaluator_input.p'%dirname,'wb'))
# for cluster_idx in range(len(clusters)):
#     subprocess.call(['mpiexec','-n','2','python','evaluator_prob.py',
#         '--cluster-idx','%d'%cluster_idx,
#         '--backend','statevector_simulator','--dirname','%s'%dirname])
#     print('-'*100)
# print('*'*100)
# all_cluster_prob = []
# for cluster_idx in range(len(clusters)):
#     cluster_prob = pickle.load( open('%s/cluster_%d_prob.p'%(dirname,cluster_idx), 'rb' ))
#     all_cluster_prob.append(cluster_prob)

# print('Running uniter')
# sv_cutting_noiseless = uniter.reconstruct(complete_path_map, circ, clusters, all_cluster_prob)
sv_noiseless_fc = evaluator.simulate_circ(circ=circ,backend='statevector_simulator',noisy=False,qasm_info=None)
# print('fidelity = ',state_fidelity(sv_cutting_noiseless,sv_noiseless_fc))
# print('first element comparison:', sv_cutting_noiseless[0],sv_noiseless_fc[0])

print('*'*70,'Manual Construction','*'*70)

print('S = 1')
cluster_0_circ = clusters[0]
cluster_0_dag = circuit_to_dag(cluster_0_circ)
cluster_0_circ = dag_to_circuit(cluster_0_dag)
cluster_0 = evaluator.simulate_circ(cluster_0_circ, 'statevector_simulator', noisy=False,qasm_info=None)
cluster_0_collapsed = [cluster_0[0]+cluster_0[1],cluster_0[2]+cluster_0[3]]
cluster_1_circ = clusters[1]
cluster_1_dag = circuit_to_dag(cluster_1_circ)
cluster_1_circ = dag_to_circuit(cluster_1_dag)
cluster_1 = evaluator.simulate_circ(cluster_1_circ, 'statevector_simulator', noisy=False,qasm_info=None)
term_1 = np.kron(cluster_0_collapsed,cluster_1)*(1/2)
print('Cluster 0 Evaluate (''zero'', ''zero'') (''I'', ''I'')')
print(cluster_0_circ)
print(cluster_0_collapsed)
print('Cluster 1 Evaluate (''zero'', ''zero'') (''I'', ''I'')')
print(cluster_1_circ)
print(cluster_1)
print('-'*100)

print('S = 2')
cluster_0_circ = clusters[0]
cluster_0_dag = circuit_to_dag(cluster_0_circ)
cluster_0_circ = dag_to_circuit(cluster_0_dag)
cluster_0 = evaluator.simulate_circ(cluster_0_circ, 'statevector_simulator', noisy=False,qasm_info=None)
cluster_0_collapsed = [cluster_0[0]+cluster_0[1],cluster_0[2]+cluster_0[3]]
cluster_1_circ = clusters[1]
cluster_1_dag = circuit_to_dag(cluster_1_circ)
cluster_1_dag.apply_operation_front(op=XGate(),qargs=[cluster_1_circ.qubits[0]],cargs=[])
cluster_1_circ = dag_to_circuit(cluster_1_dag)
cluster_1 = evaluator.simulate_circ(cluster_1_circ, 'statevector_simulator', noisy=False,qasm_info=None)
term_2 = np.kron(cluster_0_collapsed,cluster_1)*(1/2)
print('Cluster 0 Evaluate (''zero'', ''zero'') (''I'', ''I'')')
print(cluster_0_circ)
print(cluster_0_collapsed)
print('Cluster 1 Evaluate (''one'', ''zero'') (''I'', ''I'')')
print(cluster_1_circ)
print(cluster_1)
print('-'*100)

print('S = 3')
cluster_0_circ = clusters[0]
cluster_0_dag = circuit_to_dag(cluster_0_circ)
cluster_0_dag.apply_operation_back(op=HGate(),qargs=[cluster_0_circ.qubits[1]],cargs=[])
cluster_0_circ = dag_to_circuit(cluster_0_dag)
cluster_0 = evaluator.simulate_circ(cluster_0_circ, 'statevector_simulator', noisy=False,qasm_info=None)
cluster_0_collapsed = [cluster_0[0]-cluster_0[1],cluster_0[2]-cluster_0[3]]
cluster_1_circ = clusters[1]
cluster_1_dag = circuit_to_dag(cluster_1_circ)
cluster_1_dag.apply_operation_front(op=HGate(),qargs=[cluster_1_circ.qubits[0]],cargs=[])
cluster_1_circ = dag_to_circuit(cluster_1_dag)
cluster_1 = evaluator.simulate_circ(cluster_1_circ, 'statevector_simulator', noisy=False,qasm_info=None)
term_3 = np.kron(cluster_0_collapsed,cluster_1)*(1/2)
print('Cluster 0 Evaluate (''zero'', ''zero'') (''I'', ''X'')')
print(cluster_0_circ)
print(cluster_0_collapsed)
print('Cluster 1 Evaluate (''plus'', ''zero'') (''I'', ''I'')')
print(cluster_1_circ)
print(cluster_1)
print('-'*100)

print('S = 4')
cluster_0_circ = clusters[0]
cluster_0_dag = circuit_to_dag(cluster_0_circ)
cluster_0_dag.apply_operation_back(op=HGate(),qargs=[cluster_0_circ.qubits[1]],cargs=[])
cluster_0_circ = dag_to_circuit(cluster_0_dag)
cluster_0 = evaluator.simulate_circ(cluster_0_circ, 'statevector_simulator', noisy=False,qasm_info=None)
cluster_0_collapsed = [cluster_0[0]-cluster_0[1],cluster_0[2]-cluster_0[3]]
cluster_1_circ = clusters[1]
cluster_1_dag = circuit_to_dag(cluster_1_circ)
cluster_1_dag.apply_operation_front(op=HGate(),qargs=[cluster_1_circ.qubits[0]],cargs=[])
cluster_1_dag.apply_operation_front(op=XGate(),qargs=[cluster_1_circ.qubits[0]],cargs=[])
cluster_1_circ = dag_to_circuit(cluster_1_dag)
cluster_1 = evaluator.simulate_circ(cluster_1_circ, 'statevector_simulator', noisy=False,qasm_info=None)
term_4 = np.kron(cluster_0_collapsed,cluster_1)*(-1/2)
print('Cluster 0 Evaluate (''zero'', ''zero'') (''I'', ''X'')')
print(cluster_0_circ)
print(cluster_0_collapsed)
print('Cluster 1 Evaluate (''minus'', ''zero'') (''I'', ''I'')')
print(cluster_1_circ)
print(cluster_1)
print('-'*100)

print('S = 5')
cluster_0_circ = clusters[0]
cluster_0_dag = circuit_to_dag(cluster_0_circ)
cluster_0_dag.apply_operation_back(op=SdgGate(),qargs=[cluster_0_circ.qubits[1]],cargs=[])
cluster_0_dag.apply_operation_back(op=HGate(),qargs=[cluster_0_circ.qubits[1]],cargs=[])
cluster_0_circ = dag_to_circuit(cluster_0_dag)
cluster_0 = evaluator.simulate_circ(cluster_0_circ, 'statevector_simulator', noisy=False,qasm_info=None)
cluster_0_collapsed = [cluster_0[0]-cluster_0[1],cluster_0[2]-cluster_0[3]]
cluster_1_circ = clusters[1]
cluster_1_dag = circuit_to_dag(cluster_1_circ)
cluster_1_dag.apply_operation_front(op=SGate(),qargs=[cluster_1_circ.qubits[0]],cargs=[])
cluster_1_dag.apply_operation_front(op=HGate(),qargs=[cluster_1_circ.qubits[0]],cargs=[])
cluster_1_circ = dag_to_circuit(cluster_1_dag)
cluster_1 = evaluator.simulate_circ(cluster_1_circ, 'statevector_simulator', noisy=False,qasm_info=None)
term_5 = np.kron(cluster_0_collapsed,cluster_1)*(1/2)
print('Cluster 0 Evaluate (''zero'', ''zero'') (''I'', ''Y'')')
print(cluster_0_circ)
print(cluster_0_collapsed)
print('Cluster 1 Evaluate (''plus_i'', ''zero'') (''I'', ''I'')')
print(cluster_1_circ)
print(cluster_1)
print('-'*100)

print('S = 6')
cluster_0_circ = clusters[0]
cluster_0_dag = circuit_to_dag(cluster_0_circ)
cluster_0_dag.apply_operation_back(op=SdgGate(),qargs=[cluster_0_circ.qubits[1]],cargs=[])
cluster_0_dag.apply_operation_back(op=HGate(),qargs=[cluster_0_circ.qubits[1]],cargs=[])
cluster_0_circ = dag_to_circuit(cluster_0_dag)
cluster_0 = evaluator.simulate_circ(cluster_0_circ, 'statevector_simulator', noisy=False,qasm_info=None)
cluster_0_collapsed = [cluster_0[0]-cluster_0[1],cluster_0[2]-cluster_0[3]]
cluster_1_circ = clusters[1]
cluster_1_dag = circuit_to_dag(cluster_1_circ)
cluster_1_dag.apply_operation_front(op=SGate(),qargs=[cluster_1_circ.qubits[0]],cargs=[])
cluster_1_dag.apply_operation_front(op=HGate(),qargs=[cluster_1_circ.qubits[0]],cargs=[])
cluster_1_dag.apply_operation_front(op=XGate(),qargs=[cluster_1_circ.qubits[0]],cargs=[])
cluster_1_circ = dag_to_circuit(cluster_1_dag)
cluster_1 = evaluator.simulate_circ(cluster_1_circ, 'statevector_simulator', noisy=False,qasm_info=None)
term_6 = np.kron(cluster_0_collapsed,cluster_1)*(-1/2)
print('Cluster 0 Evaluate (''zero'', ''zero'') (''I'', ''Y'')')
print(cluster_0_circ)
print(cluster_0_collapsed)
print('Cluster 1 Evaluate (''minus_i'', ''zero'') (''I'', ''I'')')
print(cluster_1_circ)
print(cluster_1)
print('-'*100)

print('S = 7')
cluster_0_circ = clusters[0]
cluster_0_dag = circuit_to_dag(cluster_0_circ)
cluster_0_circ = dag_to_circuit(cluster_0_dag)
cluster_0 = evaluator.simulate_circ(cluster_0_circ, 'statevector_simulator', noisy=False,qasm_info=None)
cluster_0_collapsed = [cluster_0[0]-cluster_0[1],cluster_0[2]-cluster_0[3]]
cluster_1_circ = clusters[1]
cluster_1_dag = circuit_to_dag(cluster_1_circ)
cluster_1_circ = dag_to_circuit(cluster_1_dag)
cluster_1 = evaluator.simulate_circ(cluster_1_circ, 'statevector_simulator', noisy=False,qasm_info=None)
term_7 = np.kron(cluster_0_collapsed,cluster_1)*(1/2)
print('Cluster 0 Evaluate (''zero'', ''zero'') (''I'', ''Z'')')
print(cluster_0_circ)
print(cluster_0_collapsed)
print('Cluster 1 Evaluate (''zero'', ''zero'') (''I'', ''I'')')
print(cluster_1_circ)
print(cluster_1)
print('-'*100)

print('S = 8')
cluster_0_circ = clusters[0]
cluster_0_dag = circuit_to_dag(cluster_0_circ)
cluster_0_circ = dag_to_circuit(cluster_0_dag)
cluster_0 = evaluator.simulate_circ(cluster_0_circ, 'statevector_simulator', noisy=False,qasm_info=None)
cluster_0_collapsed = [cluster_0[0]-cluster_0[1],cluster_0[2]-cluster_0[3]]
cluster_1_circ = clusters[1]
cluster_1_dag = circuit_to_dag(cluster_1_circ)
cluster_1_dag.apply_operation_front(op=XGate(),qargs=[cluster_1_circ.qubits[0]],cargs=[])
cluster_1_circ = dag_to_circuit(cluster_1_dag)
cluster_1 = evaluator.simulate_circ(cluster_1_circ, 'statevector_simulator', noisy=False,qasm_info=None)
term_8 = np.kron(cluster_0_collapsed,cluster_1)*(-1/2)
print('Cluster 0 Evaluate (''zero'', ''zero'') (''I'', ''Z'')')
print(cluster_0_circ)
print(cluster_0_collapsed)
print('Cluster 1 Evaluate (''one'', ''zero'') (''I'', ''I'')')
print(cluster_1_circ)
print(cluster_1)
print('-'*100)

manual_reconstruction = term_1+term_2+term_3+term_4+term_5+term_6+term_7+term_8
summation = sum(manual_reconstruction)
if isinstance(summation, complex):
    print('manual reconstruction fidelity = ',state_fidelity(manual_reconstruction,sv_noiseless_fc))
    print('identical distributions fidelity = ',state_fidelity(sv_noiseless_fc,sv_noiseless_fc))
    print(manual_reconstruction)
    print(sv_noiseless_fc)
else:
    print('sum of prob =',summation)
    print('manual reconstruction cross entropy = ',cross_entropy(manual_reconstruction,sv_noiseless_fc))
    print('identical distributions cross entropy = ',cross_entropy(sv_noiseless_fc,sv_noiseless_fc))
from datetime import datetime
from qiskit.providers.models import BackendProperties
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit import Aer, IBMQ, execute
from qcg.generators import gen_supremacy, gen_hwea
from helper_fun import simulate_circ, find_saturated_shots, load_IBMQ, cross_entropy
from qiskit.providers.aer import noise
from qiskit.transpiler.passes import NoiseAdaptiveLayout
import MIQCP_searcher as searcher
import cutter
from evaluator_prob import find_rank_combinations, evaluate_cluster
from uniter_prob import reconstruct
from time import time

provider = load_IBMQ()
public_provider = IBMQ.get_provider('ibm-q')
mel = public_provider.get_backend('ibmq_16_melbourne')
prop = mel.properties()
qubit_list = prop.qubits[:]
for i in range(len(qubit_list)):
    idx = -1
    for j,nduv in enumerate(qubit_list[i]):
        if nduv.name == 'readout_error':
            idx = j
            break
    if idx != -1:
        qubit_list[i][idx].value = 0.0
calib_time = datetime(year=2019, month=10, day=15, hour=0, minute=0, second=0) #junk, set any time you like
bprop = BackendProperties(last_update_date=calib_time, backend_name="no_readout_error", qubits=qubit_list, backend_version="1.0.0", gates=prop.gates, general=[])
bprop_noise_model = noise.device.basic_device_noise_model(bprop)
coupling_map = mel.configuration().coupling_map
noise_model = noise.device.basic_device_noise_model(prop)
basis_gates = noise_model.basis_gates
case = (3,3)
circ = gen_supremacy(int(case[0]),int(case[1]),8,order='75601234')
saturated_shots = find_saturated_shots(circ)
qasm_info = [mel,prop,coupling_map,noise_model,basis_gates,saturated_shots]
print(circ)
print('saturated shots = ',saturated_shots)

ground_truth = simulate_circ(circ,'statevector_simulator',None)
qasm = simulate_circ(circ,'noiseless_qasm_simulator',qasm_info)
vanilla_execution = simulate_circ(circ,'noisy_qasm_simulator',qasm_info,bprop_noise_model)

hardness, positions, ancilla, d, num_cluster, m = searcher.find_cuts(circ,num_clusters=range(2,4),hw_max_qubit=6,evaluator_weight=1)
m.print_stat()
clusters, complete_path_map, K, d = cutter.cut_circuit(circ, positions)

evaluator_input = {case:[None,None,None,None,None,clusters,complete_path_map]}
rank = 0
size = 2
num_workers = size - 1
rank_combinations = find_rank_combinations(evaluator_input,rank,size)
rank_results = {}
rank_classical_time = {}
rank_quantum_time = {}
for key in rank_combinations:
    rank_results[key] = {}
    rank_quantum_time[key] = 0
    rank_classical_time[key] = 0
    _,_,_,_,_,clusters,complete_path_map = evaluator_input[key]
    for cluster_idx in range(len(rank_combinations[key])):
        quantum_evaluator_begin = time()
        rank_shots = max(int(saturated_shots/len(rank_combinations[key][cluster_idx])/num_workers)+1,500)
        print('rank {} runs case {}, cluster_{} * {} on QUANTUM, sameTotal shots = {}'.format(
            rank,key,cluster_idx,
            len(rank_combinations[key][cluster_idx]),rank_shots))
        cluster_prob = evaluate_cluster(complete_path_map=complete_path_map,
        cluster_circ=clusters[cluster_idx],
        combinations=rank_combinations[key][cluster_idx],
        backend='noisy_qasm_simulator',num_shots=rank_shots,provider=provider,bprop_noise_model=bprop_noise_model)
        # cluster_prob = evaluate_cluster(complete_path_map=complete_path_map,
        # cluster_circ=clusters[cluster_idx],
        # combinations=rank_combinations[key][cluster_idx],
        # backend='statevector_simulator')
        rank_quantum_time[key] += time()-quantum_evaluator_begin
        rank_results[key][cluster_idx] = cluster_prob

reconstructed_prob = reconstruct(complete_path_map=complete_path_map, full_circ=circ, cluster_circs=clusters, cluster_sim_probs=rank_results[case])
print(cross_entropy(ground_truth,ground_truth))
print(cross_entropy(ground_truth,vanilla_execution))
print(cross_entropy(ground_truth,reconstructed_prob))
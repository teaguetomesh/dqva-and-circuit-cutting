from qcg.generators import gen_supremacy, gen_hwea
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
import MIQCP_searcher as searcher
import cutter
import pickle
import os
from qiskit import IBMQ
from qiskit.providers.aer import noise

provider = IBMQ.load_account()
device = provider.get_backend('ibmq_16_melbourne')
properties = device.properties()
coupling_map = device.configuration().coupling_map

noise_model = noise.device.basic_device_noise_model(properties)
basis_gates = noise_model.basis_gates
provider_info=(provider,noise_model,coupling_map,basis_gates)

circ = gen_supremacy(4,4,8,order='75601234')
print(circ)

hardness, positions, ancilla, d, num_cluster, m = searcher.find_cuts(circ,num_clusters=range(2,5),hw_max_qubit=10)
m.print_stat()
clusters, complete_path_map, K, d = cutter.cut_circuit(circ, positions)
print('Complete path map:')
[print(x,complete_path_map[x]) for x in complete_path_map]
print('*'*200)

dirname = './data'
if not os.path.exists(dirname):
    os.mkdir(dirname)
pickle.dump(circ, open( '%s/full_circ.p'%dirname, 'wb' ) )
pickle.dump([clusters, complete_path_map, provider_info],open('%s/evaluator_input.p'%dirname, 'wb'))
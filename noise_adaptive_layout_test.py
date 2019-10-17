import numpy as np
from qcg.generators import gen_supremacy, gen_hwea
from qiskit.transpiler.passes import NoiseAdaptiveLayout
from qiskit.converters import circuit_to_dag
from qiskit.compiler import transpile
from qiskit import Aer, IBMQ, execute
from qiskit.providers.aer import noise
from qiskit.circuit.classicalregister import ClassicalRegister
from qiskit import QuantumCircuit
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance

def cross_entropy(target,obs):
    assert len(target)==len(obs)
    alpha = 1e-14
    if 0 in obs:
        obs = [(x+alpha)/(1+alpha*len(obs)) for x in obs]
    assert abs(sum(obs)-1)<1e-5
    h = 0
    for p,q in zip(target,obs):
        if p==0:
            h += 0
        else:
            assert q>=0
            h += -p*np.log(q)
    return h

provider = IBMQ.load_account()
device = provider.get_backend('ibmq_16_melbourne')
properties = device.properties()
coupling_map = device.configuration().coupling_map
noise_model = noise.device.basic_device_noise_model(properties)
basis_gates = noise_model.basis_gates
num_shots = int(1e5)

circ = gen_supremacy(2,3,4)

# Ground truth
backend = Aer.get_backend('statevector_simulator')
job = execute(circ, backend=backend)
result = job.result()
outputstate = result.get_statevector(circ)
ground_truth = [np.power(np.absolute(x),2) for x in outputstate]
print(circ)
print('ground truth:',sum(ground_truth),len(ground_truth))
print('-'*100)

# qasm
c = ClassicalRegister(len(circ.qubits), 'c')
meas = QuantumCircuit(circ.qregs[0], c)
meas.barrier(circ.qubits)
meas.measure(circ.qubits,c)
qc = circ+meas
backend = Aer.get_backend('qasm_simulator')
result = execute(experiments=qc,
backend=backend,
shots=num_shots).result()
counts = result.get_counts(qc)
qasm_prob = [0 for i in range(np.power(2,len(circ.qubits)))]
for i in counts:
    qasm_prob[int(i,2)] = counts[i]/num_shots
print(qc)
print('qasm:',sum(qasm_prob),len(qasm_prob))
print('-'*100)

# qasm + noise
c = ClassicalRegister(len(circ.qubits), 'c')
meas = QuantumCircuit(circ.qregs[0], c)
meas.barrier(circ.qubits)
meas.measure(circ.qubits,c)
qc = circ+meas
backend = Aer.get_backend('qasm_simulator')
result = execute(experiments=qc,
backend=backend,
noise_model=noise_model,
coupling_map=coupling_map,
basis_gates=basis_gates,
shots=num_shots,
initial_layout=None).result()
counts = result.get_counts(qc)
qasm_noise_prob = [0 for i in range(np.power(2,len(circ.qubits)))]
for i in counts:
    qasm_noise_prob[int(i,2)] = counts[i]/num_shots
print(qc)
print('qasm+noise:',sum(qasm_noise_prob),len(qasm_noise_prob))
print('-'*100)

# Noise adaptive
dag = circuit_to_dag(circ)
noise_mapper = NoiseAdaptiveLayout(properties)
noise_mapper.run(dag)
initial_layout = noise_mapper.property_set['layout']
# new_circuit = transpile(circ, backend=device, basis_gates=basis_gates,coupling_map=coupling_map,backend_properties=properties,initial_layout=initial_layout)
new_circuit = circ
layout = initial_layout.get_physical_bits()
print(layout)
c = ClassicalRegister(len(circ.qubits), 'c')
meas = QuantumCircuit(new_circuit.qregs[0], c)
meas.barrier(new_circuit.qubits)
meas.measure(new_circuit.qubits,c)
# for physical_q in layout:
#     virtual_q = layout[physical_q]
#     if 'ancilla' not in virtual_q[0].name:
#         meas.measure(new_circuit.qubits[physical_q],c[virtual_q[1]])
qc = new_circuit+meas
print(qc)
backend = Aer.get_backend('qasm_simulator')
result = execute(experiments=qc,
backend=backend,
noise_model=noise_model,
coupling_map=coupling_map,
basis_gates=basis_gates,
shots=num_shots,
initial_layout=initial_layout).result()
counts = result.get_counts(qc)
qasm_noise_na_prob = [0 for i in range(np.power(2,len(circ.qubits)))]
for i in counts:
    qasm_noise_na_prob[int(i,2)] = counts[i]/num_shots
print('qasm+noise+na:',sum(qasm_noise_na_prob),len(qasm_noise_na_prob))
print('-'*100)

if len(circ.qubits)<=8:
    # Plot
    x = np.arange(0,len(ground_truth))
    plt.figure(figsize=(24,8))
    plt.subplot(221)
    plt.bar(x,height=ground_truth,label='ground truth')
    plt.legend()
    plt.title('ce=%.3e, distance=%.3e'%(cross_entropy(ground_truth,ground_truth),wasserstein_distance(ground_truth,ground_truth)))
    plt.subplot(222)
    plt.bar(x,height=qasm_prob,label='qasm')
    plt.legend()
    plt.title('ce=%.3e, distance=%.3e'%(cross_entropy(ground_truth,qasm_prob),wasserstein_distance(ground_truth,qasm_prob)))
    plt.subplot(223)
    plt.bar(x,height=qasm_noise_prob,label='qasm + noise')
    plt.legend()
    plt.title('ce=%.3e, distance=%.3e'%(cross_entropy(ground_truth,qasm_noise_prob),wasserstein_distance(ground_truth,qasm_noise_prob)))
    plt.subplot(224)
    plt.bar(x,height=qasm_noise_na_prob,label='qasm + noise + na')
    plt.legend()
    plt.title('ce=%.3e, distance=%.3e'%(cross_entropy(ground_truth,qasm_noise_na_prob),wasserstein_distance(ground_truth,qasm_noise_na_prob)))
    plt.show()
else:
    print('ground truth, ce = %.3e, distance = %.3e'%(cross_entropy(ground_truth,ground_truth),wasserstein_distance(ground_truth,ground_truth)))
    print('qasm, ce = %.3e, distance = %.3e'%(cross_entropy(ground_truth,qasm_prob),wasserstein_distance(ground_truth,qasm_prob)))
    print('qasm + noise, ce = %.3e, distance = %.3e'%(cross_entropy(ground_truth,qasm_noise_prob),wasserstein_distance(ground_truth,qasm_noise_prob)))
    print('qasm + noise + na, ce = %.3e, distance = %.3e'%(cross_entropy(ground_truth,qasm_noise_na_prob),wasserstein_distance(ground_truth,qasm_noise_na_prob)))
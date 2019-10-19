from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.circuit.classicalregister import ClassicalRegister
from qiskit.transpiler.passes import NoiseAdaptiveLayout
from qiskit import QuantumCircuit
from qiskit import Aer, IBMQ, execute
from qiskit.compiler import transpile
from qiskit.providers.aer import noise
import numpy as np

def cross_entropy(target,obs):
    assert len(target)==len(obs)
    # obs = [x if x>=0 else 0 for x in obs]
    alpha = 1e-14
    if 0 in obs:
        obs = [(x+alpha)/(1+alpha*len(obs)) for x in obs]
    h = 0
    for p,q in zip(target,obs):
        if p==0:
            h += 0
        else:
            h += -p*np.log(q)
    return h

def find_saturated_shots(circ):
    ground_truth = simulate_circ(circ=circ,backend='statevector_simulator',qasm_info=None)
    min_ce = cross_entropy(target=ground_truth,obs=ground_truth)
    num_shots = 1000
    while 1:
        qasm = simulate_circ(circ=circ,backend='noiseless_qasm_simulator',qasm_info=(None,None,None,None,None,num_shots))
        # NOTE: toggle here to control cross entropy accuracy
        if abs(cross_entropy(target=ground_truth,obs=qasm)-min_ce)/min_ce<1e-3:
            return num_shots
        else:
            num_shots *= 2

def reverseBits(num,bitSize): 
    binary = bin(num)
    reverse = binary[-1:1:-1] 
    reverse = reverse + (bitSize - len(reverse))*'0'
    return int(reverse,2)

def simulate_circ(circ, backend, qasm_info):
    if backend == 'statevector_simulator':
        # print('using statevector simulator')
        backend = Aer.get_backend('statevector_simulator')
        job = execute(circ, backend=backend)
        result = job.result()
        outputstate = result.get_statevector(circ)
        outputstate_ordered = [0 for sv in outputstate]
        for i, sv in enumerate(outputstate):
            reverse_i = reverseBits(i,len(circ.qubits))
            outputstate_ordered[reverse_i] = sv
        output_prob = [np.power(np.absolute(x),2) for x in outputstate_ordered]
        return output_prob
    # TODO: add saturated shots
    elif backend == 'noiseless_qasm_simulator':
        # print('using noiseless qasm simulator %d shots'%num_shots)
        backend = Aer.get_backend('qasm_simulator')
        c = ClassicalRegister(len(circ.qubits), 'c')
        meas = QuantumCircuit(circ.qregs[0], c)
        meas.barrier(circ.qubits)
        meas.measure(circ.qubits,c)
        qc = circ+meas

        _,_,_,_,_,num_shots = qasm_info
        job_sim = execute(qc, backend, shots=num_shots)
        result = job_sim.result()
        noiseless_counts = result.get_counts(qc)
        noiseless_prob = [0 for x in range(np.power(2,len(circ.qubits)))]
        for state in noiseless_counts:
            reversed_state = reverseBits(int(state,2),len(circ.qubits))
            noiseless_prob[reversed_state] = noiseless_counts[state]/num_shots
        return noiseless_prob
    elif backend == 'noisy_qasm_simulator':
        # print('using noisy qasm simulator {} shots, NA = {}'.format(num_shots,initial_layout!=None))
        backend = Aer.get_backend('qasm_simulator')
        c = ClassicalRegister(len(circ.qubits), 'c')
        meas = QuantumCircuit(circ.qregs[0], c)
        meas.barrier(circ.qubits)
        meas.measure(circ.qubits,c)
        qc = circ+meas
        device,properties,coupling_map,noise_model,basis_gates,num_shots = qasm_info
        dag = circuit_to_dag(qc)
        noise_mapper = NoiseAdaptiveLayout(properties)
        noise_mapper.run(dag)
        initial_layout = noise_mapper.property_set['layout']
        new_circuit = transpile(qc, backend=device, basis_gates=basis_gates,coupling_map=coupling_map,backend_properties=properties,initial_layout=initial_layout)
        # FIXME: Do I need to pass initial_layout?
        na_result = execute(experiments=new_circuit,
        backend=backend,
        noise_model=noise_model,
        coupling_map=coupling_map,
        basis_gates=basis_gates,
        shots=num_shots).result()
        na_counts = na_result.get_counts(new_circuit)
        na_prob = [0 for x in range(np.power(2,len(circ.qubits)))]
        for state in na_counts:
            reversed_state = reverseBits(int(state,2),len(circ.qubits))
            na_prob[reversed_state] = na_counts[state]/num_shots
        return na_prob
    else:
        raise Exception('Illegal backend:',backend)
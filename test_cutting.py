import qiskit
import qsplit_circuit_cutter as qcc
import qsplit_mlrecon_methods as qmm
from utils import cutting_funcs

# Build a simple circuit
circ = qiskit.QuantumCircuit(3, name='q')
circ.h(0)
circ.cx(0,1)
circ.cx(1,2)

# Cut it after the first CNOT on qubit 1
cuts = [(qiskit.circuit.Qubit(qiskit.QuantumRegister(3,'q'), 1), 1)]
print('Cuts:', cuts)
print('Full circ:')
print(circ)

# Cut the circuit
fragments, wire_path_map = qcc.cut_circuit(circ, cuts)
for i, frag in enumerate(fragments):
    print('Fragment', i+1)
    print(frag)

# Simulate and Recombine
shots = 100000
frag_shots = shots // qmm.fragment_variants(wire_path_map)
print('frag shots:', frag_shots)
backend = qiskit.Aer.get_backend('statevector_simulator')
recombined_dist = cutting_funcs.sim_with_cutting(fragments, wire_path_map, frag_shots, backend)
print(recombined_dist)

# Evaluate the original circuit for comparison
result = qiskit.execute(circ, backend=backend).result()
sv = result.get_statevector(circ)
exact_dist = qiskit.quantum_info.Statevector(sv).probabilities_dict(decimals=5)
print()
print(exact_dist)

# Compute the Hellinger Fidelity
print('Hellinger fidelity:', qiskit.quantum_info.hellinger_fidelity(recombined_dist, exact_dist))

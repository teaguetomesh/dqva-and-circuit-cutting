from qiskit import Aer, IBMQ, execute
from qiskit.compiler import transpile, assemble

from utils.helper_fun import generate_circ, apply_measurement
from utils.conversions import dict_to_array

if __name__ == '__main__':

    circ0 = generate_circ(full_circ_size=9,circuit_type='supremacy_grid')
    qc0 = apply_measurement(circ0)

    circ1 = generate_circ(full_circ_size=6,circuit_type='bv')
    qc1 = apply_measurement(circ1)

    backend = Aer.get_backend('qasm_simulator')
    backend_options = {'max_memory_mb': 2**30*16/1024**2}
    noiseless_qasm_result = execute([qc0,qc1], backend, shots=10,backend_options=backend_options).result()
    noiseless_counts = noiseless_qasm_result.get_counts(0)
    print(noiseless_counts)
    noiseless_counts = noiseless_qasm_result.get_counts(1)
    print(noiseless_counts)
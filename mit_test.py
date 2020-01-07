from qiskit import QuantumRegister, QuantumCircuit, ClassicalRegister, Aer, execute
from qiskit.ignis.mitigation.measurement import (complete_meas_cal, tensored_meas_cal,CompleteMeasFitter, TensoredMeasFitter)
from qiskit.compiler import transpile, assemble
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.extensions.standard import HGate, SGate, SdgGate, XGate, RXGate, RYGate, RZGate
from utils.helper_fun import generate_circ, get_evaluator_info, reverseBits
from utils.mitigation import TensoredMitigation
import copy
import math

full_circ_size = 5
circ = generate_circ(full_circ_size=full_circ_size,circuit_type='supremacy')
device_name = 'ibmq_boeblingen'
evaluator_info = get_evaluator_info(circ=circ,device_name=device_name,
        fields=['device','basis_gates','coupling_map','properties','initial_layout','noise_model'])
device_max_experiments = evaluator_info['device'].configuration().max_experiments
device_max_shots = evaluator_info['device'].configuration().max_shots

qr = QuantumRegister(len(evaluator_info['properties'].qubits))

mit_pattern = []
qubit_group = []
_initial_layout = evaluator_info['initial_layout'].get_physical_bits()
for q in _initial_layout:
    if 'ancilla' not in _initial_layout[q].register.name and 2**(len(qubit_group)+1)<=device_max_experiments:
        qubit_group.append(q)
    else:
        mit_pattern.append(qubit_group)
        qubit_group = [q]
if len(qubit_group)>0:
    mit_pattern.append(qubit_group)

meas_calibs, state_labels = tensored_meas_cal(mit_pattern=mit_pattern, qr=qr, circlabel='')

# modified_circ = meas_calibs[0]
# modified_dag = circuit_to_dag(modified_circ)
# modified_dag.apply_operation_front(op=RXGate(theta=math.pi/2),qargs=[modified_circ.qubits[mit_pattern[0][0]]],cargs=[])
# modified_circ = dag_to_circuit(modified_dag)
# meas_calibs[0] = modified_circ

meas_calibs_transpiled = transpile(meas_calibs, backend=evaluator_info['device'])
qobj = assemble(meas_calibs_transpiled, backend=evaluator_info['device'], shots=device_max_shots)
# print('Qiskit tensored mitigation mit_pattern:',mit_pattern)

backend = Aer.get_backend('qasm_simulator')
job = Aer.get_backend('qasm_simulator').run(qobj)
cal_results = job.result()
meas_fitter = TensoredMeasFitter(cal_results, mit_pattern=mit_pattern)
calibration_matrix = copy.deepcopy(meas_fitter.cal_matrices[0])
for row_idx in range(2**full_circ_size):
    reversed_row_idx = reverseBits(num=row_idx,bitSize=full_circ_size)
    for col_idx in range(2**full_circ_size):
        reversed_col_idx = reverseBits(num=col_idx,bitSize=full_circ_size)
        calibration_matrix[reversed_row_idx][reversed_col_idx] = meas_fitter.cal_matrices[0][row_idx][col_idx]

circ_dict = {'test':{'circ':circ}}
tensored_mitigation = TensoredMitigation(circ_dict=circ_dict,device_name=device_name)
tensored_mitigation.run()
tensored_mitigation.retrieve()
circ_dict = tensored_mitigation.circ_dict
calibration_matrix = circ_dict['test']['calibration_matrix']
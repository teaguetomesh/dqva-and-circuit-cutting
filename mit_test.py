from qiskit import QuantumRegister, QuantumCircuit, ClassicalRegister, Aer, execute
from qiskit.ignis.mitigation.measurement import (complete_meas_cal, tensored_meas_cal,CompleteMeasFitter, TensoredMeasFitter)
from utils.helper_fun import generate_circ, get_evaluator_info
from utils.mitigation import TensoredMitigation
import numpy as np

np.random.seed(1234)
full_circ_size = 2
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
meas_calibs, state_labels = tensored_meas_cal(mit_pattern=mit_pattern, qr=qr, circlabel='mcal')
print('Qiskit tensored mitigation mit_pattern:',mit_pattern)

backend = Aer.get_backend('qasm_simulator')
job = execute(meas_calibs, backend=backend, shots=device_max_shots, noise_model=evaluator_info['noise_model'])
# job = execute(meas_calibs, backend=backend, shots=device_max_shots)
cal_results = job.result()
meas_fitter = TensoredMeasFitter(cal_results, mit_pattern=mit_pattern)
print(meas_fitter.cal_matrices[0])

circ_dict = {'test':{'circ':circ}}
tensored_mitigation = TensoredMitigation(circ_dict=circ_dict,device_name=device_name)
tensored_mitigation.run()
tensored_mitigation.retrieve()
circ_dict = tensored_mitigation.circ_dict
calibration_matrix = circ_dict['test']['calibration_matrix']
print(calibration_matrix)
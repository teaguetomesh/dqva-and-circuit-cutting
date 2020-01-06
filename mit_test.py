from qiskit import QuantumRegister, QuantumCircuit, ClassicalRegister, Aer, execute
from qiskit.ignis.mitigation.measurement import (complete_meas_cal, tensored_meas_cal,CompleteMeasFitter, TensoredMeasFitter)
from utils.helper_fun import generate_circ, get_evaluator_info
from utils.mitigation import TensoredMitigation

full_circ_size = 2
device_name = 'ibmq_boeblingen'

qr = QuantumRegister(full_circ_size)
mit_pattern = [range(full_circ_size)]
meas_calibs, state_labels = tensored_meas_cal(mit_pattern=mit_pattern, qr=qr, circlabel='mcal')

evaluator_info = get_evaluator_info(circ=None,device_name=device_name,
        fields=['device','basis_gates','coupling_map','properties','noise_model'])

backend = Aer.get_backend('qasm_simulator')
job = execute(meas_calibs, backend=backend, shots=8192, noise_model=evaluator_info['noise_model'])
cal_results = job.result()
meas_fitter = TensoredMeasFitter(cal_results, mit_pattern=mit_pattern)
print(meas_fitter.cal_matrices[0])

circ = generate_circ(full_circ_size=full_circ_size,circuit_type='supremacy')
circ_dict = {'test':{'circ':circ}}
tensored_mitigation = TensoredMitigation(circ_dict=circ_dict,device_name=device_name)
tensored_mitigation.run()
tensored_mitigation.retrieve()
circ_dict = tensored_mitigation.circ_dict
calibration_matrix = circ_dict['test']['calibration_matrix']
print(calibration_matrix)
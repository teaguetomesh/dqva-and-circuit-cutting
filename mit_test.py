from qiskit import QuantumRegister, QuantumCircuit, ClassicalRegister, Aer, execute
from qiskit.ignis.mitigation.measurement import (complete_meas_cal, tensored_meas_cal,CompleteMeasFitter, TensoredMeasFitter)
from qiskit.compiler import transpile, assemble
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.extensions.standard import HGate, SGate, SdgGate, XGate, RXGate, RYGate, RZGate
from qiskit.tools.visualization import plot_histogram
from qiskit.providers.aer import noise
from utils.helper_fun import generate_circ, get_evaluator_info, reverseBits, apply_measurement, evaluate_circ, reverseBits
from utils.mitigation import TensoredMitigation
from utils.submission import Scheduler
import copy
import math

device_name = 'ibmq_boeblingen'
evaluator_info = get_evaluator_info(circ=None,device_name='ibmq_boeblingen',fields=['basis_gates'])
# Make a 3Q GHZ state
full_circ_size = 3
qr = QuantumRegister(full_circ_size)
ghz = generate_circ(full_circ_size=full_circ_size,circuit_type='supremacy')
ground_truth = evaluate_circ(circ=ghz,backend='statevector_simulator',evaluator_info=None,reverse=True)
ground_truth_dict = {}
for i, p in enumerate(ground_truth):
    bin_i = bin(i)[2:].zfill(full_circ_size)
    ground_truth_dict[bin_i] = p

ghz = transpile(ghz, basis_gates=evaluator_info['basis_gates'])
ghz = apply_measurement(circ=ghz)
print(ghz)

# Generate a noise model for the 5 qubits
noise_model = noise.NoiseModel()
for qi in range(full_circ_size):
    read_err = noise.errors.readout_error.ReadoutError([[0.9, 0.1],[0.25,0.75]])
    noise_model.add_readout_error(read_err, [qi])

backend = Aer.get_backend('qasm_simulator')
job = execute([ghz], backend=backend, shots=5000, noise_model=noise_model)
results = job.result()

# Results without mitigation
raw_counts = results.get_counts()

# Generate the calibration circuits
mit_pattern = [range(full_circ_size)]
meas_calibs, state_labels = tensored_meas_cal(mit_pattern=mit_pattern, qr=qr, circlabel='mcal')
# Execute the calibration circuits
backend = Aer.get_backend('qasm_simulator')
job = execute(meas_calibs, backend=backend, shots=5000, noise_model=noise_model)
cal_results = job.result()
meas_fitter = TensoredMeasFitter(cal_results, mit_pattern=mit_pattern)

# Get the filter object
meas_filter = meas_fitter.filter

# Results with mitigation
mitigated_results = meas_filter.apply(results,method='least_squares')
mitigated_counts = mitigated_results.get_counts(0)
mitigated_counts_dict = {}
for i in mitigated_counts:
    p = mitigated_counts[i]
    reverse_i = reverseBits(int(i,2),full_circ_size)
    bin_i = bin(reverse_i)[2:].zfill(full_circ_size)
    mitigated_counts_dict[bin_i] = p

circ_dict = {'test':{'circ':ghz,'shots':5000}}

tensored_mitigation = TensoredMitigation(circ_dict=circ_dict,device_name=device_name)
tensored_mitigation.run()

scheduler = Scheduler(circ_dict=circ_dict,device_name=device_name)
scheduler.run()

scheduler.retrieve()
tensored_mitigation.retrieve()
tensored_mitigation.apply(unmitigated=scheduler.circ_dict)
mitigated_circ_dict = tensored_mitigation.circ_dict
print(mitigated_circ_dict['test'].keys())
my_raw = mitigated_circ_dict['test']['hw']
my_mitigated = mitigated_circ_dict['test']['mitigated_hw']

my_raw_dict = {}
for i, p in enumerate(my_raw):
    bin_i = bin(i)[2:].zfill(full_circ_size)
    my_raw_dict[bin_i] = p
my_mitigated_dict = {}
for i, p in enumerate(my_mitigated):
    bin_i = bin(i)[2:].zfill(full_circ_size)
    my_mitigated_dict[bin_i] = p

plot_histogram([raw_counts, mitigated_counts_dict, ground_truth_dict], legend=['raw','mitigated','truth'])

plot_histogram([my_raw_dict, my_mitigated_dict,ground_truth_dict], legend=['my_raw', 'my_mitigated','truth'])

plot_histogram([mitigated_counts_dict, my_mitigated_dict], legend=['mitigated', 'my_mitigated'])
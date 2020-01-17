from qiskit import QuantumRegister, QuantumCircuit, ClassicalRegister, Aer, execute
from qiskit.ignis.mitigation.measurement import (complete_meas_cal, tensored_meas_cal,CompleteMeasFitter, TensoredMeasFitter)
from qiskit.compiler import transpile, assemble
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.extensions.standard import HGate, SGate, SdgGate, XGate, RXGate, RYGate, RZGate
from qiskit.tools.visualization import plot_histogram
from qiskit.providers.aer import noise
from utils.helper_fun import generate_circ, get_evaluator_info, reverseBits
from utils.mitigation import TensoredMitigation
from utils.submission import Scheduler
import copy
import math

# Make a 3Q GHZ state
qr = QuantumRegister(5)
cr = ClassicalRegister(5)
ghz = QuantumCircuit(qr, cr)
ghz.h(qr[2])
ghz.cx(qr[2], qr[3])
ghz.cx(qr[3], qr[4])
ghz.measure(qr,cr)
print(ghz)

# Generate a noise model for the 5 qubits
noise_model = noise.NoiseModel()
for qi in range(5):
    read_err = noise.errors.readout_error.ReadoutError([[0.9, 0.1],[0.25,0.75]])
    noise_model.add_readout_error(read_err, [qi])

backend = Aer.get_backend('qasm_simulator')
job = execute([ghz], backend=backend, shots=5000, noise_model=noise_model)
results = job.result()

# Results without mitigation
raw_counts = results.get_counts()

# Generate the calibration circuits
mit_pattern = [[0,1,2,3,4]]
meas_calibs, state_labels = tensored_meas_cal(mit_pattern=mit_pattern, qr=qr, circlabel='mcal')
# Execute the calibration circuits
backend = Aer.get_backend('qasm_simulator')
job = execute(meas_calibs, backend=backend, shots=5000, noise_model=noise_model)
cal_results = job.result()
meas_fitter = TensoredMeasFitter(cal_results, mit_pattern=mit_pattern)

# Get the filter object
meas_filter = meas_fitter.filter

# Results with mitigation
mitigated_results = meas_filter.apply(results)
mitigated_counts = mitigated_results.get_counts(0)

device_name = 'ibmq_boeblingen'
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

plot_histogram([raw_counts, mitigated_counts, mitigated_circ_dict['test']['hw']], legend=['raw', 'mitigated', 'my_raw','my_mitigated'])
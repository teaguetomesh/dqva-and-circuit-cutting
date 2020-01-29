from qiskit import QuantumRegister, QuantumCircuit, ClassicalRegister, Aer, execute
from qiskit.ignis.mitigation.measurement import (complete_meas_cal, tensored_meas_cal,CompleteMeasFitter, TensoredMeasFitter)
from qiskit.compiler import transpile, assemble
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.extensions.standard import HGate, SGate, SdgGate, XGate, RXGate, RYGate, RZGate
from qiskit.tools.visualization import plot_histogram
from qiskit.providers.aer import noise
from utils.helper_fun import generate_circ, get_evaluator_info, apply_measurement, evaluate_circ
from utils.metrics import kl_divergence, chi2_distance
from scipy.stats import wasserstein_distance
from utils.mitigation import TensoredMitigation
from utils.schedule import Scheduler
from utils.conversions import list_to_dict, dict_to_array
import copy
import math
import numpy as np
import matplotlib.pyplot as plt

def compute_metrics(ground_truth,raw_counts,mitigated_counts,metric):
    if metric == 'ce':
        truth_ce = kl_divergence(target=dict_to_array(distribution_dict=ground_truth,force_prob=True),
        obs=dict_to_array(distribution_dict=ground_truth,force_prob=True))
        raw_ce = kl_divergence(target=dict_to_array(distribution_dict=ground_truth,force_prob=True),
        obs=dict_to_array(distribution_dict=raw_counts,force_prob=True))
        mit_ce = kl_divergence(target=dict_to_array(distribution_dict=ground_truth,force_prob=True),
        obs=dict_to_array(distribution_dict=mitigated_counts,force_prob=True))
        return truth_ce, raw_ce, mit_ce
    elif metric == 'chi2':
        truth_chi2 = chi2_distance(target=dict_to_array(distribution_dict=ground_truth,force_prob=True),
        obs=dict_to_array(distribution_dict=ground_truth,force_prob=True))
        raw_chi2 = chi2_distance(target=dict_to_array(distribution_dict=ground_truth,force_prob=True),
        obs=dict_to_array(distribution_dict=raw_counts,force_prob=True))
        mit_chi2 = chi2_distance(target=dict_to_array(distribution_dict=ground_truth,force_prob=True),
        obs=dict_to_array(distribution_dict=mitigated_counts,force_prob=True))
        return truth_chi2, raw_chi2, mit_chi2
    elif metric == 'wasserstein_distance':
        truth_distance = wasserstein_distance(u_values=dict_to_array(distribution_dict=ground_truth,force_prob=True),
        v_values=dict_to_array(distribution_dict=ground_truth,force_prob=True))
        raw_distance = wasserstein_distance(u_values=dict_to_array(distribution_dict=ground_truth,force_prob=True),
        v_values=dict_to_array(distribution_dict=raw_counts,force_prob=True))
        mit_distance = wasserstein_distance(u_values=dict_to_array(distribution_dict=ground_truth,force_prob=True),
        v_values=dict_to_array(distribution_dict=mitigated_counts,force_prob=True))
        return truth_distance, raw_distance, mit_distance
    else:
        raise Exception('gg')

def plot_bar(data,legends,title):
    nqubits = len(list(data[0].keys())[0])
    labels = [bin(state)[2:].zfill(nqubits) for state in range(2**nqubits)]
    x = np.arange(len(labels))
    width = 1/(len(data)+2)

    fig, ax = plt.subplots(figsize=(30,10))
    counter = -int(len(data)/2)
    for datum, legend in zip(data,legends):
        datum = dict_to_array(distribution_dict=datum,force_prob=True)
        rects = ax.bar(x + counter*width, datum, width, label=legend)
        counter += 1
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels,rotation='vertical')
    ax.legend()
    fig.tight_layout()

    fig.savefig('%s.png'%title)
    return

device_name = 'ibmq_boeblingen'
evaluator_info = get_evaluator_info(circ=None,device_name='ibmq_boeblingen',fields=['basis_gates','properties'])
device_qubits = len(evaluator_info['properties'].qubits)

full_circ_size = 4
ghz = generate_circ(full_circ_size=full_circ_size,circuit_type='supremacy')
ground_truth = evaluate_circ(circ=ghz,backend='statevector_simulator',evaluator_info=None,force_prob=True)
print('Ground truth:',ground_truth)
qreg = ghz.qregs[0]

ghz_transpiled = apply_measurement(circ=ghz)
ghz_transpiled = transpile(ghz_transpiled, basis_gates=evaluator_info['basis_gates'],initial_layout={qreg[x]:x for x in range(full_circ_size)})
print('Generated circuit:')
print(ghz_transpiled)

# Generate a noise model for the qubits
noise_model = noise.NoiseModel()
for qi in range(device_qubits):
    if qi < 0:
        read_err = noise.errors.readout_error.ReadoutError([[1, 0],[0, 1]])
    else:
        read_err = noise.errors.readout_error.ReadoutError([[0.93, 1-0.93],[1-0.89, 0.89]])
    noise_model.add_readout_error(read_err, [qi])

backend = Aer.get_backend('qasm_simulator')
qobj = assemble(ghz_transpiled, backend=backend, shots=500000)
job = Aer.get_backend('qasm_simulator').run(qobj,noise_model=noise_model)
results = job.result()

# Results without mitigation
raw_counts = results.get_counts()
# print('Qiskit raw counts:',raw_counts)

# Generate the calibration circuits
mit_pattern = [range(full_circ_size)]
print('Qiskit mit_pattern = ',mit_pattern)
qr = QuantumRegister(full_circ_size)
meas_calibs, state_labels = tensored_meas_cal(mit_pattern=mit_pattern, qr=qr, circlabel='mcal')
# Execute the calibration circuits
backend = Aer.get_backend('qasm_simulator')
qobj = assemble(meas_calibs, backend=backend, shots=8192)
job = Aer.get_backend('qasm_simulator').run(qobj,noise_model=noise_model)
cal_results = job.result()
meas_fitter = TensoredMeasFitter(cal_results, mit_pattern=mit_pattern)

# Get the filter object
meas_filter = meas_fitter.filter

# Results with mitigation
mitigated_results = meas_filter.apply(results,method='least_squares')
mitigated_counts = mitigated_results.get_counts(0)
# print('Qiskit mitigated counts:',mitigated_counts)

truth_metric, qiskit_raw_metric, qiskit_mit_metric = compute_metrics(ground_truth=ground_truth,raw_counts=raw_counts,mitigated_counts=mitigated_counts,metric='chi2')
print('Qiskit metric: {:.3e}-->{:.3e}'.format(qiskit_raw_metric,qiskit_mit_metric))

fig = plot_histogram([raw_counts, mitigated_counts, ground_truth], legend=['raw = %.3e'%qiskit_raw_metric,'mitigated = %.3e'%qiskit_mit_metric,'truth = %.3e'%truth_metric],figsize=(35,10),title='qiskit mitigation')
fig.savefig('qiskit_mitigation.png')

circ_dict = {'test':{'circ':ghz,'shots':500000}}

mitigation_correspondence_dict = {'test':['test']}

scheduler = Scheduler(circ_dict=circ_dict,device_name=device_name)
scheduler.run(real_device=False)

tensored_mitigation = TensoredMitigation(circ_dict=circ_dict,device_name=device_name)
tensored_mitigation.run(real_device=False)

scheduler.retrieve(force_prob=True)
tensored_mitigation.retrieve()
tensored_mitigation.apply(unmitigated=scheduler.circ_dict,mitigation_correspondence_dict=mitigation_correspondence_dict)
mitigated_circ_dict = tensored_mitigation.circ_dict

print(mitigated_circ_dict['test'].keys())
my_raw = mitigated_circ_dict['test']['hw']
my_mitigated = mitigated_circ_dict['test']['mitigated_hw']
my_raw_dict = list_to_dict(l=list(my_raw))
my_mitigated_dict = list_to_dict(l=list(my_mitigated))

truth_ce, my_raw_ce, my_mit_ce = compute_metrics(ground_truth=ground_truth,raw_counts=my_raw_dict,mitigated_counts=my_mitigated_dict,metric='wasserstein_distance')
print('My metric: {:.3e}-->{:.3e}'.format(my_raw_ce,my_mit_ce))

fig = plot_histogram([my_raw_dict, my_mitigated_dict, ground_truth], legend=['my_raw = %.3e'%my_raw_ce,'my_mitigated = %.3e'%my_mit_ce,'truth = %.3e'%truth_ce],figsize=(35,10),title='my mitigation')
fig.savefig('my_mitigation.png')

# fig = plot_histogram([mitigated_counts, my_mitigated_dict, ground_truth], legend=['mitigated', 'my_mitigated', 'truth'],figsize=(35,10),title='mitigations comparison')
# fig.savefig('mitigations.png')
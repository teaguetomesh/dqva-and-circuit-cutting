from utils.helper_fun import generate_circ, get_evaluator_info, evaluate_circ, apply_measurement
from utils.metrics import fidelity
from utils.conversions import dict_to_array
from utils.schedule import Scheduler
from qiskit import execute, Aer


circ = generate_circ(full_circ_size=10,circuit_type='bv')
num_shots = 81920
evaluator_info = get_evaluator_info(circ=circ,device_name='ibmq_boeblingen',fields=['device','basis_gates',
    'coupling_map','properties','initial_layout','noise_model'])
evaluator_info['num_shots'] = num_shots
print(evaluator_info['initial_layout'])

ground_truth = evaluate_circ(circ=circ,backend='statevector_simulator',evaluator_info=None,force_prob=True)
ground_truth = dict_to_array(distribution_dict=ground_truth,force_prob=True)
noisy = evaluate_circ(circ=circ,backend='noisy_qasm_simulator',evaluator_info=evaluator_info,force_prob=True)
noisy = dict_to_array(distribution_dict=noisy,force_prob=True)
print('noisy sim fidelity = ',fidelity(target=ground_truth,obs=noisy))

circ_dict = {}
circ_dict['test'] = {'circ':circ,'shots':num_shots,'initial_layout':evaluator_info['initial_layout']}
scheduler = Scheduler(circ_dict=circ_dict,device_name='ibmq_boeblingen')
scheduler.run(real_device=False)
scheduler.retrieve(force_prob=False)
circ_dict = scheduler.circ_dict
print('my hw fidelity = ', fidelity(target=ground_truth,obs=circ_dict['test']['hw']))

qc = apply_measurement(circ)
# result = execute([qc for x in range(10)],backend=Aer.get_backend('qasm_simulator'),shots=8192).result()
result = execute([qc for x in range(10)],backend=evaluator_info['device'],initial_layout=evaluator_info['initial_layout'],shots=8192).result()
all_counts = {}
for i in range(10):
    counts = result.get_counts(i)
    for state in counts:
        if state in all_counts:
            all_counts[state] += counts[state]
        else:
            all_counts[state] = counts[state]
print(sum(all_counts.values()))
qiskit_prob = dict_to_array(all_counts,force_prob=True)
print('qiskit hw fidelity = ', fidelity(target=ground_truth,obs=qiskit_prob))
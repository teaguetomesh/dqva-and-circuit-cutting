import numpy as np
from time import time
import copy
from qiskit.circuit.quantumregister import QuantumRegister
from utils.helper_fun import get_evaluator_info
from qiskit.ignis.mitigation.measurement import tensored_meas_cal
from utils.submission import Scheduler
from qiskit.compiler import transpile

def break_state(bin_state,mit_pattern):
    start_idx = 0
    bin_state_parts = []
    for qubit_group in mit_pattern:
        end_idx = start_idx + len(qubit_group)
        bin_state_part = bin_state[start_idx:end_idx]
        bin_state_parts.append(bin_state_part)
        start_idx = end_idx
    assert sum([len(x) for x in bin_state_parts]) == len(bin_state)
    return bin_state_parts

class TensoredMitigation:
    def __init__(self,circ_dict,device_name):
        self.circ_dict = circ_dict
        self.device_name = device_name
        self.check_status()
        self.meas_calibs_dict = self.get_mitigation_circuits()
        self.scheduler = Scheduler(circ_dict=self.meas_calibs_dict,device_name=self.device_name)

    def check_status(self):
        assert isinstance(self.circ_dict,dict)
        for key in self.circ_dict:
            value = self.circ_dict[key]
            if 'circ' not in value:
                raise Exception('Input circ_dict does not have circ for key {}'.format(key))
        try:
            evaluator_info = get_evaluator_info(circ=None,device_name=self.device_name,fields=['device','properties'])
        except:
            raise Exception('Illegal input device : {}'.format(self.device_name))
    
    def get_mitigation_circuits(self):
        meas_calibs_dict = {}
        for key in self.circ_dict:
            circ = self.circ_dict[key]['circ']
            evaluator_info = get_evaluator_info(circ=circ,device_name=self.device_name,
            fields=['device','basis_gates','coupling_map','properties','initial_layout'])
            device_max_shots = evaluator_info['device'].configuration().max_shots
            device_max_experiments = evaluator_info['device'].configuration().max_experiments
            num_qubits = len(evaluator_info['properties'].qubits)
            qr = QuantumRegister(num_qubits)
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
            print('Circuit %s has mit_pattern:'%key,mit_pattern)
            self.circ_dict[key]['mit_pattern'] = mit_pattern
            meas_calibs, state_labels = tensored_meas_cal(mit_pattern=mit_pattern, qr=qr, circlabel='')
            meas_calibs_transpiled = transpile(meas_calibs, backend=evaluator_info['device'])
            for meas_calib_circ in meas_calibs_transpiled:
                meas_calibs_dict_key = '%s|%s'%(key,meas_calib_circ.name.split('_')[1][::-1])
                assert meas_calibs_dict_key not in meas_calibs_dict
                meas_calibs_dict.update({meas_calibs_dict_key:{'circ':meas_calib_circ,'shots':device_max_shots}})
                # print(meas_calibs_dict_key)
        return meas_calibs_dict

    def run(self,real_device=False):
        self.scheduler.run(real_device=real_device)

    def retrieve(self):
        self.scheduler.retrieve()
        for key in self.circ_dict:
            circ = self.circ_dict[key]['circ']
            num_qubits = len(circ.qubits)
            mit_pattern = self.circ_dict[key]['mit_pattern']
            # print('Circuit %s'%key)
            perturbation_probabilities = [[0]*4**len(qubit_group) for qubit_group in mit_pattern]
            for meas_calibs_dict_key in self.scheduler.circ_dict:
                if meas_calibs_dict_key.split('|')[0]==key:
                    full_actual = meas_calibs_dict_key.split('|')[1]
                    qubit_group_actual_states = break_state(bin_state=full_actual,mit_pattern=mit_pattern)
                    # print('Qubit group actual states:',qubit_group_actual_states)
                    measured = self.scheduler.circ_dict[meas_calibs_dict_key]['hw']
                    for meas_state, p in enumerate(measured):
                        bin_meas_state = bin(meas_state)[2:].zfill(num_qubits)
                        qubit_group_meas_states = break_state(bin_state=bin_meas_state,mit_pattern=mit_pattern)
                        # print('Qubit group measured states: {}, p = {:.3e}'.format(qubit_group_meas_states,p))
                        for qubit_group_idx in range(len(mit_pattern)):
                            qubit_group_actual_state = qubit_group_actual_states[qubit_group_idx]
                            qubit_group_meas_state = qubit_group_meas_states[qubit_group_idx]
                            calibration_matrix_entry_key = ''
                            for a, m in zip(qubit_group_actual_state,qubit_group_meas_state):
                                calibration_matrix_entry_key += '%s%s'%(a,m)
                            perturbation_idx = int(calibration_matrix_entry_key,2)
                            perturbation_probabilities[qubit_group_idx][perturbation_idx] += p
            for qubit_group_idx, qubit_group in enumerate(mit_pattern):
                num_repetitions = 2**(max([len(x) for x in mit_pattern]) - len(qubit_group))
                if num_repetitions>1:
                    for perturbation_idx in range(4**len(qubit_group)):
                        perturbation_probabilities[qubit_group_idx][perturbation_idx] /= num_repetitions
            # print('mit_pattern = {} perturbation_probabilities = {}'.format(mit_pattern,perturbation_probabilities))
            self.circ_dict[key]['calibration_matrix'] = self.get_calibration_matrix(perturbation_probabilities=perturbation_probabilities)
    
    def get_calibration_matrix(self,perturbation_probabilities):
        num_qubits = int(sum([np.log(len(x))/np.log(4) for x in perturbation_probabilities]))
        begin = time()
        base = [1]
        calibration_matrix = np.zeros(shape=(2**num_qubits,2**num_qubits))
        for qubit_perturbation in perturbation_probabilities:
            base = np.kron(base,qubit_perturbation)
        for meas in range(2**num_qubits):
            meas_state = bin(meas)[2:].zfill(num_qubits)
            for actual in range(2**num_qubits):
                actual_state = bin(actual)[2:].zfill(num_qubits)
                binary_position_str = ''
                for a,m in zip(actual_state,meas_state):
                    binary_position_str += '%s%s'%(a,m)
                position = int(binary_position_str,2)
                calibration_matrix[meas][actual] = base[position]
        print('Computing calibration matrix for %d qubit circuit took %.3e seconds'%(num_qubits,time()-begin))
        return calibration_matrix

    def apply(self,unmitigated):
        mitigated = {}
        for key in unmitigated:
            assert key in self.circ_dict
            mitigated[key] = copy.deepcopy(unmitigated[key])
            calibration_matrix = self.circ_dict[key]['calibration_matrix']
            filter_matrix = np.linalg.inv(calibration_matrix)
            unmitigated_prob = np.reshape(unmitigated[key]['hw'],(-1,1))
            mitigated_prob = np.reshape(filter_matrix.dot(unmitigated_prob),(1,-1)).tolist()[0]
            assert abs(sum(mitigated_prob)-1)<1e-10
            mitigated[key]['mitigated_hw'] = copy.deepcopy(mitigated_prob)
        self.circ_dict = copy.deepcopy(mitigated)
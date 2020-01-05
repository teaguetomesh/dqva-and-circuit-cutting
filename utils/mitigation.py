import numpy as np
from time import time
from qiskit.circuit.quantumregister import QuantumRegister
from utils.helper_fun import get_evaluator_info
from qiskit.ignis.mitigation.measurement import tensored_meas_cal
from utils.submission import Scheduler
from qiskit.compiler import transpile

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
            num_qubits = len(evaluator_info['properties'].qubits)
            qr = QuantumRegister(num_qubits)
            mit_pattern = []
            _initial_layout = evaluator_info['initial_layout'].get_physical_bits()
            for q in _initial_layout:
                if 'ancilla' not in _initial_layout[q].register.name:
                    mit_pattern.append([q])
            meas_calibs, state_labels = tensored_meas_cal(mit_pattern=mit_pattern, qr=qr, circlabel='')
            meas_calibs_transpiled = transpile(meas_calibs, backend=evaluator_info['device'])
            for meas_calib_circ in meas_calibs_transpiled:
                meas_calibs_dict.update({'%s|%s'%(key,meas_calib_circ.name.split('_')[1]):{'circ':meas_calib_circ,'shots':device_max_shots}})
        return meas_calibs_dict

    def run(self,real_device=False):
        self.scheduler.run(real_device=real_device)

    def retrieve(self):
        self.scheduler.retrieve()
        self.calibration_matrices = {}
        for key in self.circ_dict:
            circ = self.circ_dict[key]['circ']
            num_qubits = len(circ.qubits)
            perturbation_probabilities = [[0,0,0,0] for i in range(num_qubits)]
            meas_calib_0_key = '%s|%s'%(key,'0'*num_qubits)
            meas_calib_0_prob = self.scheduler.circ_dict[meas_calib_0_key]['hw']
            for state, prob in enumerate(meas_calib_0_prob):
                bin_state = bin(state)[2:].zfill(num_qubits)
                for qubit_idx, b in enumerate(bin_state):
                    b = int(b)
                    perturbation_probabilities[qubit_idx][b] += prob
            
            meas_calib_1_key = '%s|%s'%(key,'1'*num_qubits)
            meas_calib_1_prob = self.scheduler.circ_dict[meas_calib_1_key]['hw']
            for state, prob in enumerate(meas_calib_1_prob):
                bin_state = bin(state)[2:].zfill(num_qubits)
                for qubit_idx, b in enumerate(bin_state):
                    b = int(b)
                    perturbation_probabilities[qubit_idx][b+2] += prob

            # print('Key %s has perturbation_probabilities:'%key,perturbation_probabilities)
            self.calibration_matrices[key] = self.get_calibration_matrix(perturbation_probabilities=perturbation_probabilities)
    
    def get_calibration_matrix(self,perturbation_probabilities):
        num_qubits = len(perturbation_probabilities)
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
        print(time()-begin)
        return calibration_matrix
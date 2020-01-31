import numpy as np
from scipy.optimize import minimize
from time import time
import copy
from qiskit.circuit.quantumregister import QuantumRegister
from utils.helper_fun import get_evaluator_info
from qiskit.ignis.mitigation.measurement import tensored_meas_cal
from utils.schedule import Scheduler
from qiskit.compiler import transpile

class TensoredMitigation:
    def __init__(self,circ_dict,device_name):
        self.circ_dict = copy.deepcopy(circ_dict)
        self.device_name = device_name
        self.check_status()

    def check_status(self):
        assert isinstance(self.circ_dict,dict)
        evaluator_info = get_evaluator_info(circ=None,device_name=self.device_name,fields=['device'])
        device_max_experiments = evaluator_info['device'].configuration().max_experiments
        keys_to_delete = []
        for key in self.circ_dict:
            value = self.circ_dict[key]
            if 'circ' not in value:
                raise Exception('Input circ_dict does not have circ for key {}'.format(key))
            elif 2**len(value['circ'].qubits)>device_max_experiments:
                keys_to_delete.append(key)
        for key in keys_to_delete:
            del self.circ_dict[key]
    
    def get_mitigation_circuits(self,real_device):
        meas_calibs_dict = {}
        device_evaluator_info = get_evaluator_info(circ=None,device_name=self.device_name,fields=['device','properties','basis_gates','coupling_map'])
        device_max_shots = device_evaluator_info['device'].configuration().max_shots
        device_max_experiments = device_evaluator_info['device'].configuration().max_experiments
        device_qubits = len(device_evaluator_info['properties'].qubits)
        for key in self.circ_dict:
            circ = self.circ_dict[key]['circ']
            mit_pattern = []
            qubit_group = []
            if real_device:
                _initial_layout = self.circ_dict[key]['initial_layout'].get_physical_bits()
                for q in _initial_layout:
                    if 'ancilla' not in _initial_layout[q].register.name:
                        if 2**(len(qubit_group)+1)<=device_max_experiments:
                            qubit_group.append(q)
                        else:
                            mit_pattern.append(qubit_group)
                            qubit_group = [q]
                if len(qubit_group)>0:
                    mit_pattern.append(qubit_group)
                qr = QuantumRegister(device_qubits)
            else:
                mit_pattern = [range(len(circ.qubits))]
                qr = QuantumRegister(len(circ.qubits))

            print('Circuit %s has mit_pattern:'%key,mit_pattern)
            self.circ_dict[key]['mit_pattern'] = mit_pattern
            meas_calibs, state_labels = tensored_meas_cal(mit_pattern=mit_pattern, qr=qr, circlabel='')
            meas_calibs_transpiled = transpile(meas_calibs,basis_gates=device_evaluator_info['basis_gates'])
            for meas_calib_circ in meas_calibs_transpiled:
                meas_calibs_dict_key = (key,meas_calib_circ.name.split('_')[1])
                assert meas_calibs_dict_key not in meas_calibs_dict
                if real_device:
                    meas_calibs_dict.update({meas_calibs_dict_key:{'circ':meas_calib_circ,'shots':device_max_shots,'initial_layout':self.circ_dict[key]['initial_layout']}})
                else:
                    meas_calibs_dict.update({meas_calibs_dict_key:{'circ':meas_calib_circ,'shots':device_max_shots}})
                # print(meas_calibs_dict_key)
        return meas_calibs_dict

    def run(self,real_device):
        self.meas_calibs_dict = self.get_mitigation_circuits(real_device=real_device)
        self.scheduler = Scheduler(circ_dict=self.meas_calibs_dict,device_name=self.device_name)
        self.scheduler.run(real_device=real_device)

    def retrieve(self):
        self.scheduler.retrieve(force_prob=False)
        for key in self.circ_dict:
            circ = self.circ_dict[key]['circ']
            num_qubits = len(circ.qubits)
            mit_pattern = self.circ_dict[key]['mit_pattern']
            qubit_list_sizes = []
            indices_list = []
            self.circ_dict[key]['calibration_matrices'] = []
            # print('Circuit %s'%key)
            for qubit_group in mit_pattern:
                self.circ_dict[key]['calibration_matrices'].append(np.zeros([2**len(qubit_group), 2**len(qubit_group)],dtype=float))
                qubit_list_sizes.append(len(qubit_group))
                indices_list.append({bin(idx)[2:].zfill(len(qubit_group)):idx for idx in range(2**len(qubit_group))})
            # print('qubit list sizes:',qubit_list_sizes)
            # print('indices list:',indices_list)
            for meas_calibs_dict_key in self.scheduler.circ_dict:
                if meas_calibs_dict_key[0]==key:
                    prepared_state = meas_calibs_dict_key[1]
                    # print('prepared state:',prepared_state)
                    state_cnts = self.scheduler.circ_dict[meas_calibs_dict_key]['hw']
                    for measured_state, counts in enumerate(state_cnts):
                        if counts==0:
                            continue
                        else:
                            measured_state = bin(measured_state)[2:].zfill(num_qubits)
                            end_index = num_qubits
                            # print('measured_state: {}, counts = {}'.format(measured_state,counts))
                            for cal_ind, cal_mat in enumerate(self.circ_dict[key]['calibration_matrices']):
                                # print('Filling up calibration matrix {:d}'.format(cal_ind))
                                start_index = end_index - qubit_list_sizes[cal_ind]
                                # print('Substate from {:d}-->{:d}'.format(start_index,end_index))
                                prepared_substate_index = indices_list[cal_ind][prepared_state[start_index:end_index]]
                                measured_substate_index = indices_list[cal_ind][measured_state[start_index:end_index]]
                                # print('prepared_state piece:',prepared_state[start_index:end_index],'measured_state piece:',measured_state[start_index:end_index])
                                # print('prepared_substate_index = {}, measured_substate_index = {}'.format(prepared_substate_index,measured_substate_index))
                                end_index = start_index
                                cal_mat[measured_substate_index][prepared_substate_index] += counts
            for mat_index, _ in enumerate(self.circ_dict[key]['calibration_matrices']):
                sums_of_columns = np.sum(self.circ_dict[key]['calibration_matrices'][mat_index], axis=0)
                self.circ_dict[key]['calibration_matrices'][mat_index] = np.divide(
                    self.circ_dict[key]['calibration_matrices'][mat_index], sums_of_columns,
                    out=np.zeros_like(self.circ_dict[key]['calibration_matrices'][mat_index]),
                    where=sums_of_columns != 0)

    def apply(self,unmitigated,mitigation_correspondence_dict):
        mitigated = copy.deepcopy(unmitigated)
        for key in self.circ_dict:
            keys_to_mitigate = mitigation_correspondence_dict[key]
            calibration_matrices = self.circ_dict[key]['calibration_matrices']
            qubit_list_sizes = [int(np.log(np.shape(mat)[0])/np.log(2)) for mat in calibration_matrices]
            indices_list = [{bin(ind)[2:].zfill(group_size): ind for ind in range(2**group_size)} for group_size in qubit_list_sizes]
            nqubits = sum(qubit_list_sizes)
            num_of_states = 2**nqubits
            all_states = [bin(state)[2:].zfill(nqubits) for state in range(2**nqubits)]
            for unmitigated_key in keys_to_mitigate:
                unmitigated_prob = np.array(unmitigated[unmitigated_key]['hw'],dtype=float)
                # print('unmitigated_prob:',unmitigated_prob)
                # print('qubit list sizes:',qubit_list_sizes)
                # print('indices_list:',indices_list)
                # print('nqubits:',nqubits)
                # print('all_states:',all_states)
                # print('num_of_states:',num_of_states)

                def fun(x):
                    mat_dot_x = np.zeros([num_of_states], dtype=float)
                    for state1_idx, state1 in enumerate(all_states):
                        mat_dot_x[state1_idx] = 0.
                        for state2_idx, state2 in enumerate(all_states):
                            if x[state2_idx] != 0:
                                product = 1.
                                end_index = nqubits
                                for c_ind, cal_mat in enumerate(calibration_matrices):

                                    start_index = end_index - qubit_list_sizes[c_ind]

                                    state1_as_int = indices_list[c_ind][state1[start_index:end_index]]

                                    state2_as_int = indices_list[c_ind][state2[start_index:end_index]]

                                    end_index = start_index
                                    product *= cal_mat[state1_as_int][state2_as_int]
                                    if product == 0:
                                        break
                                mat_dot_x[state1_idx] += (product * x[state2_idx])
                    return sum((unmitigated_prob - mat_dot_x)**2)
                
                x0 = np.random.rand(num_of_states)
                x0 = x0 / sum(x0)
                nshots = sum(unmitigated_prob)
                # print('random initial x0 = {}, nshots = {}'.format(x0,nshots))
                cons = ({'type': 'eq', 'fun': lambda x: nshots - sum(x)})
                bnds = tuple((0, nshots) for x in x0)
                res = minimize(fun, x0, method='SLSQP',constraints=cons, bounds=bnds, tol=1e-6)
                mitigated_cnts = res.x
                # print('mitigated_prob:',mitigated_cnts)
                mitigated[unmitigated_key]['mitigated_hw'] = copy.deepcopy(mitigated_cnts)
        for unmitigated_key in mitigated:
            if 'mitigated_hw' not in mitigated[unmitigated_key]:
                mitigated[unmitigated_key]['mitigated_hw'] = copy.deepcopy(mitigated[unmitigated_key]['hw'])
        self.circ_dict = copy.deepcopy(mitigated)

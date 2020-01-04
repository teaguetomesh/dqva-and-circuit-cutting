from qiskit.circuit.quantumregister import QuantumRegister
from utils.helper_fun import get_evaluator_info
from qiskit.ignis.mitigation.measurement import complete_meas_cal, tensored_meas_cal,CompleteMeasFitter, TensoredMeasFitter

class TensoredMitigation:
    def __init__(self,circ_dict,device_name):
        self.circ_dict = circ_dict
        self.device_name = device_name
        self.check_status()
        self.meas_calibs_dict = self.get_mitigation_circuits()
    
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
            num_qubits = len(evaluator_info['properties'].qubits)
            qr = QuantumRegister(num_qubits)
            mit_pattern = []
            _initial_layout = evaluator_info['initial_layout'].get_physical_bits()
            for q in _initial_layout:
                if 'ancilla' not in _initial_layout[q].register.name:
                    mit_pattern.append([q])
            meas_calibs, state_labels = tensored_meas_cal(mit_pattern=mit_pattern, qr=qr, circlabel='mcal')
            for circ in meas_calibs:
                meas_calibs_dict.update({'%s|%s'%(key,circ.name):circ})
        return meas_calibs_dict
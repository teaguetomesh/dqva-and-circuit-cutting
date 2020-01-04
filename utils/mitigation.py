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
            meas_calibs, state_labels = tensored_meas_cal(mit_pattern=mit_pattern, qr=qr, circlabel='mcal')
            meas_calibs_transpiled = transpile(meas_calibs, backend=evaluator_info['device'])
            for circ in meas_calibs_transpiled:
                meas_calibs_dict.update({'%s|%s'%(key,circ.name):{'circ':circ,'shots':device_max_shots}})
        return meas_calibs_dict

    def run(self,real_device=False):
        self.scheduler.run(real_device=real_device)

    def retrieve(self):
        self.scheduler.retrieve()
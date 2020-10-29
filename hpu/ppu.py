from qiskit import QuantumCircuit
from cutqc.initialization import check_valid
from cutqc.cutter import find_cuts
from cutqc.evaluator import find_subcircuit_O_rho_qubits, find_all_combinations, get_subcircuit_instance
from hpu.component import ComponentInterface

class PPU(ComponentInterface):
    '''
    Pre-processing Unit
    cuts an input circuit
    returns the cut_solution found and control signals for the MUX to attribute shots
    '''
    def __init__(self, config):
        self.max_subcircuit_qubit = config['max_subcircuit_qubit']
        self.num_subcircuits = config['num_subcircuits']
        self.max_cuts = config['max_cuts']

    def run(self, circuit):
        assert isinstance(circuit,QuantumCircuit)
        valid = check_valid(circuit=circuit)
        assert valid
        self.circuit = circuit

        cut_solution = find_cuts(circuit=self.circuit,
        max_subcircuit_qubit=self.max_subcircuit_qubit,
        num_subcircuits=self.num_subcircuits,
        max_cuts=self.max_cuts,verbose=False)
        if len(cut_solution)>0:
            full_circuit = cut_solution['circuit']
            subcircuits = cut_solution['subcircuits']
            complete_path_map = cut_solution['complete_path_map']

            circ_dict = {}
            all_indexed_combinations = {}
            for subcircuit_idx, subcircuit in enumerate(subcircuits):
                O_qubits, rho_qubits = find_subcircuit_O_rho_qubits(complete_path_map=complete_path_map,subcircuit_idx=subcircuit_idx)
                combinations, indexed_combinations = find_all_combinations(O_qubits, rho_qubits, subcircuit.qubits)
                circ_dict.update(get_subcircuit_instance(subcircuit_idx=subcircuit_idx,subcircuit=subcircuit, combinations=combinations))
                all_indexed_combinations[subcircuit_idx] = indexed_combinations
            cut_solution['subcircuit_instances'] = circ_dict
            cut_solution['all_indexed_combinations'] = all_indexed_combinations
        self.cut_solution = cut_solution

    def get_output(self):
        return self.cut_solution

    def close(self):
        pass
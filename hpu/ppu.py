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
    def __init__(self):
        pass
    
    def load_input(self,circuit):
        print('PPU gets circuit:')
        print(circuit)
        self.circuit = circuit
        valid = check_valid(circuit=self.circuit)
        assert valid

    def run(self,options):
        print('PPU options : {')
        [print(x,'=',options[x]) for x in options]
        print('}')
        cut_solution = find_cuts(circuit=self.circuit,
        max_subcircuit_qubit=options['max_subcircuit_qubit'],
        num_subcircuits=options['num_subcircuits'],
        max_cuts=options['max_cuts'])
        if len(cut_solution)==0:
            return RuntimeError, 'PPU found no cuts'
        else:
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
        return {'subcircuits':circ_dict, 'mux_control':all_indexed_combinations}, 'OK'

    def observe(self):
        pass

    def close(self):
        pass
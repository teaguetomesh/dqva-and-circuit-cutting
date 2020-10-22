from cutqc.initialization import check_valid
from cutqc.cutter import find_cuts
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
        return cut_solution

    def observe(self):
        pass

    def close(self):
        pass
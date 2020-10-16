from cutqc.cutter import find_cuts

class CutQC:
    def __init__(self):
        super().__init__()
    
    def cut(self, circuit, max_subcircuit_qubit, num_subcircuits, max_cuts):
        self.solution_dict = find_cuts(circuit=circuit,
        max_subcircuit_qubit=max_subcircuit_qubit,
        num_subcircuits=num_subcircuits,
        max_cuts=max_cuts)
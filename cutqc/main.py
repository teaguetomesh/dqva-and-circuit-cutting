from cutqc.initialization import check_valid
from cutqc.cutter import find_cuts

class CutQC:
    def __init__(self, circuits, results_directory):
        self.circuits = circuits
        self.results_directory = results_directory

        self.check_input()

    def check_input(self):
        for circuit_name in self.circuits:
            circuit = self.circuits[circuit_name]
            valid = check_valid(circuit=circuit)
            assert valid
    
    def cut(self, max_subcircuit_qubit, num_subcircuits, max_cuts):
        for circuit_name in self.circuits:
            circuit = self.circuits[circuit_name]
            cut_solution = find_cuts(circuit=circuit,
            max_subcircuit_qubit=max_subcircuit_qubit,
            num_subcircuits=num_subcircuits,
            max_cuts=max_cuts)
            self.circuits[circuit_name] = cut_solution
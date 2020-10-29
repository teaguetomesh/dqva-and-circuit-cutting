import os, subprocess, pickle

from cutqc.initialization import check_valid
from cutqc.cutter import find_cuts
from cutqc.file_manage import get_dirname

class CutQC:
    def __init__(self, circuits):
        self.circuits = circuits
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
            max_cuts=max_cuts,verbose=True)
            self.circuits[circuit_name] = cut_solution
            dirname = get_dirname(circuit_name=circuit_name,cc_size=max_subcircuit_qubit,
            early_termination=None,eval_mode=None,num_workers=None,qubit_limit=None,field='generator')
            if os.path.exists(dirname):
                subprocess.run(['rm','-r',dirname])
            os.makedirs(dirname)
            pickle.dump(cut_solution, open('%s/subcircuits.pckl'%(dirname),'wb'))
    
    # def evaluate(self,multi_nodes):
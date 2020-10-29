import os, subprocess, pickle, glob

from helper_functions.non_ibmq_functions import evaluate_circ

from cutqc.initialization import check_valid
from cutqc.cutter import find_cuts
from cutqc.evaluator import find_subcircuit_O_rho_qubits, find_all_combinations, get_subcircuit_instance, mutate_measurement_basis
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
            dirname = get_dirname(circuit_name=circuit_name,max_subcircuit_qubit=max_subcircuit_qubit,
            early_termination=None,eval_mode=None,num_workers=None,qubit_limit=None,field='cutter')
            if os.path.exists(dirname):
                subprocess.run(['rm','-r',dirname])
            os.makedirs(dirname)
            pickle.dump(cut_solution, open('%s/subcircuits.pckl'%(dirname),'wb'))
    
    def evaluate(self):
        self._run_subcircuits()
        self._measure()
    
    def _run_subcircuits(self):
        for circuit_name in self.circuits:
            full_circuit = self.circuits[circuit_name]['circuit']
            max_subcircuit_qubit = self.circuits[circuit_name]['max_subcircuit_qubit']
            subcircuits = self.circuits[circuit_name]['subcircuits']
            complete_path_map = self.circuits[circuit_name]['complete_path_map']
            counter = self.circuits[circuit_name]['counter']

            eval_folder = get_dirname(circuit_name=circuit_name,max_subcircuit_qubit=max_subcircuit_qubit,
            early_termination=None,num_workers=None,eval_mode='sv',qubit_limit=None,field='evaluator')
            if os.path.exists(eval_folder):
                subprocess.run(['rm','-r',eval_folder])
            os.makedirs(eval_folder)

            circ_dict = {}
            all_indexed_combinations = {}
            total_subcircuit_instances = 0
            for subcircuit_idx, subcircuit in enumerate(subcircuits):
                O_qubits, rho_qubits = find_subcircuit_O_rho_qubits(complete_path_map=complete_path_map,subcircuit_idx=subcircuit_idx)
                total_subcircuit_instances += 4**len(rho_qubits)*3**len(O_qubits)
                combinations, indexed_combinations = find_all_combinations(O_qubits, rho_qubits, subcircuit.qubits)
                circ_dict.update(get_subcircuit_instance(subcircuit_idx=subcircuit_idx,subcircuit=subcircuit, combinations=combinations))
                all_indexed_combinations[subcircuit_idx] = indexed_combinations
            for key in circ_dict:
                subcircuit_idx, inits, meas = key
                subcircuit_inst_prob = evaluate_circ(circuit=circ_dict[key]['circuit'],backend='statevector_simulator')
                mutated_meas = mutate_measurement_basis(meas)
                for meas in mutated_meas:
                    index = all_indexed_combinations[subcircuit_idx][(tuple(inits),tuple(meas))]
                    eval_file_name = '%s/raw_%d_%d.txt'%(eval_folder,subcircuit_idx,index)
                    # print('running',eval_file_name)
                    eval_file = open(eval_file_name,'w')
                    eval_file.write('d=%d effective=%d\n'%(counter[subcircuit_idx]['d'],counter[subcircuit_idx]['effective']))
                    [eval_file.write('%s '%x) for x in inits]
                    eval_file.write('\n')
                    [eval_file.write('%s '%x) for x in meas]
                    eval_file.write('\n')
                    [eval_file.write('%e '%x) for x in subcircuit_inst_prob]
                    eval_file.close()
    
    def _measure(self):
        subprocess.run(['rm','./cutqc/measure'])
        subprocess.run(['icc','./cutqc/measure.c','-o','./cutqc/measure','-lm'])

        for circuit_name in self.circuits:
            full_circuit = self.circuits[circuit_name]['circuit']
            subcircuits = self.circuits[circuit_name]['subcircuits']
            max_subcircuit_qubit = self.circuits[circuit_name]['max_subcircuit_qubit']

            eval_folder = get_dirname(circuit_name=circuit_name,max_subcircuit_qubit=max_subcircuit_qubit,
            early_termination=None,num_workers=None,eval_mode='sv',qubit_limit=None,field='evaluator')
            for subcircuit_idx in range(len(subcircuits)):
                eval_files = glob.glob('%s/raw_%d_*.txt'%(eval_folder,subcircuit_idx))
                eval_files = [str(x) for x in range(len(eval_files))]
                subprocess.run(args=['./cutqc/measure', '0', eval_folder,
                '%d'%full_circuit.num_qubits,'%d'%subcircuit_idx, '%d'%len(eval_files), *eval_files])
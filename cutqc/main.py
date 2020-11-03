import os, subprocess, pickle, glob

from qiskit_helper_functions.non_ibmq_functions import evaluate_circ, read_dict, find_process_jobs

from cutqc.initialization import check_valid
from cutqc.cutter import find_cuts
from cutqc.evaluator import find_subcircuit_O_rho_qubits, find_all_combinations, get_subcircuit_instance, mutate_measurement_basis
from cutqc.post_process import get_combinations, build
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
        self._organize(num_workers=1)
        self._vertical_collapse(early_termination=0)
        self._vertical_collapse(early_termination=1)
    
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
            pickle.dump(all_indexed_combinations, open('%s/all_indexed_combinations.pckl'%(eval_folder),'wb'))
    
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
    
    def _organize(self, num_workers):
        for circuit_name in self.circuits:
            full_circuit = self.circuits[circuit_name]['circuit']
            max_subcircuit_qubit = self.circuits[circuit_name]['max_subcircuit_qubit']
            subcircuits = self.circuits[circuit_name]['subcircuits']
            complete_path_map = self.circuits[circuit_name]['complete_path_map']
            counter = self.circuits[circuit_name]['counter']

            eval_folder = get_dirname(circuit_name=circuit_name,max_subcircuit_qubit=max_subcircuit_qubit,
            early_termination=None,num_workers=None,eval_mode='sv',qubit_limit=None,field='evaluator')

            all_indexed_combinations = read_dict(filename='%s/all_indexed_combinations.pckl'%(eval_folder))
            O_rho_pairs, combinations = get_combinations(complete_path_map=complete_path_map)
            kronecker_terms, _ = build(full_circuit=full_circuit, combinations=combinations,
            O_rho_pairs=O_rho_pairs, subcircuits=subcircuits, all_indexed_combinations=all_indexed_combinations)

            for rank in range(num_workers):
                subcircuit_kron_terms_file = open('%s/subcircuit_kron_terms_%d.txt'%(eval_folder,rank),'w')
                subcircuit_kron_terms_file.write('%d subcircuits\n'%len(kronecker_terms))
                for subcircuit_idx in kronecker_terms:
                    rank_subcircuit_kron_terms = find_process_jobs(jobs=list(kronecker_terms[subcircuit_idx].keys()),rank=rank,num_workers=num_workers)
                    subcircuit_kron_terms_file.write('subcircuit %d kron_terms %d num_effective %d\n'%(
                        subcircuit_idx,len(rank_subcircuit_kron_terms),counter[subcircuit_idx]['effective']))
                    for subcircuit_kron_term in rank_subcircuit_kron_terms:
                        subcircuit_kron_terms_file.write('subcircuit_kron_index=%d kron_term_len=%d\n'%(kronecker_terms[subcircuit_idx][subcircuit_kron_term],len(subcircuit_kron_term)))
                        [subcircuit_kron_terms_file.write('%d,%d '%(x[0],x[1])) for x in subcircuit_kron_term]
                        subcircuit_kron_terms_file.write('\n')
                    if rank==0:
                        print('Rank %d needs to vertical collapse %d/%d instances of subcircuit %d'%(rank,len(rank_subcircuit_kron_terms),len(kronecker_terms[subcircuit_idx]),subcircuit_idx),flush=True)
                subcircuit_kron_terms_file.close()
    
    def _vertical_collapse(self,early_termination):
        subprocess.run(['rm','./cutqc/vertical_collapse'])
        subprocess.run(['icc','-mkl','./cutqc/vertical_collapse.c','-o','./cutqc/vertical_collapse','-lm'])

        for circuit_name in self.circuits:
            full_circuit = self.circuits[circuit_name]['circuit']
            max_subcircuit_qubit = self.circuits[circuit_name]['max_subcircuit_qubit']
            subcircuits = self.circuits[circuit_name]['subcircuits']
            complete_path_map = self.circuits[circuit_name]['complete_path_map']
            counter = self.circuits[circuit_name]['counter']

            eval_folder = get_dirname(circuit_name=circuit_name,max_subcircuit_qubit=max_subcircuit_qubit,
            early_termination=None,num_workers=None,eval_mode='sv',qubit_limit=None,field='evaluator')
            vertical_collapse_folder = get_dirname(circuit_name=circuit_name,max_subcircuit_qubit=max_subcircuit_qubit,
            early_termination=early_termination,num_workers=None,eval_mode='sv',qubit_limit=None,field='vertical_collapse')

            rank_files = glob.glob('%s/subcircuit_kron_terms_*.txt'%eval_folder)
            if len(rank_files)==0:
                raise Exception('There are no rank_files for _vertical_collapse')
            if os.path.exists(vertical_collapse_folder):
                subprocess.run(['rm','-r',vertical_collapse_folder])
            os.makedirs(vertical_collapse_folder)
            child_processes = []
            for rank in range(len(rank_files)):
                subcircuit_kron_terms_file = '%s/subcircuit_kron_terms_%d.txt'%(eval_folder,rank)
                p = subprocess.Popen(args=['./cutqc/vertical_collapse', '%d'%full_circuit.num_qubits, '%s'%subcircuit_kron_terms_file, '%s'%eval_folder, '%s'%vertical_collapse_folder, '%d'%early_termination, '%d'%rank])
                child_processes.append(p)
            for rank in range(len(rank_files)):
                cp = child_processes[rank]
                cp.wait()
                if early_termination==1:
                    subprocess.run(['rm','%s/subcircuit_kron_terms_%d.txt'%(eval_folder,rank)])
            if early_termination==1:
                measured_files = glob.glob('%s/measured*.txt'%eval_folder)
                for measured_file in measured_files:
                    subprocess.run(['rm',measured_file])
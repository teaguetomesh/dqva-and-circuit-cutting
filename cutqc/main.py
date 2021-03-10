import os, subprocess, pickle, glob, random, time
from termcolor import colored
import numpy as np
import itertools
import multiprocessing as mp
from datetime import datetime

# from qiskit_helper_functions.non_ibmq_functions import evaluate_circ, read_dict, find_process_jobs
# from qiskit_helper_functions.schedule import Scheduler

from cutqc.helper_fun import check_valid, get_dirname
from cutqc.cutter import find_cuts, solve_model, subcircuits_parser
from cutqc.evaluator import simulate_subcircuit, write_subcircuit, generate_subcircuit_instances
from cutqc.distributor import distribute
from cutqc.post_process import get_combinations, build

class CutQC:
    '''
    The CutQC class should only handle the IOs
    Leave all actual functions to individual modules
    '''
    def __init__(self, circuits, max_subcircuit_qubit, num_subcircuits, max_cuts, verbose):
        self.circuits = circuits
        self.verbose = verbose
        self._check_input()
        self.circuit = list(circuits.values())[0]
        #self._cut(max_subcircuit_qubit=max_subcircuit_qubit,num_subcircuits=num_subcircuits,max_cuts=max_cuts)

    def _check_input(self):
        for circuit_name in self.circuits:
            circuit = self.circuits[circuit_name]
            valid = check_valid(circuit=circuit)
            assert valid

    def get_MIP_model(self, max_subcircuit_qubit, num_subcircuits, max_cuts):
        mip_model = solve_model(self.circuit, max_subcircuit_qubit,
                                num_subcircuits, max_cuts, self.verbose)
        return mip_model

    def get_subcircs_from_model(self, fullcirc, model):
        return subcircuits_parser(model.subcircuits_vertices, fullcirc)

    def _cut(self, max_subcircuit_qubit, num_subcircuits, max_cuts):
        if self.verbose > 0:
            print('-'*20,'Cut','-'*20)

        #circuit = self.circuits[list(self.circuits.keys())[0]]
        #cut_solutions = [find_cuts(circuit, max_subcircuit_qubit, num_subcircuits, max_cuts, False)]

        pool = mp.Pool(processes=mp.cpu_count())
        data = []
        for circuit_name in self.circuits:
            circuit = self.circuits[circuit_name]
            data.append([circuit,max_subcircuit_qubit,num_subcircuits,max_cuts,False])
        cut_solutions = pool.starmap(find_cuts,data)

        self.cut_solns = cut_solutions

        #for circuit_name, cut_solution in zip(self.circuits,cut_solutions):
        #    source_folder = get_dirname(circuit_name=circuit_name, max_subcircuit_qubit=max_subcircuit_qubit,
        #                                early_termination=None, eval_mode=None, num_threads=None,
        #                                qubit_limit=None, field='cutter')
        #    if os.path.exists(source_folder):
        #        subprocess.run(['rm','-r',source_folder])
        #    os.makedirs(source_folder)
        #    pickle.dump(cut_solution, open('%s/subcircuits.pckl'%(source_folder),'wb'))
        #    if self.verbose > 0:
        #        print('{:s} : {:d} cuts --> {}'.format(circuit_name, len(cut_solution['positions']),
        #                                               cut_solution['counter']))

    def evaluate(self,circuit_cases,eval_mode,num_nodes,num_threads,early_termination,ibmq):
        if self.verbose > 0:
            print('-'*20,'Evaluate, mode = %s'%eval_mode,'-'*20)
        self.circuit_cases = circuit_cases
        self._run_subcircuits(eval_mode=eval_mode,num_nodes=num_nodes,num_threads=num_threads,ibmq=ibmq)
        self._measure(eval_mode=eval_mode,num_nodes=num_nodes,num_threads=num_threads)
        self._organize(eval_mode=eval_mode,num_threads=num_threads)
        if 0 in early_termination:
            self._vertical_collapse(early_termination=0,eval_mode=eval_mode)
        if 1 in early_termination:
            self._vertical_collapse(early_termination=1,eval_mode=eval_mode)

    def post_process(self,circuit_cases,eval_mode,num_nodes,num_threads,early_termination,qubit_limit,recursion_depth):
        if self.verbose > 0:
            print('-'*20,'Postprocess, mode = %s'%eval_mode,'-'*20)
        # TODO: handle runtime mode
        self.circuit_cases = circuit_cases
        subprocess.run(['rm','./cutqc/merge'])
        subprocess.run(['icc','-mkl','./cutqc/merge.c','-o','./cutqc/merge','-lm'])
        subprocess.run(['rm','./cutqc/build'])
        subprocess.run(['icc','-fopenmp','-mkl','-lpthread','-march=native','./cutqc/build.c','-o','./cutqc/build','-lm'])

        for circuit_case in self.circuit_cases:
            circuit_name = circuit_case.split('|')[0]
            max_subcircuit_qubit = int(circuit_case.split('|')[1])
            source_folder = get_dirname(circuit_name=circuit_name,max_subcircuit_qubit=max_subcircuit_qubit,
            early_termination=None,eval_mode=None,num_threads=None,qubit_limit=None,field='cutter')
            cut_solution = read_dict('%s/subcircuits.pckl'%(source_folder))
            assert(max_subcircuit_qubit == cut_solution['max_subcircuit_qubit'])
            full_circuit = cut_solution['circuit']
            subcircuits = cut_solution['subcircuits']
            complete_path_map = cut_solution['complete_path_map']
            counter = cut_solution['counter']

            dest_folder = get_dirname(circuit_name=circuit_name, max_subcircuit_qubit=max_subcircuit_qubit,
                                      early_termination=early_termination, num_threads=num_threads,
                                      eval_mode=eval_mode,qubit_limit=qubit_limit,field='build')

            if os.path.exists('%s'%dest_folder):
                subprocess.run(['rm','-r',dest_folder])
            os.makedirs(dest_folder)

            vertical_collapse_folder = get_dirname(circuit_name=circuit_name,
                    max_subcircuit_qubit=max_subcircuit_qubit, early_termination=early_termination,
                    num_threads=None,eval_mode=eval_mode,qubit_limit=None,field='vertical_collapse')

            rec_layers = []
            for recursion_layer in range(recursion_depth):
                if self.verbose > 0:
                    print('-----> %s Recursion Layer %d'%(circuit_case,recursion_layer),flush=True)
                recursion_qubit = qubit_limit
                # TODO: reduce IO for runtime mode
                distribute(circuit_name=circuit_name, max_subcircuit_qubit=max_subcircuit_qubit,
                           eval_mode=eval_mode, early_termination=early_termination, num_threads=num_threads,
                           qubit_limit=qubit_limit, recursion_layer=recursion_layer,
                           recursion_qubit=recursion_qubit, verbose=self.verbose)
                if self.verbose > 0:
                    print('__Merge__',flush=True)
                terminated = self._merge(circuit_case=circuit_case,
                                         vertical_collapse_folder=vertical_collapse_folder,
                                         dest_folder=dest_folder, recursion_layer=recursion_layer)
                if terminated:
                    break
                if self.verbose > 0:
                    print('__Build__',flush=True)
                reconstructed_prob = self._build(circuit_case=circuit_case, dest_folder=dest_folder,
                                                 recursion_layer=recursion_layer)
                rec_layers.append(reconstructed_prob)
            return rec_layers

    def reorder(self, full_circuit, unordered, complete_path_map, subcircuits, layer_schedule):
        subcircuit_out_qubits = {}
        for input_qubit in complete_path_map:
            path = complete_path_map[input_qubit]
            output_qubit = path[-1]
            if output_qubit['subcircuit_idx'] in subcircuit_out_qubits:
                subcircuit_out_qubits[output_qubit['subcircuit_idx']].append((output_qubit['subcircuit_qubit'],full_circuit.qubits.index(input_qubit)))
            else:
                subcircuit_out_qubits[output_qubit['subcircuit_idx']] = [(output_qubit['subcircuit_qubit'],full_circuit.qubits.index(input_qubit))]

        for subcircuit_idx in subcircuit_out_qubits:
            subcircuit_out_qubits[subcircuit_idx] = sorted(subcircuit_out_qubits[subcircuit_idx],
                                                           key=lambda x:subcircuits[subcircuit_idx].qubits.index(x[0]),
                                                           reverse=True)
            subcircuit_out_qubits[subcircuit_idx] = [x[1] for x in subcircuit_out_qubits[subcircuit_idx]]

        if self.verbose > 0:
            print('subcircuit_out_qubits:',subcircuit_out_qubits,'smart_order:',layer_schedule['smart_order'])

        unordered_qubit_format = []
        unordered_qubit_state = []
        for subcircuit_idx in layer_schedule['smart_order']:
            if self.verbose > 0:
                print('subcircuit %d'%subcircuit_idx,layer_schedule['subcircuit_state'][subcircuit_idx])
            if subcircuit_idx in subcircuit_out_qubits:
                unordered_qubit_format += subcircuit_out_qubits[subcircuit_idx]
                unordered_qubit_state += layer_schedule['subcircuit_state'][subcircuit_idx]

        labelled_probs = {}
        num_active = unordered_qubit_state.count('active')
        num_merged = unordered_qubit_state.count('merged')
        merged_states = list(itertools.product(['0','1'],repeat=num_merged))
        for state_ctr, unordered_p in enumerate(unordered):
            avg_unordered_p = unordered_p/2**num_merged
            for merged_state in merged_states:
                bin_state_ctr = bin(state_ctr)[2:].zfill(num_active)
                bin_full_state = []
                for qubit in unordered_qubit_state:
                    if qubit=='active':
                        bin_full_state.append(bin_state_ctr[0])
                        bin_state_ctr = bin_state_ctr[1:]
                    elif qubit=='merged':
                        bin_full_state.append(merged_state[0])
                        merged_state = merged_state[1:]
                    elif qubit=='0' or qubit =='1':
                        bin_full_state.append(qubit)
                _, bin_full_state = zip(*sorted(zip(unordered_qubit_format, bin_full_state),reverse=True))
                bin_full_state = ''.join(bin_full_state)
                full_state = int(bin_full_state,2)
                labelled_probs[full_state] = avg_unordered_p

        # convert ordered into a probability dict
        threshold = 1e-10
        probability_dict = {'{:0{}b}'.format(key, len(full_circuit.qubits)): labelled_probs[key] for key in labelled_probs.keys() \
                            if labelled_probs[key] > threshold}
        return probability_dict

    def verify(self,circuit_cases, early_termination, num_threads, qubit_limit, eval_mode):
        # TODO: make verify as functions
        for circuit_case in circuit_cases:
            circuit_name = circuit_case.split('|')[0]
            max_subcircuit_qubit = int(circuit_case.split('|')[1])
            subprocess.run(['python','-m','cutqc.verify',
            '--circuit_name',circuit_name,
            '--max_subcircuit_qubit',str(max_subcircuit_qubit),
            '--early_termination',str(early_termination),
            '--num_threads',str(num_threads),
            '--qubit_limit',str(qubit_limit),
            '--eval_mode',eval_mode])

    def _run_subcircuits(self,eval_mode,num_nodes,num_threads,ibmq):
        for circuit_case in self.circuit_cases:
            circuit_name = circuit_case.split('|')[0]
            max_subcircuit_qubit = int(circuit_case.split('|')[1])
            source_folder = get_dirname(circuit_name=circuit_name,max_subcircuit_qubit=max_subcircuit_qubit,
            early_termination=None,eval_mode=None,num_threads=None,qubit_limit=None,field='cutter')
            cut_solution = read_dict('%s/subcircuits.pckl'%(source_folder))
            if len(cut_solution)==0:
                continue
            assert(max_subcircuit_qubit == cut_solution['max_subcircuit_qubit'])
            subcircuits = cut_solution['subcircuits']
            complete_path_map = cut_solution['complete_path_map']
            counter = cut_solution['counter']

            eval_folder = get_dirname(circuit_name=circuit_name,max_subcircuit_qubit=max_subcircuit_qubit,
            early_termination=None,num_threads=None,eval_mode=eval_mode,qubit_limit=None,field='evaluator')
            if os.path.exists(eval_folder):
                subprocess.run(['rm','-r',eval_folder])
            os.makedirs(eval_folder)

            circ_dict, all_indexed_combinations = generate_subcircuit_instances(subcircuits=subcircuits,complete_path_map=complete_path_map)
            pickle.dump(all_indexed_combinations, open('%s/all_indexed_combinations.pckl'%(eval_folder),'wb'))

            if eval_mode=='sv':
                data = []
                for key in circ_dict:
                    data.append([key,circ_dict[key]['circuit'],eval_mode,eval_folder,counter])
                random.shuffle(data) # Ensure a somewhat fair distribution of workloads
                chunksize = max(len(data)//num_threads//10,1)
                pool = mp.Pool(processes=num_threads)
                pool.starmap(simulate_subcircuit,data,chunksize=chunksize)
            elif eval_mode=='runtime':
                data = []
                subcircuit_idx_written = []
                for key in circ_dict:
                    subcircuit_idx, _, _ = key
                    if subcircuit_idx not in subcircuit_idx_written:
                        subcircuit_idx_written.append(subcircuit_idx)
                        data.append([key,circ_dict[key]['circuit'],eval_mode,eval_folder,counter])
                random.shuffle(data) # Ensure a somewhat fair distribution of workloads
                chunksize = max(len(data)//num_threads//10,1)
                pool = mp.Pool(processes=num_threads)
                pool.starmap(simulate_subcircuit,data,chunksize=chunksize)
            elif 'ibmq' in eval_mode:
                # NOTE: control whether to use real device
                scheduler = Scheduler(circ_dict=circ_dict,
                token=ibmq['token'],hub=ibmq['hub'],group=ibmq['group'],project=ibmq['project'],device_name=eval_mode,datetime=datetime.now())
                scheduler.submit_jobs(real_device=True,transpilation=True,verbose=True)
                scheduler.retrieve_jobs(force_prob=True,save_memory=False,save_directory=None,verbose=True)
                data = []
                for key in scheduler.circ_dict:
                    data.append([key,eval_folder,counter,scheduler.circ_dict[key]['prob']])
                random.shuffle(data) # Ensure a somewhat fair distribution of workloads
                chunksize = max(len(data)//num_threads//10,1)
                pool = mp.Pool(processes=num_threads)
                pool.starmap(write_subcircuit,data,chunksize=chunksize)
            else:
                raise NotImplementedError

    def _measure(self, eval_mode, num_nodes, num_threads):
        subprocess.run(['rm','./cutqc/measure'])
        subprocess.run(['icc','./cutqc/measure.c','-o','./cutqc/measure','-lm'])

        for circuit_case in self.circuit_cases:
            circuit_name = circuit_case.split('|')[0]
            max_subcircuit_qubit = int(circuit_case.split('|')[1])
            source_folder = get_dirname(circuit_name=circuit_name,max_subcircuit_qubit=max_subcircuit_qubit,
            early_termination=None,eval_mode=None,num_threads=None,qubit_limit=None,field='cutter')
            cut_solution = read_dict('%s/subcircuits.pckl'%(source_folder))
            if len(cut_solution)==0:
                continue
            assert(max_subcircuit_qubit == cut_solution['max_subcircuit_qubit'])
            full_circuit = cut_solution['circuit']
            subcircuits = cut_solution['subcircuits']

            eval_folder = get_dirname(circuit_name=circuit_name,max_subcircuit_qubit=max_subcircuit_qubit,
            early_termination=None,num_threads=None,eval_mode=eval_mode,qubit_limit=None,field='evaluator')
            for subcircuit_idx in range(len(subcircuits)):
                eval_files = glob.glob('%s/raw_%d_*.txt'%(eval_folder,subcircuit_idx))
                child_processes = []
                for rank in range(num_threads):
                    process_eval_files = find_process_jobs(jobs=range(len(eval_files)),rank=rank,num_threads=num_threads)
                    process_eval_files = [str(x) for x in process_eval_files]
                    if rank==0:
                        if self.verbose > 0:
                            print('%s subcircuit %d : rank %d/%d needs to measure %d/%d instances'%(
                            circuit_case,subcircuit_idx,rank,num_threads,len(process_eval_files),len(eval_files)),flush=True)
                    p = subprocess.Popen(args=['./cutqc/measure', '%d'%rank, eval_folder,
                    '%d'%full_circuit.num_qubits,'%d'%subcircuit_idx, '%d'%len(process_eval_files), *process_eval_files])
                    child_processes.append(p)
                [cp.wait() for cp in child_processes]

    def _organize(self, eval_mode, num_threads):
        '''
        Organize parallel processing for the subsequent vertical collapse procedure
        '''
        for circuit_case in self.circuit_cases:
            circuit_name = circuit_case.split('|')[0]
            max_subcircuit_qubit = int(circuit_case.split('|')[1])
            source_folder = get_dirname(circuit_name=circuit_name,max_subcircuit_qubit=max_subcircuit_qubit,
            early_termination=None,eval_mode=None,num_threads=None,qubit_limit=None,field='cutter')
            cut_solution = read_dict('%s/subcircuits.pckl'%(source_folder))
            if len(cut_solution)==0:
                continue
            assert(max_subcircuit_qubit == cut_solution['max_subcircuit_qubit'])
            full_circuit = cut_solution['circuit']
            subcircuits = cut_solution['subcircuits']
            complete_path_map = cut_solution['complete_path_map']
            counter = cut_solution['counter']

            eval_folder = get_dirname(circuit_name=circuit_name,max_subcircuit_qubit=max_subcircuit_qubit,
            early_termination=None,num_threads=None,eval_mode=eval_mode,qubit_limit=None,field='evaluator')

            all_indexed_combinations = read_dict(filename='%s/all_indexed_combinations.pckl'%(eval_folder))
            O_rho_pairs, combinations = get_combinations(complete_path_map=complete_path_map)
            kronecker_terms, _ = build(full_circuit=full_circuit, combinations=combinations,
            O_rho_pairs=O_rho_pairs, subcircuits=subcircuits, all_indexed_combinations=all_indexed_combinations)

            for rank in range(num_threads):
                subcircuit_kron_terms_file = open('%s/subcircuit_kron_terms_%d.txt'%(eval_folder,rank),'w')
                subcircuit_kron_terms_file.write('%d subcircuits\n'%len(kronecker_terms))
                for subcircuit_idx in kronecker_terms:
                    if eval_mode=='runtime':
                        rank_subcircuit_kron_terms = [list(kronecker_terms[subcircuit_idx].keys())[0]]
                    else:
                        rank_subcircuit_kron_terms = find_process_jobs(jobs=list(kronecker_terms[subcircuit_idx].keys()),rank=rank,num_threads=num_threads)
                    subcircuit_kron_terms_file.write('subcircuit %d kron_terms %d num_effective %d\n'%(
                        subcircuit_idx,len(rank_subcircuit_kron_terms),counter[subcircuit_idx]['effective']))
                    for subcircuit_kron_term in rank_subcircuit_kron_terms:
                        subcircuit_kron_terms_file.write('subcircuit_kron_index=%d kron_term_len=%d\n'%(kronecker_terms[subcircuit_idx][subcircuit_kron_term],len(subcircuit_kron_term)))
                        if eval_mode=='runtime':
                            [subcircuit_kron_terms_file.write('%d,0 '%(x[0])) for x in subcircuit_kron_term]
                        else:
                            [subcircuit_kron_terms_file.write('%d,%d '%(x[0],x[1])) for x in subcircuit_kron_term]
                        subcircuit_kron_terms_file.write('\n')
                    if rank==0 and self.verbose > 0:
                        print('%s subcircuit %d : rank %d/%d needs to vertical collapse %d/%d instances'%(
                            circuit_case,subcircuit_idx,rank,num_threads,len(rank_subcircuit_kron_terms),len(kronecker_terms[subcircuit_idx])),flush=True)
                subcircuit_kron_terms_file.close()
    
    def _vertical_collapse(self,early_termination,eval_mode):
        subprocess.run(['rm','./cutqc/vertical_collapse'])
        subprocess.run(['icc','-mkl','./cutqc/vertical_collapse.c','-o','./cutqc/vertical_collapse','-lm'])

        for circuit_case in self.circuit_cases:
            circuit_name = circuit_case.split('|')[0]
            max_subcircuit_qubit = int(circuit_case.split('|')[1])
            source_folder = get_dirname(circuit_name=circuit_name,max_subcircuit_qubit=max_subcircuit_qubit,
            early_termination=None,eval_mode=None,num_threads=None,qubit_limit=None,field='cutter')
            cut_solution = read_dict('%s/subcircuits.pckl'%(source_folder))
            if len(cut_solution)==0:
                continue
            assert(max_subcircuit_qubit == cut_solution['max_subcircuit_qubit'])
            full_circuit = cut_solution['circuit']
            subcircuits = cut_solution['subcircuits']
            complete_path_map = cut_solution['complete_path_map']
            counter = cut_solution['counter']

            eval_folder = get_dirname(circuit_name=circuit_name,max_subcircuit_qubit=max_subcircuit_qubit,
            early_termination=None,num_threads=None,eval_mode=eval_mode,qubit_limit=None,field='evaluator')
            vertical_collapse_folder = get_dirname(circuit_name=circuit_name,max_subcircuit_qubit=max_subcircuit_qubit,
            early_termination=early_termination,num_threads=None,eval_mode=eval_mode,qubit_limit=None,field='vertical_collapse')

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
            [cp.wait() for cp in child_processes]
            if early_termination==1:
                measured_files = glob.glob('%s/measured*.txt'%eval_folder)
                [subprocess.run(['rm',measured_file]) for measured_file in measured_files]
                [subprocess.run(['rm','%s/subcircuit_kron_terms_%d.txt'%(eval_folder,rank)]) for rank in range(len(rank_files))]

    def _merge(self, circuit_case, dest_folder, recursion_layer, vertical_collapse_folder):
        dynamic_definition_folder = '%s/dynamic_definition_%d'%(dest_folder,recursion_layer)
        if not os.path.exists(dynamic_definition_folder):
            return True
        merge_files = glob.glob('%s/merge_*.txt'%dynamic_definition_folder)
        num_threads = len(merge_files)
        child_processes = []
        for rank in range(num_threads):
            merge_file = '%s/merge_%d.txt'%(dynamic_definition_folder,rank)
            p = subprocess.Popen(args=['./cutqc/merge', '%s'%merge_file, '%s'%vertical_collapse_folder, '%s'%dynamic_definition_folder,
            '%d'%rank, '%d'%recursion_layer])
            child_processes.append(p)
        elapsed = 0
        for rank in range(num_threads):
            cp = child_processes[rank]
            cp.wait()
        time.sleep(1)
        rank_logs = open('%s/rank_%d_summary.txt'%(dynamic_definition_folder,rank), 'r')
        lines = rank_logs.readlines()
        assert lines[-2].split(' = ')[0]=='Total merge time' and lines[-1]=='DONE'
        elapsed = max(elapsed,float(lines[-2].split(' = ')[1]))

        if self.verbose > 0:
            print('%s _merge took %.3e seconds'%(circuit_case,elapsed),flush=True)
        pickle.dump({'merge_time_%d'%recursion_layer:elapsed}, open('%s/summary.pckl'%(dest_folder),'ab'))
        return False

    def _build(self, circuit_case, dest_folder, recursion_layer):
        dynamic_definition_folder = '%s/dynamic_definition_%d'%(dest_folder,recursion_layer)
        build_files = glob.glob('%s/build_*.txt'%dynamic_definition_folder)
        num_threads = len(build_files)
        child_processes = []
        for rank in range(num_threads):
            build_file = '%s/build_%d.txt'%(dynamic_definition_folder,rank)
            p = subprocess.Popen(args=['./cutqc/build', '%s'%build_file, '%s'%dynamic_definition_folder, 
            '%s'%dynamic_definition_folder, '%d'%rank, '%d'%recursion_layer])
            child_processes.append(p)

        elapsed = []
        reconstructed_prob = None
        for rank in range(num_threads):
            cp = child_processes[rank]
            cp.wait()
            rank_logs = open('%s/rank_%d_summary.txt'%(dynamic_definition_folder,rank), 'r')
            lines = rank_logs.readlines()
            assert lines[-2].split(' = ')[0]=='Total build time' and lines[-1] == 'DONE'
            elapsed.append(float(lines[-2].split(' = ')[1]))

            fp = open('%s/reconstructed_prob_%d.txt'%(dynamic_definition_folder,rank), 'r')
            for i, line in enumerate(fp):
                rank_reconstructed_prob = line.split(' ')[:-1]
                rank_reconstructed_prob = np.array(rank_reconstructed_prob)
                rank_reconstructed_prob = rank_reconstructed_prob.astype(np.float)
                if i>0:
                    raise Exception('C build_output should not have more than 1 line')
            fp.close()
            subprocess.run(['rm','%s/reconstructed_prob_%d.txt'%(dynamic_definition_folder,rank)])
            if isinstance(reconstructed_prob,np.ndarray):
                reconstructed_prob += rank_reconstructed_prob
            else:
                reconstructed_prob = rank_reconstructed_prob
        if self.verbose > 0:
            print('%s _build took %.3e seconds'%(circuit_case,max(elapsed)),flush=True)
        pickle.dump({'build_time_%d'%recursion_layer:np.array(elapsed)}, open('%s/summary.pckl'%(dest_folder),'ab'))
        max_states = sorted(range(len(reconstructed_prob)),key=lambda x:reconstructed_prob[x],reverse=True)
        pickle.dump({'zoomed_ctr':0,'max_states':max_states,'reconstructed_prob':reconstructed_prob},open('%s/build_output.pckl'%(dynamic_definition_folder),'wb'))
        return reconstructed_prob

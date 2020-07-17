import subprocess
import argparse
import glob
import os
import random
from time import time
from termcolor import colored
import pickle

from utils.helper_fun import find_process_jobs, get_dirname, read_file

def generator(circuit_type,all_circ_sizes,cc_size):
    random.Random(4).shuffle(all_circ_sizes)
    num_generator_workers = 1
    child_processes = []
    for rank in range(num_generator_workers):
        process_jobs = find_process_jobs(jobs=all_circ_sizes,rank=rank,num_workers=num_generator_workers)
        if len(process_jobs)==0:
            continue
        else:
            p = subprocess.Popen(args=['python', 'generator.py',
            '--circuit_type',circuit_type,
            '--circ_sizes',*[str(x) for x in process_jobs],
            '--cc_size',str(cc_size)])
            child_processes.append(p)
    
    for cp in child_processes:
        cp.wait()

def evaluator(circuit_type,all_circ_sizes,cc_size,eval_mode):
    for full_circ_size in all_circ_sizes:
        source_folder = get_dirname(circuit_type=circuit_type,cc_size=cc_size,full_circ_size=full_circ_size,
        techniques=None,eval_mode=None,field='generator')
        eval_folder = get_dirname(circuit_type=circuit_type,cc_size=cc_size,full_circ_size=full_circ_size,
        techniques=None,eval_mode=eval_mode,field='evaluator')
        if not os.path.exists(source_folder):
            continue
        else:
            if os.path.exists(eval_folder):
                subprocess.run(['rm','-r',eval_folder])
            os.makedirs(eval_folder)
    child_processes = []
    num_eval_workers = 1
    for rank in range(num_eval_workers):
        p = subprocess.Popen(args=['python', 'evaluator.py',
        '--circuit_type',circuit_type,
        '--circ_sizes',*[str(x) for x in all_circ_sizes],
        '--cc_size',str(cc_size),
        '--eval_workers',str(num_eval_workers),
        '--eval_rank',str(rank),
        '--eval_mode',eval_mode])
        child_processes.append(p)
    
    for cp in child_processes:
        cp.wait()
    
def measure(circuit_type,all_circ_sizes,cc_size,eval_mode):
    subprocess.run(['rm','./measure'])
    subprocess.run(['icc','./measure.c','-o','./measure','-lm'])

    for full_circ_size in all_circ_sizes:
        source_folder = get_dirname(circuit_type=circuit_type,cc_size=cc_size,full_circ_size=full_circ_size,
        techniques=None,eval_mode=None,field='generator')
        eval_folder = get_dirname(circuit_type=circuit_type,cc_size=cc_size,full_circ_size=full_circ_size,
        techniques=None,eval_mode=eval_mode,field='evaluator')
        meas_folder = get_dirname(circuit_type=circuit_type,cc_size=cc_size,full_circ_size=full_circ_size,
        techniques=None,eval_mode=eval_mode,field='measure')
        if not os.path.exists(eval_folder):
            continue
        if os.path.exists(meas_folder):
            subprocess.run(['rm','-r',meas_folder])
        os.makedirs(meas_folder)

        case_dict = read_file(filename='%s/subcircuits.pckl'%source_folder)
        subcircuit_circs = case_dict['subcircuit_circs']
        counter = case_dict['counter']

        total_measure_time = 0
        for subcircuit_idx, subcircuit_circ in enumerate(subcircuit_circs):
            eval_files = glob.glob('%s/%d_*.txt'%(eval_folder,subcircuit_idx))
            child_processes = []
            num_measure_workers = 8
            for rank in range(num_measure_workers):
                process_eval_files = find_process_jobs(jobs=range(len(eval_files)),rank=rank,num_workers=num_measure_workers)
                process_eval_files = [str(x) for x in process_eval_files]
                # print('Rank %d measure %d-q subcircuit %d %d/%d circuits'%(
                #     rank,full_circ_size,subcircuit_idx,len(process_eval_files),len(eval_files)),flush=True)
            
                p = subprocess.Popen(args=['./measure', '%d'%rank, eval_folder, meas_folder,
                '%d'%subcircuit_idx, '%d'%len(process_eval_files), *process_eval_files])
                child_processes.append(p)
        
            subcircuit_measure_time = 0
            for rank in range(num_measure_workers):
                cp = child_processes[rank]
                cp.wait()
                rank_logs = open('%s/subcircuit_%d_rank_%d.txt'%(meas_folder,subcircuit_idx,rank), 'r')
                lines = rank_logs.readlines()
                assert lines[0].split(' = ')[0]=='Total measure time' and lines[1] == 'DONE'
                subcircuit_measure_time = max(subcircuit_measure_time,float(lines[0].split(' = ')[1]))
            total_measure_time += subcircuit_measure_time

        time_str = colored('%d-q %s on %d-q cc_size took %.3e seconds'%(full_circ_size,circuit_type,cc_size,total_measure_time),'blue')
        print(time_str,flush=True)
        pickle.dump({'measure_time':total_measure_time}, open('%s/summary.pckl'%(meas_folder),'wb'))

def distributor(circuit_type,all_circ_sizes,cc_size,techniques,eval_mode):
    subprocess.run(args=['python', 'distributor.py',
    '--circuit_type',circuit_type,
    '--circ_sizes',*[str(x) for x in all_circ_sizes],
    '--cc_size',str(cc_size),
    '--techniques',*[str(x) for x in techniques],
    '--eval_mode',eval_mode])

def vertical_collapse(circuit_type,all_circ_sizes,cc_size,techniques,eval_mode):
    _, early_termination, num_workers, _ = techniques
    subprocess.run(['rm','./vertical_collapse'])
    subprocess.run(['icc','-mkl','./vertical_collapse.c','-o','./vertical_collapse','-lm'])

    for full_circ_size in all_circ_sizes:
        dest_folder = get_dirname(circuit_type=circuit_type,cc_size=cc_size,full_circ_size=full_circ_size,
        techniques=techniques,eval_mode=eval_mode,field='rank')
        meas_folder = get_dirname(circuit_type=circuit_type,cc_size=cc_size,full_circ_size=full_circ_size,
        techniques=None,eval_mode=eval_mode,field='measure')
        if os.path.exists('%s/vertical_collapse'%dest_folder):
            subprocess.run(['rm','-r','%s/vertical_collapse'%dest_folder])
        has_rank_folders = len(glob.glob('%s/rank_*'%dest_folder))>0
        if not has_rank_folders:
            continue
        os.makedirs('%s/vertical_collapse'%dest_folder)
        child_processes = []
        for rank in range(num_workers):
            subcircuit_kron_terms_file = '%s/rank_%d/subcircuit_kron_terms.txt'%(dest_folder,rank)
            p = subprocess.Popen(args=['./vertical_collapse', '%s'%subcircuit_kron_terms_file, '%s'%dest_folder, '%s'%meas_folder, '%d'%early_termination, '%d'%rank])
            child_processes.append(p)
        
        elapsed = 0
        for rank in range(num_workers):
            cp = child_processes[rank]
            cp.wait()
            rank_logs = open('%s/rank_%d/summary.txt'%(dest_folder,rank), 'r')
            lines = rank_logs.readlines()
            assert lines[0].split(' = ')[0]=='Total vertical_collapse time' and lines[1] == 'DONE'
            elapsed = max(elapsed,float(lines[0].split(' = ')[1]))
        
        time_str = colored('%d-q %s on %d-q cc_size took %.3e seconds'%(full_circ_size,circuit_type,cc_size,elapsed),'blue')
        print(time_str,flush=True)
        pickle.dump({'vertical_collapse_time':elapsed}, open('%s/summary.pckl'%(dest_folder),'wb'))

def bulid(circuit_type,all_circ_sizes,cc_size,techniques,eval_mode):
    _, _, num_workers, _ = techniques
    subprocess.run(['rm','./build'])
    subprocess.run(['icc','-mkl','./build.c','-o','./build','-lm'])
    for full_circ_size in all_circ_sizes:
        dest_folder = get_dirname(circuit_type=circuit_type,cc_size=cc_size,full_circ_size=full_circ_size,
        techniques=techniques,eval_mode=eval_mode,field='rank')
        if not os.path.exists('%s'%dest_folder):
            continue
        child_processes = []
        for rank in range(num_workers):
            summation_terms_file = '%s/rank_%d/summation_terms.txt'%(dest_folder,rank)
            p = subprocess.Popen(args=['./build', '%s'%summation_terms_file, '%s/vertical_collapse'%dest_folder, '%s'%dest_folder, '%d'%rank])
            child_processes.append(p)
        
        elapsed = 0
        for rank in range(num_workers):
            cp = child_processes[rank]
            cp.wait()
            rank_logs = open('%s/rank_%d/summary.txt'%(dest_folder,rank), 'r')
            lines = rank_logs.readlines()
            # print('%s/rank_%d/summary.txt'%(dest_folder,rank))
            # print(lines[-2].split(' = ')[0])
            # print(lines[-1])
            assert lines[-2].split(' = ')[0]=='Total build time' and lines[-1] == 'DONE'
            elapsed = max(elapsed,float(lines[-2].split(' = ')[1]))

        time_str = colored('%d-q %s on %d-q cc_size took %.3e seconds'%(full_circ_size,circuit_type,cc_size,elapsed),'blue')
        print(time_str,flush=True)
        pickle.dump({'build_time':elapsed}, open('%s/summary.pckl'%(dest_folder),'ab'))

def verify(circuit_type,all_circ_sizes,cc_size,techniques,eval_mode):
    subprocess.run(['python','./verify.py',
    '--circuit_type',circuit_type,
    '--circ_sizes',*[str(x) for x in all_circ_sizes],
    '--cc_size',str(cc_size),
    '--techniques',*[str(x) for x in techniques],
    '--eval_mode',eval_mode])

def plot(cc_size):
    print('-'*50,'Plot','-'*50,flush=True)
    # subprocess.run(['python','-m','partial_c_clean.plot',
    # '--circuit-type',circuit_type,
    # '--device-size',str(cc_size),
    # '--qubit-limit',str(qubit_limit),
    # '--target-folder',target_folder])

    subprocess.run(['python','expand_plot.py',
    '--cc_size',str(cc_size)])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Worker instance')
    parser.add_argument('--circuit_type', type=str, choices=['supremacy_linear','supremacy_grid','hwea','bv','aqft','adder'])
    parser.add_argument('--stage', metavar='S', type=str,choices=['generator','evaluator','process','verify','plot'])
    parser.add_argument('--size_range', nargs='+', type=int,help='(min, max) full circuit size to run')
    parser.add_argument('--cc_size', metavar='N', type=int,help='CC size to use')
    parser.add_argument('--techniques', nargs='+', type=int,default=None,help='Techniques : smart_order, early_termination, num_workers, qubit_limit')
    parser.add_argument('--eval_mode', type=str,default=None,help='Quantum evaluation backend mode')
    args = parser.parse_args()

    if args.stage!='generator':
        assert args.eval_mode=='qasm' or args.eval_mode=='runtime' or 'ibmq' in args.eval_mode

    full_circ_sizes = []
    for full_circ_size in range(args.size_range[0],args.size_range[1]+1):
        if args.circuit_type != 'supremacy_grid' and full_circ_size%2!=0:
            continue
        elif full_circ_size<=args.cc_size:
            continue
        else:
            full_circ_sizes.append(full_circ_size)

    if args.stage=='generator':
        print('-'*50,'Generator','-'*50,flush=True)
        generator(circuit_type=args.circuit_type,
        all_circ_sizes=full_circ_sizes,
        cc_size=args.cc_size)
    elif args.stage=='evaluator':
        print('-'*50,'Evaluator','-'*50,flush=True)
        evaluator(circuit_type=args.circuit_type,
        all_circ_sizes=full_circ_sizes,
        cc_size=args.cc_size,
        eval_mode=args.eval_mode)

        # print('-'*50,'Measure','-'*50,flush=True)
        # measure(circuit_type=args.circuit_type,all_circ_sizes=full_circ_sizes,cc_size=args.cc_size,eval_mode=args.eval_mode)
    elif args.stage=='process':
        print('-'*50,'Distribute Workload','-'*50,flush=True)
        distributor(circuit_type=args.circuit_type,
        all_circ_sizes=full_circ_sizes,
        cc_size=args.cc_size,
        techniques=args.techniques,
        eval_mode=args.eval_mode)

        # print('-'*50,'Vertical Collapse','-'*50,flush=True)
        # vertical_collapse(circuit_type=args.circuit_type,all_circ_sizes=full_circ_sizes,
        # cc_size=args.cc_size,techniques=args.techniques,eval_mode=args.eval_mode)

        # print('-'*50,'Build','-'*50,flush=True)
        # bulid(circuit_type=args.circuit_type,all_circ_sizes=full_circ_sizes,cc_size=args.cc_size,techniques=args.techniques,eval_mode=args.eval_mode)
    
    elif args.stage=='verify':
        print('-'*50,'Verify','-'*50,flush=True)
        verify(circuit_type=args.circuit_type,all_circ_sizes=full_circ_sizes,cc_size=args.cc_size,techniques=args.techniques,eval_mode=args.eval_mode)

    elif args.stage=='plot':
        plot(cc_size=args.cc_size)

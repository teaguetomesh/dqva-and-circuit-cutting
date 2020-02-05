import pickle
import matplotlib.pyplot as plt
import numpy as np
from utils.helper_fun import get_filename, read_file
from utils.metrics import fidelity, chi2_distance
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Check output')
    parser.add_argument('--experiment-name', metavar='S', type=str,help='which experiment to run')
    parser.add_argument('--device-name', metavar='S', type=str,help='which evaluator device output file to reconstruct')
    parser.add_argument('--circuit-type', metavar='S', type=str,help='which circuit input file to run')
    parser.add_argument('--shots-mode', metavar='S', type=str,help='saturated/sametotal shots mode')
    parser.add_argument('--evaluation-method', metavar='S', type=str,help='which evaluator backend file to reconstruct')
    args = parser.parse_args()

    dirname, plotter_input_filename = get_filename(experiment_name=args.experiment_name,circuit_type=args.circuit_type,device_name=args.device_name,field='plotter_input',evaluation_method=args.evaluation_method,shots_mode=args.shots_mode)
    plotter_input = read_file(dirname+plotter_input_filename)
    print(plotter_input.keys())

    case = (4,5)
    plotter_input = plotter_input[case]

    d1 = plotter_input['sv']
    d2 = plotter_input['qasm']
    if args.experiment_name == 'simulator':
        d3 = plotter_input['qasm+noise']
    else:
        d3 = plotter_input['hw']
    d4 = [abs(x) for x in plotter_input['cutting']]
    
    ground_truth_metric = chi2_distance(target=d1,obs=d1)
    cutting_metric = chi2_distance(target=d1,obs=d4)
    std_metric = chi2_distance(target=d1,obs=d3)
    metric_percent_change = 100*(std_metric - cutting_metric)/(std_metric - ground_truth_metric)

    ground_truth_fid = fidelity(target=d1,obs=d1)
    cutting_fid = fidelity(target=d1,obs=d4)
    std_fid = fidelity(target=d1,obs=d3)
    fid_percent_change = 100*(cutting_fid-std_fid)/std_fid
    print('std fid = %.3f, cutting fid = %.3f'%(std_fid,cutting_fid))
    print('metric reduction =',metric_percent_change,'fid improvement =',fid_percent_change)

    plot_range = min(1024,len(d1))
    x = np.arange(len(d1))[:plot_range]
    y_lim = 0
    for d in [d1,d2,d3,d4]:
        y_lim = max(y_lim,max(d))
    y_lim *= 1.1

    plt.figure(figsize=(10,5))
    plt.subplot(221)
    plt.bar(x,height=d1[:plot_range],label='ground truth, fid = %.3e, metric = %.3e'%(fidelity(d1,d1),chi2_distance(d1,d1)))
    plt.ylim(0,y_lim)
    plt.xlabel('quantum state')
    plt.ylabel('probability')
    plt.legend()


    plt.subplot(222)
    plt.bar(x,height=d4[:plot_range],label='cutting mode, fid = %.3e, metric = %.3e'%(fidelity(d1,d4),chi2_distance(d1,d4)))
    plt.ylim(0,y_lim)
    plt.xlabel('quantum state')
    plt.ylabel('probability')
    plt.legend()

    plt.subplot(223)
    plt.bar(x,height=d3[:plot_range],label='standard mode, fid = %.3e, metric = %.3e'%(fidelity(d1,d3),chi2_distance(d1,d3)))
    plt.ylim(0,y_lim)
    plt.xlabel('quantum state')
    plt.ylabel('probability')
    plt.legend()

    plt.subplot(224)
    plt.bar(x,height=d2[:plot_range],label='noiseless qasm, fid = %.3e, metric = %.3e'%(fidelity(d1,d2),chi2_distance(d1,d2)))
    plt.ylim(0,y_lim)
    plt.xlabel('quantum state')
    plt.ylabel('probability')
    plt.suptitle('Case {}'.format(case))
    plt.legend()
    plt.savefig('%s/check_output_eg.png'%dirname,dpi=400)
    plt.close()

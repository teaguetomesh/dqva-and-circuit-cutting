import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from utils.helper_fun import get_filename, read_file
import os
from scipy import optimize
import argparse

def exp_func(x, a, b):
    return a * np.exp(b*x)

def get_hw_sizes(cases):
    organized_cases = {}
    for case in cases:
        if case[0] not in organized_cases:
            organized_cases[case[0]] = [case]
        else:
            organized_cases[case[0]].append(case)
    return organized_cases

def get_fc_sizes(plotter_input):
    unique_fc_sizes = []
    std_times = []
    for case in plotter_input:
        if case[1] not in unique_fc_sizes:
            unique_fc_sizes.append(case[1])
            std_times.append(plotter_input[case]['std_time'])
    unique_fc_sizes, std_times = zip(*sorted(zip(unique_fc_sizes, std_times)))
    return unique_fc_sizes, std_times

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='generate evaluator inputs')
    parser.add_argument('--circuit-type', metavar='S', type=str,help='which circuit input file to run')
    args = parser.parse_args()

    dirname, plotter_input_filename = get_filename(experiment_name='large_on_small',circuit_type=args.circuit_type,device_name='ibmq_boeblingen',
    field='plotter_input',evaluation_method='statevector_simulator')

    plotter_input = read_file(dirname+plotter_input_filename)
    print(plotter_input_filename)
    print(plotter_input.keys())
    organized_cases = get_hw_sizes(cases=plotter_input.keys())
    print(organized_cases)
    unique_fc_sizes, std_times = get_fc_sizes(plotter_input=plotter_input)
    print(unique_fc_sizes)
    print(std_times)

    plt.figure()
    plt.xticks([x for x in unique_fc_sizes])
    plt.plot(unique_fc_sizes,std_times,marker='x',label='Classical')
    for hw_size in organized_cases:
        print('plotting curve hw-%d'%hw_size)
        xvals = []
        yvals = []
        for case in organized_cases[hw_size]:
            case_dict = plotter_input[case]
            xvals.append(case[1])
            yvals.append(case_dict['hybrid_time'])
            num_clusters = len(case_dict['clusters'])
            print('case {} std_time = {}'.format(case,case_dict['std_time']))
        plt.plot(xvals,yvals,marker='o',label='%d-hw'%hw_size)
    plt.xlabel('Circuit Size')
    plt.ylabel('Hybrid Time (s)')
    plt.title('%d Clusters'%num_clusters)
    plt.legend()
    plt.savefig(dirname+'%d_cluster_hybrid.png'%num_clusters,dpi=400)
    plt.close()
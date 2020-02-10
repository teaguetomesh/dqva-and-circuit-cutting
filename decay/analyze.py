import matplotlib.pyplot as plt
import argparse
from utils.helper_fun import read_file, evaluate_circ
from utils.conversions import dict_to_array
import math
import numpy as np

def get_xticks(xvals,compulsory):
    if len(xvals)<=10:
        return xvals
    else:
        x_ticks = []
        step = math.ceil(len(xvals)/10)
        for idx, x in enumerate(xvals):
            if idx%step==0 or idx==len(xvals)-1 or x in compulsory:
                x_ticks.append(x)
        return x_ticks

def make_plot(metrics_list,cutoffs,full_circ_size,shots_increment,derivative_thresholds):
    first_derivative_threshold, second_derivative_threshold = derivative_thresholds
    fig, ax1 = plt.subplots()
    max_xval = 0
    for metric_l in metrics_list:
        max_xval = max(max_xval,len(metric_l))
    xvals = range(1,max_xval+1)
    x_ticks = get_xticks(xvals=xvals,compulsory=cutoffs)
    plt.xticks(ticks=x_ticks,labels=x_ticks)
    
    metric_l = metrics_list[0]
    cutoff = cutoffs[0]
    xvals = range(1,len(metric_l)+1)
    ax1.plot(xvals,metric_l,color='b')
    ax1.axvline(x=cutoffs[0],label='noiseless cutoff = %d'%cutoffs[0] if len(metrics_list[0])>=cutoffs[0]+1 else 'noiseless diverged = %d'%cutoffs[0],color='b',linestyle='--')
    ax1.set_ylabel('noiseless metric, lower is better',color='b')
    ax1.set_xlabel('shots [*%d]'%shots_increment)
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.legend()

    # ax2 = ax1.twinx()
    # metric_l = metrics_list[1]
    # cutoff = cutoffs[1]
    # xvals = range(1,len(metric_l)+1)
    # ax2.axvline(x=cutoffs[1],label='noisy cutoff = %d'%cutoffs[1] if len(metrics_list[1])>=cutoffs[1]+1 else 'noisy diverged = %d'%cutoffs[1],color='r',linestyle='--')
    # ax2.plot(xvals,metric_l,color='r')
    # ax2.set_ylabel('noisy metric, lower is better',color='r')
    # ax2.tick_params(axis='y', labelcolor='r')
    # ax2.legend()
    
    plt.title('%d qubit circuit, derivative_thresholds : %.3e, %.3e'%(full_circ_size,first_derivative_threshold,second_derivative_threshold))
    # plt.legend()
    plt.tight_layout()
    plt.savefig('decay/%d_decay.png'%(full_circ_size),dpi=400)
    plt.close()

def find_saturation(metrics_list,derivative_thresholds):
    cutoffs = []
    for metric_l in metrics_list:
        assert len(metric_l)>=3
        first_derivative_threshold, second_derivative_threshold = derivative_thresholds
        for cutoff in range(1,len(metric_l)-1):
            first_derivative = (metric_l[cutoff+1]-metric_l[cutoff-1])/2
            second_derivative = (metric_l[cutoff+1]+metric_l[cutoff-1]-2*metric_l[cutoff])/2
            # print('chi2 = %.3f, first derivative = %.3e, second derivative = %.3e'%(metric_l[cutoff],first_derivative,second_derivative),flush=True)
            if abs(first_derivative)<first_derivative_threshold and abs(second_derivative)<second_derivative_threshold:
                break
        cutoff += 1
        cutoffs.append(cutoff)
    return cutoffs

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='generate evaluator inputs')
    parser.add_argument('--first-derivative', metavar='N', type=float,help='First derivative threshold')
    parser.add_argument('--second-derivative', metavar='N', type=float,help='Second derivative threshold')
    args = parser.parse_args()

    decay_dict = read_file(filename='./decay/decay.pickle')

    for full_circ_size in decay_dict:
        circ = decay_dict[full_circ_size]['circ']
        noiseless_chi2_l = decay_dict[full_circ_size]['noiseless_chi2_l']
        shots_increment = decay_dict[full_circ_size]['shots_increment']
        cutoffs = find_saturation(metrics_list=[noiseless_chi2_l],derivative_thresholds=(args.first_derivative,args.second_derivative))
        print('%d qubit circuit, noiseless cutoff = %d'%(full_circ_size,cutoffs[0]),flush=True)
        make_plot(metrics_list=[noiseless_chi2_l],cutoffs=cutoffs,
        full_circ_size=full_circ_size,shots_increment=shots_increment,derivative_thresholds=(args.first_derivative,args.second_derivative))
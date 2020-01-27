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
            if idx%step==0 or idx==len(xvals)-1 or x==compulsory:
                x_ticks.append(x)
        return x_ticks

def make_plot(metric_l,cutoff,full_circ_size,shots_increment,derivative_thresholds):
    first_derivative_threshold, second_derivative_threshold = derivative_thresholds
    xvals = range(1,len(metric_l)+1)
    plt.figure()
    plt.axvline(x=cutoff,label='saturated cutoff = %d'%cutoff if len(metric_l)>=cutoff+1 else 'diverged cutoff = %d'%cutoff,color='k',linestyle='--')
    plt.plot(xvals,metric_l,label='noiseless')
    x_ticks = get_xticks(xvals=xvals,compulsory=cutoff)
    plt.xticks(ticks=x_ticks,labels=x_ticks)
    plt.ylabel('\u03C7^2, lower is better')
    plt.xlabel('shots [*%d]'%shots_increment)
    plt.title('%d qubit circuit, derivative_thresholds : %.3e, %.3e'%(full_circ_size,first_derivative_threshold,second_derivative_threshold))
    plt.legend()
    plt.tight_layout()
    plt.savefig('decay/%d_decay.png'%(full_circ_size),dpi=400)
    plt.close()

def find_saturation(metric_l,derivative_thresholds,shots_increment):
    assert len(metric_l)>=3
    first_derivative_threshold, second_derivative_threshold = derivative_thresholds
    for cutoff in range(1,len(metric_l)-1):
        first_derivative = (metric_l[cutoff+1]-metric_l[cutoff-1])/(2*shots_increment)
        second_derivative = (metric_l[cutoff+1]+metric_l[cutoff-1]-2*metric_l[cutoff])/(2*np.power(shots_increment,2))
        # print('chi2 = %.3f, first derivative = %.3e, second derivative = %.3e'%(metric_l[cutoff],first_derivative,second_derivative),flush=True)
        if abs(first_derivative)<first_derivative_threshold and abs(second_derivative)<second_derivative_threshold:
            break
    cutoff += 1
    return cutoff, abs(first_derivative), abs(second_derivative)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='generate evaluator inputs')
    parser.add_argument('--first-derivative', metavar='N', type=float,help='First derivative threshold')
    parser.add_argument('--second-derivative', metavar='N', type=float,help='Second derivative threshold')
    args = parser.parse_args()

    decay_dict = read_file(filename='./decay/decay.pickle')

    for full_circ_size in decay_dict:
        circ = decay_dict[full_circ_size]['circ']
        chi2_l = decay_dict[full_circ_size]['chi2_l']
        metric_l = []
        for chi2 in chi2_l:
            metric_l.append(chi2)
        shots_increment = decay_dict[full_circ_size]['shots_increment']
        cutoff, first_derivative, second_derivative = find_saturation(metric_l=metric_l,derivative_thresholds=(args.first_derivative,args.second_derivative),shots_increment=shots_increment)
        print('%d qubit circuit, cutoff = %d, \u03C7^2 = %.3f, first derivative = %.3e, second derivative = %.3e'%(full_circ_size,cutoff,metric_l[cutoff],first_derivative,second_derivative),flush=True)
        make_plot(metric_l=metric_l,cutoff=cutoff,full_circ_size=full_circ_size,shots_increment=shots_increment,derivative_thresholds=(args.first_derivative,args.second_derivative))
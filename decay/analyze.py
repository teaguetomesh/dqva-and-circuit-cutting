import matplotlib.pyplot as plt
import argparse
from utils.helper_fun import read_file
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

def make_plot(ce_l,cutoff,full_circ_size,shots_increment,derivative_thresholds):
    first_derivative_threshold, second_derivative_threshold = derivative_thresholds
    xvals = range(1,len(ce_l)+1)
    plt.figure()
    plt.axvline(x=cutoff,label='saturated cutoff' if len(ce_l)>cutoff+2 else 'diverged cutoff',color='k',linestyle='--')
    plt.plot(xvals,ce_l,label='noiseless')
    x_ticks = get_xticks(xvals=xvals,compulsory=cutoff)
    plt.xticks(ticks=x_ticks,labels=x_ticks)
    plt.ylabel('\u0394H, lower is better')
    plt.xlabel('shots [*%d]'%shots_increment)
    plt.title('%d qubit circuit, derivative_thresholds : %.3e, %.3e'%(full_circ_size,first_derivative_threshold,second_derivative_threshold))
    plt.legend()
    plt.tight_layout()
    plt.savefig('decay/%d_decay.png'%(full_circ_size),dpi=400)
    plt.close()

def find_saturation(ce_l,derivative_thresholds,shots_increment):
    assert len(ce_l)>=3
    first_derivative_threshold, second_derivative_threshold = derivative_thresholds
    for cutoff in range(1,len(ce_l)-1):
        first_derivative = (ce_l[cutoff+1]-ce_l[cutoff-1])/(2*shots_increment)
        second_derivative = (ce_l[cutoff+1]+ce_l[cutoff-1]-2*ce_l[cutoff])/(2*np.power(shots_increment,2))
        # print('\u0394H = %.3f, first derivative = %.3e, second derivative = %.3e'%(ce_l[cutoff],first_derivative,second_derivative),flush=True)

        if abs(first_derivative)<first_derivative_threshold and abs(second_derivative)<second_derivative_threshold:
            break
    return cutoff, abs(first_derivative), abs(second_derivative)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='generate evaluator inputs')
    parser.add_argument('--first-derivative', metavar='N', type=float,help='First derivative threshold')
    parser.add_argument('--second-derivative', metavar='N', type=float,help='Second derivative threshold')
    args = parser.parse_args()

    decay_dict = read_file(filename='./decay/decay.pickle')

    for full_circ_size in decay_dict:
        ce_l = decay_dict[full_circ_size]['ce_l']
        shots_increment = decay_dict[full_circ_size]['shots_increment']
        cutoff, first_derivative, second_derivative = find_saturation(ce_l=ce_l,derivative_thresholds=(args.first_derivative,args.second_derivative),shots_increment=shots_increment)
        print('%d qubit circuit, cutoff = %d, \u0394H = %.3f, first derivative = %.3e, second derivative = %.3e'%(full_circ_size,cutoff,ce_l[cutoff],first_derivative,second_derivative),flush=True)
        make_plot(ce_l=ce_l,cutoff=cutoff,full_circ_size=full_circ_size,shots_increment=shots_increment,derivative_thresholds=(args.first_derivative,args.second_derivative))
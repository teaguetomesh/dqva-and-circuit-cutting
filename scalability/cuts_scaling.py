from utils.helper_fun import generate_circ
import utils.MIQCP_searcher as searcher
from time import time
import matplotlib.pyplot as plt
import math

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

def make_plot(fc_sizes,num_cuts,searcher_time):
    fig, ax1 = plt.subplots()
    x_ticks = get_xticks(xvals=fc_sizes,compulsory=fc_sizes)
    plt.xticks(ticks=x_ticks,labels=x_ticks)
    plt.yticks(ticks=num_cuts,labels=num_cuts)
    
    ax1.plot(fc_sizes,num_cuts,marker='x',color='b')
    ax1.set_ylabel('Number of cuts')
    ax1.set_xlabel('full circuit size')
    ax1.tick_params(axis='y', labelcolor='b')

    ax2 = ax1.twinx()
    ax2.plot(fc_sizes,searcher_time,marker='o',color='r')
    ax2.set_ylabel('MIP Solver Time (s)',color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    # ax2.set_yscale('log')
    ax2.legend()
    
    plt.title('')
    plt.legend()
    plt.tight_layout()
    plt.savefig('./scalability/cuts_scaling.png',dpi=400)
    plt.close()

num_cuts = []
fc_sizes = []
searcher_times = []
for fc_size in range(10,201,10):
    circ = generate_circ(full_circ_size=fc_size,circuit_type='supremacy')
    max_clusters = 3
    cluster_max_qubit = int(fc_size/1.5)
    min_objective, best_positions, num_rho_qubits, num_O_qubits, num_d_qubits, best_num_cluster, m, searcher_time = searcher.find_cuts(circ=circ,
    reconstructor_runtime_params=[4.275e-9,6.863e-1],reconstructor_weight=0,
    num_clusters=range(2,min(len(circ.qubits),max_clusters)+1),cluster_max_qubit=cluster_max_qubit)

    if m != None:
        # m.print_stat()
        fc_sizes.append(fc_size)
        num_cuts.append(len(best_positions))
        searcher_times.append(searcher_time)
        print('%d-on-%d'%(fc_size,cluster_max_qubit))
        print('{:d} cuts --> {}, searcher time = {}'.format(len(best_positions),num_d_qubits,searcher_time))
    else:
        print('NOT feasible')

print(fc_sizes)
print(num_cuts)
print(searcher_times)
make_plot(fc_sizes,num_cuts,searcher_times)

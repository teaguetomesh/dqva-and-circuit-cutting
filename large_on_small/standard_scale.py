from utils.helper_fun import generate_circ, evaluate_circ
from utils.conversions import dict_to_array, list_to_dict
from time import time
from qiskit import Aer, execute
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

def exp_func(x, a, b):
    return a * np.exp(b*x)

if __name__ == '__main__':
    fc_sizes = range(10,25,2)
    std_times = []
    for fc_size in fc_sizes:
        circ = generate_circ(full_circ_size=fc_size,circuit_type='supremacy')
        
        std_begin = time()
        backend = Aer.get_backend('statevector_simulator')
        job = execute(circ, backend=backend)
        result = job.result()
        outputstate = result.get_statevector(circ)
        outputstate_dict = list_to_dict(l=outputstate)
        for key in outputstate_dict:
            x = outputstate_dict[key]
            outputstate_dict[key] = np.absolute(x)**2
        ground_truth = dict_to_array(distribution_dict=outputstate_dict,force_prob=True)
        std_time = time() - std_begin
        print('%d qubit circuit took %.3f seconds'%(fc_size,std_time))
        std_times.append(std_time)
    plt.figure()
    plt.plot(fc_sizes,std_times,'*',label='classical state vector simulation')
    params, params_covariance = optimize.curve_fit(exp_func, fc_sizes, std_times, p0=[1, 1])
    extrapolation_range = np.arange(min(fc_sizes),max(fc_sizes)+3)
    x_vals = np.arange(min(fc_sizes),max(fc_sizes)+3,0.1)
    plt.plot(x_vals, exp_func(np.array(x_vals), params[0], params[1]),'r--',label='%.3eexp(%.3e*x)'%(params[0],params[1]))
    plt.legend()
    plt.xticks(extrapolation_range, extrapolation_range)
    plt.xlabel('Circuit size')
    plt.ylabel('Simulation time (s)')
    plt.savefig('standard_runtime_trend.png',dpi=400)
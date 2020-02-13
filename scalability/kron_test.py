import numpy as np
from time import time
import matplotlib.pyplot as plt
from scipy import optimize

def exp_func(x, a, b):
    return a * np.exp(b*x)

num_qubits = range(20,33,2)
times = []
for num_qubit in num_qubits:
    arr_len = int(num_qubit/2)
    arr = np.random.rand(2**arr_len)
    
    begin = time()
    np.kron(arr,arr)
    end = time()-begin
    
    print('%d qubit took %.3f seconds'%(num_qubit,end))
    times.append(end)

plt.figure()
plt.plot(num_qubits,times,'*',label='kron')
params, params_covariance = optimize.curve_fit(exp_func, num_qubits, times, p0=[1, 1])
extrapolation_range = np.arange(min(num_qubits),max(num_qubits)+3)
x_vals = np.arange(min(num_qubits),max(num_qubits)+3,0.1)
plt.plot(x_vals, exp_func(np.array(x_vals), params[0], params[1]),'r--',label='%.3eexp(%.3e*x)'%(params[0],params[1]))
plt.legend()
plt.xticks(extrapolation_range, extrapolation_range)
plt.xlabel('Total length')
plt.ylabel('Kron time (s)')
plt.savefig('./scalability/kron_time_trend.png',dpi=400)

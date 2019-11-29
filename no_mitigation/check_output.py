import pickle
import matplotlib.pyplot as plt
import numpy as np
from helper_fun import cross_entropy, fidelity

f = open('./benchmark_data/hwea/quantum_plotter_input_ibmq_boeblingen_hwea_saturated.p', 'rb' )
plotter_input = {}
while 1:
    try:
        plotter_input.update(pickle.load(f))
    except (EOFError):
        break
f.close()

print(plotter_input.keys())

case = (7,14)
plotter_input = plotter_input[case]
print(plotter_input.keys())
ce_percent_change = plotter_input['ce_percent_reduction']
fid_percent_change = plotter_input['fid_percent_improvement']
circ = plotter_input['full_circ']
d1 = plotter_input['evaluations']['sv_noiseless']
d2 = plotter_input['evaluations']['qasm']
d3 = plotter_input['evaluations']['qasm+noise']
d4 = [abs(x) for x in plotter_input['evaluations']['cutting']]
std_fid = fidelity(target=d1,obs=d3)
cutting_fid = fidelity(target=d1,obs=d4)
print('std fid = %.3f, cutting fid = %.3f'%(std_fid,cutting_fid))
print('ce:',ce_percent_change,'fid:',fid_percent_change)

plot_range = min(1024,len(d1))
x = np.arange(len(d1))[:plot_range]
y_lim = 0
for d in [d1,d2,d3,d4]:
    y_lim = max(y_lim,max(d))
y_lim *= 1.1

plt.figure(figsize=(10,5))
plt.subplot(221)
plt.bar(x,height=d1[:plot_range],label='ground truth, fid = %.3e, \u0394H = %.3e'%(fidelity(d1,d1),cross_entropy(d1,d1)))
plt.ylim(0,y_lim)
plt.xlabel('quantum state')
plt.ylabel('probability')
plt.legend()


plt.subplot(222)
plt.bar(x,height=d4[:plot_range],label='cutting mode, fid = %.3e, \u0394H = %.3e'%(fidelity(d1,d4),cross_entropy(d1,d4)))
plt.ylim(0,y_lim)
plt.xlabel('quantum state')
plt.ylabel('probability')
plt.legend()

plt.subplot(223)
plt.bar(x,height=d3[:plot_range],label='standard mode, fid = %.3e, \u0394H = %.3e'%(fidelity(d1,d3),cross_entropy(d1,d3)))
plt.ylim(0,y_lim)
plt.xlabel('quantum state')
plt.ylabel('probability')
plt.legend()

plt.subplot(224)
plt.bar(x,height=d2[:plot_range],label='noiseless qasm, fid = %.3e, \u0394H = %.3e'%(fidelity(d1,d2),cross_entropy(d1,d2)))
plt.ylim(0,y_lim)
plt.xlabel('quantum state')
plt.ylabel('probability')
plt.legend()
plt.savefig('check_output_eg.png',dpi=400)
plt.close()
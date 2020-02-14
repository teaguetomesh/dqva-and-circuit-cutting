import matplotlib.pyplot as plt
import numpy as np

searcher_time = np.array([0.464,0.464,0.464])
qc_time = np.array([0.096,0.096,0.096])

compute_time = np.array([11.757,2.637,0.586])
reorder_time = np.array([5.772,5.046,1.039])

reverse_time = np.array([0.899,0.832,1.064])

ind = np.arange(len(compute_time))    # the x locations for the groups
width = 0.35       # the width of the bars: can also be len(x) sequence

p1 = plt.bar(ind, qc_time, width)
p2 = plt.bar(ind, compute_time, width, bottom=qc_time)
p3 = plt.bar(ind, reorder_time, width,bottom=compute_time+qc_time)

plt.ylabel('Time (s)')
plt.title('20-qubit BV circuit ')
plt.xticks(ind, ('Single thread', 'Smart order', 'Parallel'))
# plt.yticks(np.arange(0, 81, 10))
plt.legend((p1[0], p2[0]), ('compute', 'reorder'))

plt.savefig('./large_on_small/techniques_demonstration.png',dpi=400)
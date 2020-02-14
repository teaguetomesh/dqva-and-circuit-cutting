import matplotlib.pyplot as plt
import numpy as np

searcher_time = np.array([0,1.683,1.683,1.683])
qc_time = np.array([0,0.096,0.096,3.16])

compute_time = np.array([156.533,11.757,2.637,36.527])
reorder_time = np.array([0,5.772,5.046,11.96])

reverse_time = np.array([0,0.899,0.832,16.79])

ind = np.arange(len(compute_time))    # the x locations for the groups
width = 0.35       # the width of the bars: can also be len(x) sequence

p1 = plt.bar(ind, searcher_time, width)
p2 = plt.bar(ind, qc_time, width, bottom=searcher_time)
p3 = plt.bar(ind, compute_time, width, bottom=searcher_time+qc_time)
p4 = plt.bar(ind, reorder_time, width,bottom=searcher_time+qc_time+compute_time)

plt.ylabel('Time (s)')
plt.title('24-qubit Supremacy Circuit Cut into [2,2,11,15] Clusters')
plt.xticks(ind, ('Classical', 'Single Thread', 'Smart Order', 'Parallel'))
# plt.yticks(np.arange(0, 81, 10))
plt.legend((p1[0], p2[0], p3[0], p4[0]), ('searcher', 'qc', 'compute', 'reorder'))

plt.savefig('./large_on_small/techniques_demonstration.png',dpi=400)

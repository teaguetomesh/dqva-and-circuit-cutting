import pickle
import glob
import os

benchmark_l = []

for filename in glob.glob('./benchmark_data/*_uniter_output_*.p'):
    benchmark = pickle.load(open(filename, 'rb' ))
    benchmark_l.append(benchmark)

filename = filename.split('_clusters_')[0]
filename = filename+'_clusters.p'
filename = filename.replace('uniter_output','plotter_input')
pickle.dump(benchmark_l,open('%s'%filename,'ab'))
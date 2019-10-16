import pickle
import glob
import os

benchmark_l = []

for filename in glob.glob('./benchmark_data/*_uniter_output_*.p'):
    benchmark = pickle.load(open(filename, 'rb' ))
    benchmark_l.append(benchmark)

arr = filename[:-2].split('_')
new_arr = []
for x in arr:
    if '*' not in x:
        new_arr.append(x)
new_file_name = ''
for x in new_arr:
    new_file_name+=x+'_'
new_file_name = new_file_name[:-1]
new_file_name = new_file_name.replace('uniter_output','plotter_input')+'.p'
pickle.dump(benchmark_l,open('%s'%new_file_name,'ab'))
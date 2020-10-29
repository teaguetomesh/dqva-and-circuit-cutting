def get_dirname(circuit_name,cc_size,early_termination,eval_mode,num_workers,qubit_limit,field):
    if field=='generator':
        dirname = './source_data/%s/cc_%d'%(circuit_name,cc_size)
    elif field=='evaluator':
        dirname = './source_data/%s/cc_%d/%s'%(circuit_name,cc_size,eval_mode)
    elif field=='vertical_collapse':
        dirname = './source_data/%s/cc_%d/%s/et_%d'%(circuit_name,cc_size,eval_mode,early_termination)
    elif field=='build':
        dirname = './processed_data/%s/cc_%d/%s_%d_%d_%d'%(circuit_name,cc_size,eval_mode,early_termination,qubit_limit,num_workers)
    elif field=='slurm':
        dirname = './slurm/%s/cc_%d'%(circuit_name,cc_size)
    elif field=='runtime':
        dirname = './runtime/%s/cc_%d/q_%d_%d'%(circuit_name,cc_size,qubit_limit,num_workers)
    else:
        raise Exception('Illegal field = %s'%field)
    return dirname
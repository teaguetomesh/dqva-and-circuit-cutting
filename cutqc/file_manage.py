def get_dirname(circuit_name,max_subcircuit_qubit,early_termination,eval_mode,num_workers,qubit_limit,field):
    if field=='cutter':
        dirname = './source_data/%s/cc_%d'%(circuit_name,max_subcircuit_qubit)
    elif field=='evaluator':
        dirname = './source_data/%s/cc_%d/%s'%(circuit_name,max_subcircuit_qubit,eval_mode)
    elif field=='vertical_collapse':
        dirname = './source_data/%s/cc_%d/%s/et_%d'%(circuit_name,max_subcircuit_qubit,eval_mode,early_termination)
    elif field=='build':
        dirname = './processed_data/%s/cc_%d/%s_%d_%d_%d'%(circuit_name,max_subcircuit_qubit,eval_mode,early_termination,qubit_limit,num_workers)
    elif field=='slurm':
        dirname = './slurm/%s/cc_%d'%(circuit_name,max_subcircuit_qubit)
    elif field=='runtime':
        dirname = './runtime/%s/cc_%d/q_%d_%d'%(circuit_name,max_subcircuit_qubit,qubit_limit,num_workers)
    else:
        raise Exception('Illegal field = %s'%field)
    return dirname
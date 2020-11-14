from qiskit.converters import circuit_to_dag

def check_valid(circuit):
    valid = circuit.num_unitary_factors()==1
    dag = circuit_to_dag(circuit)
    for op_node in dag.topological_op_nodes():
        if len(op_node.qargs)>2 and op_node.op.name!='barrier':
            valid = False
            break
    return valid

def get_dirname(circuit_name,max_subcircuit_qubit,early_termination,eval_mode,num_threads,qubit_limit,field):
    if field=='cutter':
        dirname = './cutqc_data/%s/cc_%d'%(circuit_name,max_subcircuit_qubit)
    elif field=='evaluator':
        dirname = './cutqc_data/%s/cc_%d/%s'%(circuit_name,max_subcircuit_qubit,eval_mode)
    elif field=='vertical_collapse':
        dirname = './cutqc_data/%s/cc_%d/%s/et_%d'%(circuit_name,max_subcircuit_qubit,eval_mode,early_termination)
    elif field=='build':
        dirname = './cutqc_data/%s/cc_%d/%s_%d_%d_%d'%(circuit_name,max_subcircuit_qubit,eval_mode,early_termination,qubit_limit,num_threads)
    elif field=='slurm':
        dirname = './slurm/%s/cc_%d'%(circuit_name,max_subcircuit_qubit)
    elif field=='runtime':
        dirname = './runtime/%s/cc_%d/q_%d_%d'%(circuit_name,max_subcircuit_qubit,qubit_limit,num_threads)
    else:
        raise Exception('Illegal field = %s'%field)
    return dirname
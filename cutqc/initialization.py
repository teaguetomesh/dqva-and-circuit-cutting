from qiskit.converters import circuit_to_dag

def check_valid(circuit):
    valid = circuit.num_unitary_factors()==1
    dag = circuit_to_dag(circuit)
    for op_node in dag.topological_op_nodes():
        if len(op_node.qargs)>2 and op_node.op.name!='barrier':
            valid = False
            break
    return valid
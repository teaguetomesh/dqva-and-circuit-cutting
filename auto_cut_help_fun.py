from qiskit import QuantumCircuit
from qiskit.dagcircuit.dagcircuit import DAGCircuit
from qiskit.converters import circuit_to_dag, dag_to_circuit
import min_cut as cut_finder

def find_neighbor(qarg, vert_info):
    for idx, (vertex_name, qargs) in enumerate(vert_info):
        if qarg in qargs:
            return vertex_name, True
    return None, False

def circ_stripping(circ):
    # Remove all single qubit gates in the circuit
    dag = circuit_to_dag(circ)
    stripped_dag = DAGCircuit()
    [stripped_dag.add_qreg(x) for x in circ.qregs]
    for vertex in dag.topological_op_nodes():
        if len(vertex.qargs) >= 2:
            stripped_dag.apply_operation_back(op=vertex.op, qargs=vertex.qargs)
    return dag_to_circuit(stripped_dag)

def circuit_to_graph(stripped_circ):
    input_qubit_itr = {}
    for x in stripped_circ.qubits:
        input_qubit_itr[x] = 0
    stripped_dag = circuit_to_dag(stripped_circ)
    vert_info = []
    for vertex in stripped_dag.topological_op_nodes():
        vertex_name = ''
        for qarg in vertex.qargs:
            if vertex_name == '':
                vertex_name += '%s[%d]%d' % (qarg[0].name,qarg[1],input_qubit_itr[qarg])
                input_qubit_itr[qarg] += 1
            else:
                vertex_name += ',%s[%d]%d' % (qarg[0].name,qarg[1],input_qubit_itr[qarg])
                input_qubit_itr[qarg] += 1
        vert_info.append((vertex_name, vertex.qargs))
    
    abstraction = []
    for idx, (vertex_name, vertex_qargs) in enumerate(vert_info):
        abstraction_item = [vertex_name]
        for qarg in vertex_qargs:
            prev_vertex_name, prev_found = find_neighbor(qarg, vert_info[:idx][::-1])
            next_vertex_name, next_found = find_neighbor(qarg, vert_info[idx+1:])
            if prev_found:
                abstraction_item.append(prev_vertex_name)
            if next_found:
                abstraction_item.append(next_vertex_name)
        abstraction.append(abstraction_item)
    graph = cut_finder.Graph(abstraction) 
    return graph
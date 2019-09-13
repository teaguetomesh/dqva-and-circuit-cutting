import numpy as np
import copy
import networkx as nx
from qiskit import QuantumCircuit
from qiskit.dagcircuit.exceptions import DAGCircuitError
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.dagcircuit.dagnode import DAGNode
from qiskit.circuit import Measure
from qiskit.circuit.quantumregister import QuantumRegister
from qiskit.circuit.classicalregister import ClassicalRegister
from qiskit.tools.visualization import dag_drawer

def find_io_node(dag, wire):
    in_node = None
    out_node = None

    # for node in dag.nodes():
    #     if node.type == 'in' and node.name == '%s[%d]' % (wire[0].name,wire[1]):
    #         in_node = node
    #     if node.type == 'out' and node.name == '%s[%d]' % (wire[0].name,wire[1]):
    #         out_node = node
    # if in_node == None or out_node == None:
    #     raise Exception('did not find {}, in_node = {}, out_node = {}'.format(wire, in_node, out_node))
    in_node = dag.input_map[wire]
    out_node = dag.output_map[wire]
    return in_node, out_node

def find_edge_key(G, u, v, wire):
    wires = nx.get_edge_attributes(G,'wire')
    for edge in wires:
        edge_u, edge_v, edge_key = edge
        if edge_u == u and edge_v == v and wires[edge] == wire:
            return edge_key
    return None
            
def cut_edges(original_dag, positions):
    '''Cut multiple edges in the original_dag.

    Args:
        original_dag (DAGCircuit): original dag circuit to cut
        positions (list): list of cutting positions in (qubit, source noce idx) tuples

    Returns:
        DAGCircuit: dag circuit after cutting

    Raises:
        dag after cutting is not successfully splitted

    '''
    cut_dag = copy.deepcopy(original_dag)
    cutQ_register = QuantumRegister(len(positions), 'cutQ')
    cut_dag.add_qreg(cutQ_register)
    path_map = {}
    for input_qubit in original_dag.qubits():
        path_map[input_qubit] = []

    for cutQ_idx, position in enumerate(positions):
        wire, source_node_idx = position
        
        nodes_before_cut = list(cut_dag.nodes_on_wire(wire=wire, only_ops=True))[:source_node_idx+1]
        nodes_after_cut = list(cut_dag.nodes_on_wire(wire=wire, only_ops=True))[source_node_idx+1:]

        cut_qubit = cutQ_register[cutQ_idx]
        path_map[wire].append(cut_qubit)
        
        _, original_out_node = find_io_node(cut_dag, wire)
        cut_in_node, cut_out_node = find_io_node(cut_dag, cut_qubit)
        
        cut_dag._multi_graph.add_edge(nodes_before_cut[len(nodes_before_cut)-1], original_out_node,
        name="%s[%s]" % (wire[0].name, wire[1]), wire=wire)
        cut_dag._multi_graph.add_edge(cut_in_node, nodes_after_cut[0],
        name="%s[%s]" % (cut_qubit[0].name, cut_qubit[1]), wire=cut_qubit)
        cut_dag._multi_graph.add_edge(nodes_after_cut[len(nodes_after_cut)-1], cut_out_node,
        name="%s[%s]" % (cut_qubit[0].name, cut_qubit[1]), wire=cut_qubit)
        
        edge_key = find_edge_key(cut_dag._multi_graph, nodes_after_cut[len(nodes_after_cut)-1], original_out_node, wire)
        cut_dag._multi_graph.remove_edge(nodes_after_cut[len(nodes_after_cut)-1], original_out_node, key=edge_key)

        edge_key = find_edge_key(cut_dag._multi_graph, nodes_before_cut[len(nodes_before_cut)-1], nodes_after_cut[0], wire)
        cut_dag._multi_graph.remove_edge(nodes_before_cut[len(nodes_before_cut)-1], nodes_after_cut[0], key=edge_key)

        edge_key = find_edge_key(cut_dag._multi_graph, cut_in_node, cut_out_node, wire)
        cut_dag._multi_graph.remove_edge(cut_in_node, cut_out_node, key=edge_key)

        for idx, node in enumerate(nodes_after_cut):
            updated_qargs = []
            for qarg in node.qargs:
                if qarg == wire:
                    updated_qargs.append(cut_qubit)
                else:
                    updated_qargs.append(qarg)
            node.qargs = updated_qargs
            if idx<len(nodes_after_cut)-1:
                edge_key = find_edge_key(cut_dag._multi_graph, nodes_after_cut[idx], nodes_after_cut[idx+1], wire)
                cut_dag._multi_graph.remove_edge(nodes_after_cut[idx], nodes_after_cut[idx+1], key=edge_key)
                cut_dag._multi_graph.add_edge(nodes_after_cut[idx], nodes_after_cut[idx+1],
                name="%s[%s]" % (cut_qubit[0].name, cut_qubit[1]), wire=cut_qubit)
    for input_qubit in path_map:
        path_map[input_qubit].append(input_qubit)
        path_map[input_qubit] = path_map[input_qubit][::-1]
    
    components = list(nx.weakly_connected_components(cut_dag._multi_graph))
    num_components = len(components)
    if num_components<2:
        raise Exception('Not a split, cut_dag only has %d component' % num_components)
    return cut_dag, path_map

def find_uncut_in_node(wire, path_map):
    for in_node in path_map:
        if wire in path_map[in_node]:
            return in_node
    return None

def clusters_generator(cut_dag, path_map):
    components = list(nx.weakly_connected_components(cut_dag._multi_graph))
    cluster_circs = []
    cluster_qubits = []
    for component_idx, component in enumerate(components):
        component_qubits = []
        relative_positions = []
        for node in component:
            if node.type == 'in':
                in_node = find_uncut_in_node(node.wire, path_map)
                relative_position = list(path_map.keys()).index(in_node)
                component_qubits.append(node.wire)
                relative_positions.append(relative_position)
                # print(node.wire,'relative position in component:',relative_position)
        relative_positions, component_qubits = (list(t) for t in zip(*sorted(zip(relative_positions, component_qubits))))
        cluster_qubits.append(component_qubits)
        component_qreg = QuantumRegister(len(component_qubits),'q')
        cluster_circ = QuantumCircuit(component_qreg)
        for node in cut_dag.topological_op_nodes():
            if node in component:
                translated_qargs = []
                for qarg in node.qargs:
                    translated_qarg = component_qreg[component_qubits.index(qarg)]
                    translated_qargs.append(translated_qarg)
                cluster_circ.append(instruction=node.op, qargs=translated_qargs)
        cluster_circs.append(cluster_circ)

    return cluster_circs, cluster_qubits

def complete_path_map_generator(path_map, cluster_qubits):
    complete_path_map = copy.deepcopy(path_map)
    for qubit in complete_path_map:
        for path_qubit_idx, path_qubit in enumerate(complete_path_map[qubit]):
            for cluster_idx, cluster in enumerate(cluster_qubits):
                if path_qubit in cluster:
                    sub_circ_idx = cluster_idx
                    cluster_qubit_idx = cluster.index(path_qubit)
                    complete_path_map[qubit][path_qubit_idx] = (sub_circ_idx, cluster_qubit_idx)
                    break
    return complete_path_map

def cut_circuit(circ, positions):
    original_dag = circuit_to_dag(circ)
    cut_dag, path_map = cut_edges(original_dag=original_dag, positions=positions)
    cluster_circs, cluster_qubits = clusters_generator(cut_dag, path_map)
    complete_path_map = complete_path_map_generator(path_map, cluster_qubits)
    K = len(positions)
    d = []
    for x in cluster_qubits:
        # d = max(d, len(x))
        d.append(len(x))

    return cluster_circs, complete_path_map, K, d
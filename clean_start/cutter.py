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

def find_io_node(dag, wire):
    in_node = None
    out_node = None

    for node in dag.nodes():
        if node.type == 'in' and node.name == '%s[%d]' % (wire[0].name,wire[1]):
            in_node = node
        if node.type == 'out' and node.name == '%s[%d]' % (wire[0].name,wire[1]):
            out_node = node
    if in_node == None or out_node == None:
        raise Exception('did not find {}, in_node = {}, out_node = {}'.format(wire, in_node, out_node))
    return in_node, out_node
            
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
    complete_path_map = {}
    for input_qubit in original_dag.qubits():
        complete_path_map[input_qubit] = []

    for cutQ_idx, position in enumerate(positions):
        wire, source_node_idx = position
        
        nodes_before_cut = list(cut_dag.nodes_on_wire(wire=wire, only_ops=True))[:source_node_idx+1]
        nodes_after_cut = list(cut_dag.nodes_on_wire(wire=wire, only_ops=True))[source_node_idx+1:]
        cut_qubit = cutQ_register[cutQ_idx]
        complete_path_map[wire].append(cut_qubit)
        
        _, original_out_node = find_io_node(cut_dag, wire)
        cut_in_node, cut_out_node = find_io_node(cut_dag, cut_qubit)
        
        cut_dag._multi_graph.add_edge(nodes_before_cut[len(nodes_before_cut)-1], original_out_node,
        name="%s[%s]" % (wire[0].name, wire[1]), wire=wire)
        cut_dag._multi_graph.add_edge(cut_in_node, nodes_after_cut[0],
        name="%s[%s]" % (cut_qubit[0].name, cut_qubit[1]), wire=cut_qubit)
        cut_dag._multi_graph.add_edge(nodes_after_cut[len(nodes_after_cut)-1], cut_out_node,
        name="%s[%s]" % (cut_qubit[0].name, cut_qubit[1]), wire=cut_qubit)
        cut_dag._multi_graph.remove_edge(nodes_after_cut[len(nodes_after_cut)-1], original_out_node)
        cut_dag._multi_graph.remove_edge(nodes_before_cut[len(nodes_before_cut)-1], nodes_after_cut[0])
        cut_dag._multi_graph.remove_edge(cut_in_node, cut_out_node)

        for idx, node in enumerate(nodes_after_cut):
            updated_qargs = []
            for qarg in node.qargs:
                if qarg == wire:
                    updated_qargs.append(cut_qubit)
                else:
                    updated_qargs.append(qarg)
            node.qargs = updated_qargs
            if idx<len(nodes_after_cut)-1:
                cut_dag._multi_graph.remove_edge(nodes_after_cut[idx], nodes_after_cut[idx+1])
                cut_dag._multi_graph.add_edge(nodes_after_cut[idx], nodes_after_cut[idx+1],
                name="%s[%s]" % (cut_qubit[0].name, cut_qubit[1]), wire=cut_qubit)
    for input_qubit in complete_path_map:
        complete_path_map[input_qubit].append(input_qubit)
        complete_path_map[input_qubit] = complete_path_map[input_qubit][::-1]
    
    components = list(nx.weakly_connected_components(cut_dag._multi_graph))
    num_components = len(components)
    if num_components<2:
        raise Exception('Not a split, cut_dag only has %d component' % num_components)
    return cut_dag, complete_path_map

def fragments_generator(cut_dag):
    components = list(nx.weakly_connected_components(cut_dag._multi_graph))
    for component_idx, component in enumerate(components):
        print('component %d' % component_idx)
        component_qregs = {}
        component_qubits = {}
        for node in component:
            if node.type == 'in':
                # print(node.type, node.name, node.wire[0].name)
                if node.wire[0].name in component_qregs:
                    old_register = component_qregs[node.wire[0].name]
                    new_register = QuantumRegister(old_register.size+1, old_register.name)
                    component_qregs[node.wire[0].name] = new_register
                    component_qubits[node.wire] = (new_register.name, new_register.size-1)
                    # print('has {}, changed from {} to {}' % (node.wire[0].name,old_register,new_register))
                else:
                    new_register = QuantumRegister(1, node.wire[0].name)
                    component_qregs[node.wire[0].name] = new_register
                    component_qubits[node.wire] = (new_register.name, new_register.size-1)
                    # print('does not have {}, add {}' % (node.wire[0].name,old_register,new_register))
        for qubit in component_qubits:
            qubit_register = component_qregs[component_qubits[qubit][0]]
            qubit_register_idx = component_qubits[qubit][1]
            component_qubits[qubit] = qubit_register[qubit_register_idx]
        print('registers:', component_qregs)
        print('qubits:', component_qubits)
        fragment = QuantumCircuit()
        for qreg in component_qregs.values():
            fragment.add_register(qreg)
        print('*'*100)
    return
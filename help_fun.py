import numpy as np
from qiskit import QuantumCircuit
from qiskit.tools.visualization import dag_drawer
import random
from qiskit.dagcircuit.exceptions import DAGCircuitError
from qiskit.dagcircuit.dagnode import DAGNode
from qiskit.circuit import Measure
from qiskit.circuit.quantumregister import QuantumRegister
from qiskit.circuit.classicalregister import ClassicalRegister
import networkx as nx
import copy

# 2 dimensional q registers
class q_register():
    def __init__(self, row, col):
        self.row = row
        self.col = col
        self.qubits = QuantumRegister(row * col, 'q')
        prev_gates = []
        for i in range(row):
            row = []
            for j in range(col):
                row.append('')
            prev_gates.append(row)
        self.prev_gates = prev_gates
        
    def __getitem__(self, pos):
        return self.qubits[pos[0] * self.col + pos[1]]

def remove_elements(l, ele):
    ret = []
    for element in l:
        if element != ele:
            ret.append(element)
    return ret

def last_oneq_gate(l):
    for element in reversed(l):
        if element != 'CZ':
            return element
    return 'none'

def full_entangle(circuit, q_reg):
	for row in range(q_reg.row):
		for col in range(q_reg.col):
			circuit.h(q_reg[[row, col]])
			q_reg.prev_gates[row][col] += 'H '
	# circuit.barrier()
	return circuit

def one_q_layer(circuit, q_reg):
    oneq_dict = {'T':['sqrt_X', 'sqrt_Y'], 'sqrt_X':['T','sqrt_Y'], 'sqrt_Y':['T','sqrt_X']}
    for row in range(q_reg.row):
        for col in range(q_reg.col):
            prev_gates = q_reg.prev_gates[row][col].split(' ')
            del prev_gates[len(prev_gates)-1]
            curr_gate = prev_gates[len(prev_gates)-1]
            if curr_gate == 'none':
                prev_gates = remove_elements(prev_gates, 'none')
                # print('qubit %d, %d, previous gates:' % (row, col), prev_gates, 'current gate: ', curr_gate)
                if prev_gates[len(prev_gates)-1] == 'CZ' and 'T' not in prev_gates:
                    circuit.t(q_reg[[row, col]])
                    # print('add T')
                    q_reg.prev_gates[row][col] += 'T '
                elif prev_gates[len(prev_gates)-1] == 'CZ' and 'T' in prev_gates:
                    curr_gate = random.choice(oneq_dict[last_oneq_gate(prev_gates)])
                    if curr_gate == 'T':
                        circuit.t(q_reg[[row, col]])
                        # print('add T')
                        q_reg.prev_gates[row][col] += 'T '
                    elif curr_gate == 'sqrt_X':
                        circuit.rx(np.pi/2, q_reg[[row, col]])
                        # print('add X')
                        q_reg.prev_gates[row][col] += 'sqrt_X '
                    elif curr_gate == 'sqrt_Y':
                        circuit.ry(np.pi/2, q_reg[[row, col]])
                        # print('add Y')
                        q_reg.prev_gates[row][col] += 'sqrt_Y '
    return circuit

def update_gates_history(q_reg, row, col, curr_gate, qubits_touched):
    q_reg.prev_gates[row][col] += curr_gate
    q_reg.prev_gates[row][col] += ' '
    qubits_touched.append([row,col])
    return q_reg, qubits_touched

def supremacy_layer(circuit, q_reg, rotation_idx, single_qubit_gates):
    qubits_touched = []
    rotation_idx = rotation_idx % 8
    if rotation_idx == 0 or rotation_idx == 1:
        for row in range(0, q_reg.row, 2):
            for col in range(0, q_reg.col, 4):
                if col + rotation_idx + 1 < q_reg.col:
                    circuit.cz(q_reg[[row, col + rotation_idx]], q_reg[[row, col + rotation_idx + 1]])
                    q_reg, qubits_touched = update_gates_history(q_reg, row, col+rotation_idx, 'CZ', qubits_touched)
                    q_reg, qubits_touched = update_gates_history(q_reg, row, col+rotation_idx+1, 'CZ', qubits_touched)
                if col + rotation_idx + 2 + 1 < q_reg.col and row+1<q_reg.row:
                    circuit.cz(q_reg[[row+1, col + rotation_idx + 2]], q_reg[[row+1, col + rotation_idx + 3]])
                    q_reg, qubits_touched = update_gates_history(q_reg, row+1, col+rotation_idx+2, 'CZ', qubits_touched)
                    q_reg, qubits_touched = update_gates_history(q_reg, row+1, col+rotation_idx+3, 'CZ', qubits_touched)
    elif rotation_idx == 2 or rotation_idx == 3:
        for row in range(0, q_reg.row, 2):
            for col in range(0, q_reg.col, 4):
                if col + rotation_idx - 2 + 1 < q_reg.col and row+1<q_reg.row:
                    circuit.cz(q_reg[[row+1, col + rotation_idx - 2]], q_reg[[row+1, col + rotation_idx - 1]])
                    q_reg, qubits_touched = update_gates_history(q_reg, row+1, col+rotation_idx-2, 'CZ', qubits_touched)
                    q_reg, qubits_touched = update_gates_history(q_reg, row+1, col+rotation_idx-1, 'CZ', qubits_touched)
                if col + rotation_idx + 1 < q_reg.col:
                    circuit.cz(q_reg[[row, col + rotation_idx]], q_reg[[row, col + rotation_idx + 1]])
                    q_reg, qubits_touched = update_gates_history(q_reg, row, col+rotation_idx, 'CZ', qubits_touched)
                    q_reg, qubits_touched = update_gates_history(q_reg, row, col+rotation_idx+1, 'CZ', qubits_touched)
    elif rotation_idx == 4 or rotation_idx == 5:
        for col in range(0, q_reg.col, 2):
            for row in range(0, q_reg.row, 4):
                if row + rotation_idx - 4 + 1 < q_reg.row:
                    circuit.cz(q_reg[[row + rotation_idx - 4, col]], q_reg[[row + rotation_idx - 3, col]])
                    q_reg, qubits_touched = update_gates_history(q_reg, row+rotation_idx-4, col, 'CZ', qubits_touched)
                    q_reg, qubits_touched = update_gates_history(q_reg, row+rotation_idx-3, col, 'CZ', qubits_touched)
                if row + rotation_idx - 4 + 2 + 1 < q_reg.row and col+1<q_reg.col:
                    circuit.cz(q_reg[[row + rotation_idx - 2, col+1]], q_reg[[row + rotation_idx - 1, col+1]])
                    q_reg, qubits_touched = update_gates_history(q_reg, row+rotation_idx-2, col+1, 'CZ', qubits_touched)
                    q_reg, qubits_touched = update_gates_history(q_reg, row+rotation_idx-1, col+1, 'CZ', qubits_touched)
    elif rotation_idx == 6 or rotation_idx == 7:
        for col in range(0, q_reg.col, 2):
            for row in range(0, q_reg.row, 4):
                if row + rotation_idx - 4 - 2 + 1 < q_reg.row and col+1<q_reg.col:
                    circuit.cz(q_reg[[row + rotation_idx - 6, col+1]], q_reg[[row + rotation_idx - 5, col+1]])
                    q_reg, qubits_touched = update_gates_history(q_reg, row+rotation_idx-6, col+1, 'CZ', qubits_touched)
                    q_reg, qubits_touched = update_gates_history(q_reg, row+rotation_idx-5, col+1, 'CZ', qubits_touched)
                if row + rotation_idx - 4 + 1 < q_reg.row:
                    circuit.cz(q_reg[[row + rotation_idx - 4, col]], q_reg[[row + rotation_idx - 3, col]])
                    q_reg, qubits_touched = update_gates_history(q_reg, row+rotation_idx-4, col, 'CZ', qubits_touched)
                    q_reg, qubits_touched = update_gates_history(q_reg, row+rotation_idx-3, col, 'CZ', qubits_touched)
    for row in range(q_reg.row):
        for col in range(q_reg.col):
            if [row,col] not in qubits_touched:
                q_reg.prev_gates[row][col] += 'none '
    if single_qubit_gates:
        # print('Adding single qubit gates, starting idx = ', rotation_idx)
        # print('All previous gates history:\n', q_reg.prev_gates)
        circuit = one_q_layer(circuit = circuit, q_reg = q_reg)
    # circuit.barrier()
    return circuit

def cut_single_edge(original_dag, wire, source_node_idx):
    """Cut a single edge in the original_dag.

    Args:
        wire (Qubit): wire to cut in original_dag
        source_node_idx (int): #op nodes in front of the edge to cut

    Returns:
        DAGCircuit: dag circuit after cutting

    Raises:
        DAGCircuitError: if a leaf node is connected to multiple outputs

    """

    cut_dag = copy.deepcopy(original_dag)

    cut_dag._check_bits([wire], cut_dag.output_map)

    original_out_node = cut_dag.output_map[wire]
    ie = list(cut_dag._multi_graph.predecessors(original_out_node))
    if len(ie) != 1:
        raise DAGCircuitError("output node has multiple in-edges")

    source_node = list(cut_dag.nodes_on_wire(wire=wire, only_ops=True))[source_node_idx]
    dest_node = list(cut_dag.nodes_on_wire(wire=wire, only_ops=True))[source_node_idx+1]

    cut_dag._multi_graph.remove_edge(source_node, dest_node)

    return cut_dag

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
    for position in positions:
        wire, source_node_idx = position
        cut_dag = cut_single_edge(cut_dag, wire, source_node_idx)
    num_components = nx.number_weakly_connected_components(cut_dag._multi_graph)
    if num_components<2:
        raise Exception('Not a split, cut_dag only has %d component' % num_components)
    return cut_dag

def io_node_is_cut(io_node, wires_being_cut):
    ''' Test if io_node is among the wires being cut
    
    Args:
        wires_being_cut (list): list of qubit tuples

    Returns:
        (-1, None) if io_node is not being cut or is not a i/o node
        (index of the wire being cut, i/o type of io_node) if io_node is being cut
    '''

    # TODO: only i/o nodes will have the same name as wires being cut. Is this necessary?
    if io_node.type != 'in' and io_node.type != 'out':
        return (-1, None)
    wires_being_cut_names = ['%s[%s]' % (x[0].name, x[1]) for x in wires_being_cut]
    for idx, name in enumerate(wires_being_cut_names):
        if io_node.name == name:
            return (idx, io_node.type)
    return (-1, None)

def qarg_being_cut(qarg, wires_being_cut):
    wires_being_cut_names = ['%s[%s]' % (x[0].name, x[1]) for x in wires_being_cut]
    qarg_name = '%s[%s]' % (qarg[0].name, qarg[1])
    for idx, name in enumerate(wires_being_cut_names):
        if qarg_name == name:
            return idx
    return -1

def reg_dict_counter(cut_dag, wires_being_cut):
    '''Count #registers required in each component.

    Args:
        cut_dag (DAGCircuit): dag circuit after being cut
        wires_being_cut (list): list of wires being cut in (QuantumRegister, idx) tuples

    Returns:
        list: list of dict object for # and type of registers in each component.
        key: name of register. value: QuantumRegister or ClassicalRegister

        dict: a dictionary for input wires from the original dag into each component
        key: DAGNode of input wires in original circuit. value: (sub circuit index, bit tuple in the sub circuit)

    '''
    num_components = nx.number_weakly_connected_components(cut_dag._multi_graph)
    components = list(nx.weakly_connected_components(cut_dag._multi_graph))
    sub_reg_dicts = []
    input_wires_mapping = {}
    for i in range(num_components):
        reg_dict = {}
        component = components[i]

        ''' Count reg_dict for the sub_circ '''
        # TODO: is there a neater way to traverse the component nodes?
        for node in cut_dag.topological_nodes():
            if node in component and node.type == 'in':
            # Input nodes in the component
                if node.wire[0].name not in reg_dict and type(node.wire[0]) == QuantumRegister:
                    reg_dict[node.wire[0].name] = QuantumRegister(1, node.wire[0].name)
                    input_wires_mapping[node] =  (i, reg_dict[node.wire[0].name][reg_dict[node.wire[0].name].size-1])
                elif node.wire[0].name in reg_dict and type(node.wire[0]) == QuantumRegister:
                    reg_dict[node.wire[0].name] = QuantumRegister(reg_dict[node.wire[0].name].size+1,node.wire[0].name)
                    input_wires_mapping[node] =  (i, reg_dict[node.wire[0].name][reg_dict[node.wire[0].name].size-1])
                
                # TODO: do we want to have ClassicalRegister in cut_dag at all?
                elif node.wire[0].name not in reg_dict and type(node.wire[0]) == ClassicalRegister:
                    reg_dict[node.wire[0].name] = ClassicalRegister(1, node.wire[0].name)
                    input_wires_mapping[node] =  (i, reg_dict[node.wire[0].name][reg_dict[node.wire[0].name].size-1])
                elif node.wire[0].name in reg_dict and type(node.wire[0]) == ClassicalRegister:
                    reg_dict[node.wire[0].name] = ClassicalRegister(reg_dict[node.wire[0].name].size+1,node.wire[0].name)
                    input_wires_mapping[node] =  (i, reg_dict[node.wire[0].name][reg_dict[node.wire[0].name].size-1])
            
            if node in component and io_node_is_cut(node, wires_being_cut)[1] == 'in' :
                ''' Contains input node of wire being cut, need to measure '''
                measure_register_name = 'measure_' + node.wire[0].name
                if measure_register_name not in reg_dict:
                    reg_dict[measure_register_name] = ClassicalRegister(1,measure_register_name)
                else:
                    reg_dict[measure_register_name] = ClassicalRegister(reg_dict[measure_register_name].size+1, measure_register_name)
            if node in component and io_node_is_cut(node, wires_being_cut)[1] == 'out' :
                ''' Contains output node of wire being cut, need to add ancilla '''
                ancilla_register_name = 'ancilla_' + node.wire[0].name
                if ancilla_register_name not in reg_dict:
                    reg_dict[ancilla_register_name] = QuantumRegister(1, ancilla_register_name)
                else:
                    reg_dict[ancilla_register_name] = QuantumRegister(reg_dict[ancilla_register_name].size+1, ancilla_register_name)

        sub_reg_dicts.append(reg_dict)
    for key in input_wires_mapping:
        sub_circ_idx = input_wires_mapping[key][0]
        sub_circ_reg_name = input_wires_mapping[key][1][0].name
        sub_circ_reg_index = input_wires_mapping[key][1][1]
        sub_circ_reg = sub_reg_dicts[sub_circ_idx][sub_circ_reg_name][sub_circ_reg_index]
        input_wires_mapping[key] = (sub_circ_idx, sub_circ_reg)
    return sub_reg_dicts, input_wires_mapping

def contains_cut_wires_io_nodes(cut_dag, wires_being_cut):
    num_components = nx.number_weakly_connected_components(cut_dag._multi_graph)
    components = list(nx.weakly_connected_components(cut_dag._multi_graph))
    contains_cut_wires_in_nodes = [[False for wire in wires_being_cut] for i in range(num_components)]
    contains_cut_wires_out_nodes = [[False for wire in wires_being_cut] for i in range(num_components)]

    for i in range(num_components):
        component = components[i]
        for node in cut_dag.topological_nodes():
            if node in component and io_node_is_cut(node, wires_being_cut)[1] == 'in':
                contains_cut_wires_in_nodes[i][io_node_is_cut(node, wires_being_cut)[0]] = True
            if node in component and io_node_is_cut(node, wires_being_cut)[1] =='out':
                contains_cut_wires_out_nodes[i][io_node_is_cut(node, wires_being_cut)[0]] = True
    return contains_cut_wires_in_nodes, contains_cut_wires_out_nodes

def total_circ_regs_counter(sub_reg_dicts):
    total_circ_regs = {}
    # Update total_circ_regs
    for reg_dict in sub_reg_dicts:
        for key in reg_dict:
            if key in total_circ_regs:
                if type(total_circ_regs[key]) == QuantumRegister:
                    total_circ_regs[key] = QuantumRegister(total_circ_regs[key].size + reg_dict[key].size, key)
                elif type(total_circ_regs[key]) == ClassicalRegister:
                    total_circ_regs[key] = ClassicalRegister(total_circ_regs[key].size + reg_dict[key].size, key)
            else:
                total_circ_regs[key] = reg_dict[key]
    return total_circ_regs

def bridges_calc(bridge_maps, input_wires_mapping):
    total_bridge_map = {}
    for idx, bridge_map in enumerate(bridge_maps):
        for key in bridge_map:
            measure = input_wires_mapping[key]
            ancilla = (idx, '%s[%s]' % (bridge_map[key][0].name, bridge_map[key][1]))
            total_bridge_map[measure] = ancilla
    return total_bridge_map

def generate_sub_circs(cut_dag, positions):
    wires_being_cut = [x[0] for x in positions]
    sub_circs = []
    sub_reg_dicts, input_wires_mapping = reg_dict_counter(cut_dag, wires_being_cut)
    contains_cut_wires_in_nodes, contains_cut_wires_out_nodes = contains_cut_wires_io_nodes(cut_dag, wires_being_cut)
    # print('input wires mapping: ', input_wires_mapping)
    # print('sub_reg_dicts calculation:', sub_reg_dicts)
    components = list(nx.weakly_connected_components(cut_dag._multi_graph))

    bridge_maps = []

    for component_idx, reg_dict in enumerate(sub_reg_dicts):
        print('Begin component ', component_idx)
        sub_circ = QuantumCircuit()
        bridge_map = {}
        # total_circ_regs = total_circ_regs_counter(sub_reg_dicts[:component_idx])
        print('reg_dict: ', reg_dict)
        # print('cumulative registers counts:', total_circ_regs)
        
        ''' Add the registers '''
        for reg in reg_dict.values():
            sub_circ.add_register(reg)
        print('added registers in the sub circuit:', sub_circ.qregs, sub_circ.cregs)
        
        # Update qargs of nodes
        cutQ_idx = 0
        for node in cut_dag.topological_op_nodes():
            if node in components[component_idx]:
                new_args = []
                old_args = node.qargs
                # print('changing args for ', node.name, old_args, end = ' ')
                for x in old_args:
                    if qarg_being_cut(x, wires_being_cut) != -1:
                        has_out = contains_cut_wires_out_nodes[component_idx][qarg_being_cut(x, wires_being_cut)]
                        has_in = contains_cut_wires_in_nodes[component_idx][qarg_being_cut(x, wires_being_cut)]
                        if not has_in:
                            key = '%s[%s]' % (x[0].name, x[1])
                            if key in bridge_map:
                                new_args.append(bridge_map[key])
                            else:
                                new_args.append(reg_dict['cutQ'][cutQ_idx])
                                bridge_map[key] = reg_dict['cutQ'][cutQ_idx]
                                cutQ_idx += 1
                        if not has_out:
                            new_args.append(reg_dict[x[0].name][input_wires_mapping['%s[%s]' % (x[0].name, x[1])][1]])
                    else:
                        new_args.append(reg_dict[x[0].name][input_wires_mapping['%s[%s]' % (x[0].name, x[1])][1]])
                node.qargs = new_args
                # print('to ', new_args)
                sub_circ.append(instruction=node.op, qargs=node.qargs, cargs=node.cargs)
        cutC_idx = 0
        for idx, has_in in enumerate(contains_cut_wires_in_nodes[component_idx]):
            if has_in:
                wire = wires_being_cut[idx]
                meas_reg = reg_dict[wire[0].name]
                meas_index = input_wires_mapping['%s[%s]' % (wire[0].name, wire[1])][1]
                # sub_circ.barrier(meas_reg[meas_index])
                sub_circ.append(instruction=Measure(), qargs=[meas_reg[meas_index]],cargs=[reg_dict['cutC'][cutC_idx]])
                cutC_idx += 1
    
        print('finished component %d\n' % component_idx)
        sub_circs.append(sub_circ)
        bridge_maps.append(bridge_map)
    bridge_map = bridges_calc(bridge_maps, input_wires_mapping)
    return sub_circs, bridge_map
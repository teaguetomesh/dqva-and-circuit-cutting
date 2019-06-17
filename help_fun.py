import numpy as np
from qiskit import *
from qiskit.visualization import *
import random
from qiskit.dagcircuit.exceptions import DAGCircuitError
from qiskit.dagcircuit.dagnode import DAGNode
from qiskit.circuit import Measure
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

def cut_edge(original_dag, wire, source_node, dest_node):
        """Cut a single edge in the original_dag.

        Args:
            wire (Qubit): wire to cut in original_dag
            source_node (DAGNode): start node of the edge to cut
            dest_node (DAGNode): end node of the edge to cut

        Returns:
            DAGCircuit: dag circuit after cutting

        Raises:
            DAGCircuitError: if a leaf node is connected to multiple outputs

        """

        cut_dag = original_dag

        original_dag._check_bits([wire], original_dag.output_map)

        original_out_node = original_dag.output_map[wire]
        ie = list(original_dag._multi_graph.predecessors(original_out_node))
        if len(ie) != 1:
            raise DAGCircuitError("output node has multiple in-edges")
        
        """Insert a measure op for wire
        After source_node and before dest_node
        """
        cut_c = ClassicalRegister(1, 'cutC')
        cut_dag._add_op_node(op=Measure(), qargs=[wire], cargs=[cut_c[0]])
        meas_node = cut_dag._id_to_node[cut_dag._max_node_id]
        cut_dag.add_creg(cut_c)
        c_reg_in_node = cut_dag._id_to_node[cut_dag._max_node_id-1]
        c_reg_out_node = cut_dag._id_to_node[cut_dag._max_node_id]
        
        """Insert ancilla qubit"""
        cut_q = QuantumRegister(1, 'cutQ')
        cut_dag.add_qreg(cut_q)
        q_reg_in_node = cut_dag._id_to_node[cut_dag._max_node_id-1]
        q_reg_out_node = cut_dag._id_to_node[cut_dag._max_node_id]

        # # print('adding edge from %s(source_node) to %s(meas_node)' % (source_node.name, meas_node.name))
        cut_dag._multi_graph.add_edge(source_node, meas_node,
        name="%s[%s]" % (wire.register.name, wire.index), wire=wire)

        # # print('removing edge from %s(source_node) to %s(dest_node)' % (source_node.name, dest_node.name))
        cut_dag._multi_graph.remove_edge(source_node, dest_node)

        # # print('adding edge from %s to %s' % (c_reg_in_node.name, meas_node.name))
        # dag._multi_graph.add_edge(c_reg_in_node, meas_node,
        # name="%s[%s]" % (cut_c.name, cut_c[0].index), wire=cut_c)

        # # print('adding edge from %s to %s' % (meas_node.name, c_reg_out_node.name))
        # dag._multi_graph.add_edge(meas_node, c_reg_out_node,
        # name="%s[%s]" % (cut_c.name, cut_c[0].index), wire=cut_c)

        # # print('adding edge from %s to %s' % (meas_node.name, original_out_node.name))
        # dag._multi_graph.add_edge(meas_node, original_out_node,
        # name="%s[%s]" % (qarg.register.name, qarg.index), wire=qarg)

        # # print('removing edge from %s to %s' % (ie[0].name, original_out_node.name))
        # dag._multi_graph.remove_edge(ie[0], original_out_node)

        # # print('removing edge from %s to %s' % (c_reg_in_node.name, c_reg_out_node.name))
        # dag._multi_graph.remove_edge(c_reg_in_node, c_reg_out_node)

        # # print('adding edge from %s to %s' % (q_reg_in_node.name, dest_node.name))
        # dag._multi_graph.add_edge(q_reg_in_node, dest_node,
        # name="%s[%s]" % (cut_q.name, cut_q[0].index), wire=cut_q)

        # # print('adding edge from %s to %s' % (ie[0].name, q_reg_out_node.name))
        # dag._multi_graph.add_edge(ie[0], q_reg_out_node,
        # name="%s[%s]" % (cut_q.name, cut_q[0].index), wire=cut_q)

        # # print('removing edge from %s to %s' % (q_reg_in_node.name, q_reg_out_node.name))
        # dag._multi_graph.remove_edge(q_reg_in_node, q_reg_out_node)

        # dag.update_edges(parent_node=q_reg_in_node, original_node=qarg, new_register=cut_q)

        return cut_dag
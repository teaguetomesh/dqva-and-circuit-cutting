import numpy as np
from qiskit import *
from qiskit.visualization import *
import random

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
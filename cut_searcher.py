from qiskit import QuantumCircuit
from qiskit.dagcircuit.dagcircuit import DAGCircuit
from qiskit.converters import circuit_to_dag, dag_to_circuit
import random
import math
import copy
from collections import Counter
from datetime import datetime
import sys

class Graph(object):
    def __init__(self, vlist):
        self.verts = {v[0]:Counter(v[1:]) for v in vlist}
        self.update_edges()

    def update_edges(self):
        self.edges = []
        
        for k, v in self.verts.items():
            self.edges += ([(k, t) for t in v.keys() for n in range(v[t]) if k < t])

    @property
    def vertex_count(self):
        return len(self.verts)

    @property
    def edge_count(self):
        return len(self.edges)

    def merge_vertices(self, edge_index):
        head_key, tail_key = self.edges[edge_index]
        # print('removing edge ', self.edges[edge_index], head_key, tail_key)
        
        head = self.verts[head_key]
        tail = self.verts[tail_key]

        # print('merging head =', head_key, head, 'tail =', tail_key, tail)

        # Remove the edge between head and tail
        del head[tail_key]
        del tail[head_key]

        # print('After removing head->tail, head =', head, 'tail =', tail)

        # Merge tails
        head.update(tail)

        # print('After merging tail, head =', head_key, head)

        # Update all the neighboring vertices of the fused vertex
        for i in tail.keys():
            v = self.verts[i]
            # print('tail neighboring vertex', i, 'vertex', v)
            v[head_key] += v[tail_key]
            del v[tail_key]
            # print('updated to', v)
            
        # Finally remove the tail vertex
        del self.verts[tail_key]

        self.update_edges()
        # print('FINISHED edges:', self.edges)
        # print('FINISHED verts:', self.verts)

def translate_edge_index(original_r, original_g, curr_g, grouping):
    # print('translating edge index', original_r)
    original_head, original_tail = original_g.edges[original_r]
    # print('original head =', original_head, 'original tail =', original_tail)
    # print('current graph:', curr_g.verts)
    curr_head = None
    if original_head in curr_g.verts:
        curr_head = original_head
    else:
        for group in grouping:
            if original_head in group.split(' '):
                curr_head = group.split(' ')[0]
    curr_tail = None
    if original_tail in curr_g.verts:
        curr_tail = original_tail
    else:
        for group in grouping:
            if original_tail in group.split(' '):
                curr_tail = group.split(' ')[0]
    curr_r = -1
    for edge_idx, edge in enumerate(curr_g.edges):
        if (curr_head, curr_tail) == edge:
            curr_r = edge_idx
    return curr_r

def contract(graph, min_v=2):
    g = copy.deepcopy(graph)
    grouping = [x for x in g.verts]
    contracted_edges = []
    contraction_order = random.sample(range(0,graph.edge_count), graph.edge_count)
    i = 0
    # print('initial grouping:', grouping)
    while g.vertex_count > min_v:
        # r = random.randrange(0, g.edge_count)
        # head, tail = g.edges[r]
        # graph_r = random.randrange(0, graph.edge_count)
        graph_r = contraction_order[i]
        i+=1
        contracted_edges.append(graph.edges[graph_r])
        g_r = translate_edge_index(original_r=graph_r, original_g=graph, curr_g=g, grouping=grouping)
        g_head, g_tail = g.edges[g_r]
        # print('contracting', graph.edges[graph_r], 'in the original graph')
        # print('keep the edge', (g_head, g_tail), 'in the contracted graph')
        g.merge_vertices(g_r)
        
        hi = -1
        ti = -1
        for group_idx, group in enumerate(grouping):
            if g_head in group:
                hi = group_idx
            if g_tail in group:
                ti = group_idx
        grouping.append(grouping[hi] + ' ' + grouping[ti])
        del grouping[min(hi,ti)]
        del grouping[max(hi,ti)-1]
        # print('updated grouping:', grouping)
        # print('*'*100)

    cut_edges = []
    for edge in graph.edges:
        head, tail = edge
        contracted = False
        for group in grouping:
            if head in group.split(' ') and tail in group.split(' '):
                contracted = True
                break
        if not contracted:
            cut_edges.append(edge)

    return g, grouping, cut_edges

def find_crevices(l):
    rolling_idx = l[0]
    d = 1
    for idx, ele in enumerate(l):
        if ele == rolling_idx:
            rolling_idx += 1
        else:
            d += 1
            rolling_idx = ele + 1
    return d

def cluster_character(grouping):
    # TODO: change to fragment hardness metric
    max_d = 0
    for group in grouping:
        group_qubits = {}
        for vertex in group.split(' '):
            qargs = vertex.split(',')
            for qarg in qargs:
                qubit = qarg.split(']')[0] + ']'
                multi_Qgate_idx = int(qarg.split(']')[1])
                if qubit not in group_qubits:
                    group_qubits[qubit] = [multi_Qgate_idx]
                else:
                    group_qubits[qubit].append(multi_Qgate_idx)
        # print(group_qubits)
        d = 0
        for qubit in group_qubits:
            l = sorted(group_qubits[qubit])
            d += find_crevices(l)
        max_d = max(max_d, d)
    return max_d

# Karger's Algorithm
# For failure probabilty upper bound of 1/n, repeat the algorithm nC2 logn times
def min_cut(graph, min_v=2):
    m = graph.edge_count
    n = graph.vertex_count
    all_K_d = {}
    print('%d edges %d vertices graph will run %d times' % (m, n, int(n * (n-1) * math.log(n)/2)))
    for i in range(int(n * (n-1) * math.log(n)/2)):
        random.seed(datetime.now())
        g, grouping, cut_edges = contract(graph, min_v)
        m = min(m, g.edge_count)
        K = g.edge_count
        d = cluster_character(grouping)
        # TODO: same K,d cluster is overwritten
        all_K_d[(K,d)] = cut_edges
        # print('Run %d cut_edges:' % i, cut_edges)
        # print('*'* 100)
    
    pareto_K_d = {}
    for key0 in all_K_d:
        K0, d0 = key0
        is_pareto = True
        for key1 in all_K_d:
            K1, d1 = key1
            if key1!=key0 and ((K1<K0 and d1<d0) or (K1==K0 and d1<d0) or (K1<K0 and d1==d0)):
                is_pareto = False
                break
        if is_pareto:
            pareto_K_d[key0] = all_K_d[key0]

    return pareto_K_d

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
    graph = Graph(abstraction) 
    return graph

def pareto_K_d_parser(pareto_K_d, circ):
    dag = circuit_to_dag(circ)
    for pareto_solution in pareto_K_d:
        cuts = pareto_K_d[pareto_solution]
        positions = []
        # print('cuts:', cuts)
        for position in cuts:
            source, dest = position
            source_qargs = [x[:len(x)-1] for x in source.split(',')]
            dest_qargs = [x[:len(x)-1] for x in dest.split(',')]
            qubit_cut = list(set(source_qargs).intersection(set(dest_qargs)))
            if len(qubit_cut)>1:
                raise Exception('one cut is cutting on multiple qubits')
            for x in source.split(','):
                if x[:len(x)-1] == qubit_cut[0]:
                    source_idx = int(x[len(x)-1])
            for x in dest.split(','):
                if x[:len(x)-1] == qubit_cut[0]:
                    dest_idx = int(x[len(x)-1])
            multi_Q_gate_idx = max(source_idx, dest_idx)
            # print('cut qubit:', qubit_cut[0], 'after %d multi qubit gate'% multi_Q_gate_idx)
            wire = None
            for qubit in circ.qubits:
                if qubit[0].name == qubit_cut[0].split('[')[0] and qubit[1] == int(qubit_cut[0].split('[')[1].split(']')[0]):
                    wire = qubit
            tmp = 0
            all_Q_gate_idx = None
            for gate_idx, gate in enumerate(list(dag.nodes_on_wire(wire=wire, only_ops=True))):
                if len(gate.qargs)>1:
                    tmp += 1
                    if tmp == multi_Q_gate_idx:
                        all_Q_gate_idx = gate_idx
            positions.append((wire, all_Q_gate_idx))
        pareto_K_d[pareto_solution] = (sorted(positions, reverse=True, key=lambda cut: cut[1]), pareto_K_d[pareto_solution][1])
    return pareto_K_d

def find_pareto_solutions(circ, num_clusters=2):
    stripped_circ = circ_stripping(circ)
    graph = circuit_to_graph(stripped_circ)
    pareto_K_d = min_cut(graph, num_clusters)
    pareto_K_d = pareto_K_d_parser(pareto_K_d, circ)
            
    return pareto_K_d
from qiskit import QuantumCircuit
from qiskit.dagcircuit.dagcircuit import DAGCircuit
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.tools.visualization import dag_drawer
import random
import math
import copy
from collections import Counter
from datetime import datetime
import sys
import numpy as np
from qcg.generators import gen_supremacy

class Graph(object):
    def __init__(self, vlist):
        self.verts = {}
        all_vertices = []
        for l in vlist:
            [all_vertices.append(x) for x in l]
            v = l[0]
            vertex = Counter(l[1:])
            self.verts[v] = vertex
        all_vertices = set(all_vertices)
        for v in all_vertices:
            if v not in self.verts:
                self.verts[v] = Counter([])
        self.update_edges()

    def update_edges(self):
        self.edges = []
        
        for src in self.verts:
            dest_l = list(self.verts[src].elements())
            self.edges += [(src,x) for x in dest_l]

    @property
    def vertex_count(self):
        return len(self.verts)

    @property
    def edge_count(self):
        return len(self.edges)

    def merge_vertices(self, edge_index):
        head_key, tail_key = self.edges[edge_index]
        
        head = self.verts[head_key]
        tail = self.verts[tail_key]

        # Remove the edge from head to tail
        del head[tail_key]

        # Merge tail into head
        head.update(tail)

        # Vertices pointing to tail should now point to head
        for vertex in self.verts:
            if tail_key in self.verts[vertex] and vertex != head_key:
                if head_key in self.verts[vertex]:
                    self.verts[vertex][head_key] += self.verts[vertex][tail_key]
                else:
                    self.verts[vertex][head_key] = self.verts[vertex][tail_key]
                del self.verts[vertex][tail_key]
            
        # Finally remove the tail vertex
        del self.verts[tail_key]

        self.update_edges()

def circ_stripping(circ):
    # Remove all single qubit gates in the circuit
    dag = circuit_to_dag(circ)
    stripped_dag = DAGCircuit()
    [stripped_dag.add_qreg(x) for x in circ.qregs]
    for vertex in dag.topological_op_nodes():
        if len(vertex.qargs) >= 2:
            stripped_dag.apply_operation_back(op=vertex.op, qargs=vertex.qargs)
    return dag_to_circuit(stripped_dag)

def circuit_to_graph(circ):
    dag = circuit_to_dag(circ)
    input_qubit_gate_counter = {}
    for qubit in circ.qubits:
        input_qubit_gate_counter[qubit] = 0
    
    id_name = {}
    for vertex in dag.topological_op_nodes():
        vertex_name = ''
        for qarg in vertex.qargs:
            gate_count = input_qubit_gate_counter[qarg]
            vertex_name += '%s[%d]%d ' % (qarg[0].name, qarg[1], gate_count)
            input_qubit_gate_counter[qarg] += 1
        vertex_name = vertex_name[:len(vertex_name)-1]
        id_name[id(vertex)] = vertex_name
    
    graph_edges = {}
    for edge in dag.edges():
        source, dest, attr = edge
        if source.type == 'op' and dest.type == 'op':
            source_vertex_name = id_name[id(source)]
            dest_vertex_name = id_name[id(dest)]
            # print(source_vertex_name, 'to', dest_vertex_name)
            if source_vertex_name in graph_edges:
                graph_edges[source_vertex_name].append(dest_vertex_name)
            else:
                graph_edges[source_vertex_name] = [dest_vertex_name]
    
    # [print(x, graph_edges[x]) for x in graph_edges]

    graph_init = []
    for source_vertex in graph_edges:
        graph_init_ele = [source_vertex]
        for dest_vertex in graph_edges[source_vertex]:
            graph_init_ele.append(dest_vertex)
        graph_init.append(graph_init_ele)
    # [print(x) for x in graph_init]

    graph = Graph(graph_init)
    return graph
    
if __name__ == '__main__':
    circ = gen_supremacy(3,4,8,'71230456')
    stripped_circ = circ_stripping(circ)
    graph = circuit_to_graph(stripped_circ)
    # [print(x, graph.verts[x]) for x in graph.verts]
    positions, hardness, K, d = min_cut(graph, 2, 24)
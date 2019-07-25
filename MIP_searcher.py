from qiskit.dagcircuit.dagcircuit import DAGCircuit
from qiskit.converters import circuit_to_dag, dag_to_circuit
from gurobipy import Model, GRB, quicksum, LinExpr
import networkx as nx
from qcg.generators import gen_supremacy
import randomized_searcher as r_s
import csv

class Basic_Model(object):
    def create_graph(self):
        G = nx.Graph()
        G.add_nodes_from(range(self.n_vertices))

        for v1, v2 in self.edges:
            G.add_edge(v1, v2) 
        self.graph = G
        self.node_sets = set()
        self.node_set_vars = dict()

    def __init__(self, n_vertices, edges, constraints, k, verbosity=0):
        self.check_graph(n_vertices, edges)
        self.n_vertices = n_vertices
        self.edges = edges
        self.k = k
        self.verbosity = verbosity
        self.create_graph()

        self.model = Model('cut_searching')
        self.model.params.updatemode = 1

        self.mvars = []
        # Indicates if a vertex is in cluster k
        for i in range(k):
            cvars = []
            for j in range(n_vertices):
                v = self.model.addVar(lb=0.0, ub=1.0, vtype=GRB.BINARY)
                cvars.append(v)
            self.mvars.append(cvars)

        # constraint: each vertex in exactly/at least one cluster
        for v in range(n_vertices):
            self.model.addConstr(quicksum([self.mvars[i][v] for i in range(k)]), 
                                          GRB.EQUAL, 1)

        # symmetry-breaking constraints
        self.model.addConstr(self.mvars[0][0], GRB.EQUAL, 1)
        for i in range(2, k):
            self.model.addConstr(quicksum([self.mvars[i-1][j] for j in range(n_vertices)]),
                            GRB.LESS_EQUAL,
                            quicksum([self.mvars[i][j] for j in range(n_vertices)]))
    
    def check_graph(self, n_vertices, edges):
        # 1. edges must include all vertices
        # 2. all u,v must be ordered and smaller than n_vertices
        vertices = set([i for (i, _) in edges])
        vertices |= set([i for (_, i) in edges])
        assert(vertices == set(range(n_vertices)))
        for u, v in edges:
            assert(u < v)
            assert(u < n_vertices)

def read_graph(graph_fname, cons_fname=None):
    node_names = dict()
    node_ids = dict()
    edges = set()
    
    def get_id(name):
        # node_id is ordered 0 to n_vertices
        if not name in node_names:
            node_id = len(node_names)
            node_names[name] = node_id
            node_ids[node_id] = name 

        return node_names[name]

    def add_edge(v1, v2):
        first = min(v1, v2)
        second = max(v1, v2)
        edges.add((first, second))


    with open(graph_fname) as f:
        f.readline()
        for line in f:
            line = line.strip()
            if not line: continue
            [n1, n2] = line.strip().split(',')
            [v1, v2] = [get_id(i) for i in (n1, n2)]
            add_edge(v1, v2)

    n_vertices = max(node_ids.keys()) + 1
    
    constraints = []
    if cons_fname is not None:
        with open(cons_fname) as f:
            f.readline()
            for line in f:
                line = line.strip()
                if not line: continue
                [n1, n2, w] = line.split(',')
                [v1, v2] = [get_id(i) for i in (n1, n2)]
                w = float(w)
                constraints.append((v1, v2, w))
    
    return n_vertices, list(edges), constraints, node_ids

if __name__ == '__main__':
    circ = gen_supremacy(3,3,8)
    stripped_circ = r_s.circ_stripping(circ)
    graph = r_s.circuit_to_graph(stripped_circ)
    with open('graph.csv', 'w') as writeFile:
        for u,v in graph.edges:
            writer = csv.writer(writeFile)
            u = u.replace(',',' ')
            v = v.replace(',',' ')
            writer.writerow([u,v])
    writeFile.close()
    n_vertices, edges, constraints, node_ids = read_graph('graph.csv')
    kwargs = dict(n_vertices=n_vertices, 
                  edges=edges,
                  constraints=constraints,
                  k=2)
    [print(x, kwargs[x]) for x in kwargs]
    print('*'*100)

    m = Basic_Model(**kwargs)
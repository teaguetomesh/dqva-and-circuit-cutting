from qiskit.dagcircuit.dagcircuit import DAGCircuit
from qiskit.converters import circuit_to_dag, dag_to_circuit
from gurobipy import *
import networkx as nx
from qcg.generators import gen_supremacy
import randomized_searcher as r_s
import csv
import numpy as np

class Basic_Model(object):
    def create_graph(self):
        G = nx.Graph()
        G.add_nodes_from(range(self.n_vertices))

        for v1, v2 in self.edges:
            G.add_edge(v1, v2) 
        self.graph = G
        self.node_sets = set()
        self.node_set_vars = dict()

    def check_graph(self, n_vertices, edges):
        # 1. edges must include all vertices
        # 2. all u,v must be ordered and smaller than n_vertices
        vertices = set([i for (i, _) in edges])
        vertices |= set([i for (_, i) in edges])
        assert(vertices == set(range(n_vertices)))
        for u, v in edges:
            assert(u < v)
            assert(u < n_vertices)
    
    def connectivity_vars(self, cluster, v1, v2):
        assert((v1, v2) not in self.edges)

        connectivity_vars = []
        for path in nx.all_simple_paths(self.graph, v1, v2):
            node_set = tuple(sorted(path[1:-1]))
            n = len(node_set)
            
            if n == 1:
                node = node_set[0]
                cvar = self.node_vars[cluster][node]
            else:            
                # check if the node set is new
                if not node_set in self.node_sets:
                    n = len(node_set)
                    for i in range(self.k):
                        var = self.model.addVar(lb=0.0, ub=1.0, vtype=GRB.BINARY)
                        ns_vars = [self.node_vars[i][j] for j in node_set]
                        self.node_set_vars[(node_set, i)] = var
                        self.model.addConstr(quicksum(ns_vars) - n*var <= n-1)
                        self.model.addConstr(quicksum(ns_vars) - n*var >= 0)

                    self.node_sets.add(node_set)
                cvar = self.node_set_vars[(node_set, cluster)]

            connectivity_vars.append(cvar)
        
        return connectivity_vars
    
    def __init__(self, n_vertices, edges, constraints, node_ids, k, hw_max_qubit, verbosity=0):
        print('*'*200)
        print('Initializing MIP model')
        self.check_graph(n_vertices, edges)
        self.n_vertices = n_vertices
        self.edges = edges
        self.n_edges = len(edges)
        self.node_ids = node_ids
        self.k = k
        self.hw_max_qubit = hw_max_qubit
        self.verbosity = verbosity
        self.create_graph()

        self.model = Model('cut_searching')
        self.model.params.updatemode = 1

        self.node_vars = []
        self.not_node_vars = []
        # Indicates if a vertex is in cluster k
        for i in range(k):
            cluster_vars = []
            not_cluster_vars = []
            for j in range(n_vertices):
                v = self.model.addVar(lb=0.0, ub=1.0, vtype=GRB.BINARY)
                not_v = self.model.addVar(lb=0.0, ub=1.0, vtype=GRB.BINARY)
                self.model.addConstr(not_v == 1-v)
                cluster_vars.append(v)
                not_cluster_vars.append(not_v)
            self.node_vars.append(cluster_vars)
            self.not_node_vars.append(not_cluster_vars)

        self.edge_vars = []
        # Indicates if an edge is in cluster k
        for i in range(k):
            cluster_vars = []
            for j in range(self.n_edges):
                v = self.model.addVar(lb=0.0, ub=1.0, vtype=GRB.BINARY)
                cluster_vars.append(v)
            self.edge_vars.append(cluster_vars)
        
        # constraint: each vertex in exactly/at least one cluster
        for v in range(n_vertices):
            self.model.addConstr(quicksum([self.node_vars[i][v] for i in range(k)]), GRB.EQUAL, 1)

        # constraint: edge_var=1 indicates one and only one vertex of an edge is in cluster k
        for i in range(k):
            for e in range(self.n_edges):
                u, v = self.edges[e]
                u_node_var = self.node_vars[i][u]
                v_node_var = self.node_vars[i][v]
                not_u_node_var = self.not_node_vars[i][u]
                not_v_node_var = self.not_node_vars[i][v]
                tmp1 = self.model.addVar(lb=0.0, ub=1.0, vtype=GRB.BINARY)
                self.model.addConstr(tmp1 == and_(u_node_var, not_v_node_var))
                tmp2 = self.model.addVar(lb=0.0, ub=1.0, vtype=GRB.BINARY)
                self.model.addConstr(tmp2 == and_(not_u_node_var, v_node_var))
                self.model.addConstr(self.edge_vars[i][e] == or_(tmp1, tmp2))

        # connectivity constraints:
        print('adding connectivity constraints')
        for v1 in range(n_vertices):
            for v2 in range(v1+1, n_vertices):
                if (v1, v2) in self.edges: continue
                for i in range(k):
                    cvars = self.connectivity_vars(i, v1, v2)
                    self.model.addConstr(self.node_vars[i][v1] + self.node_vars[i][v2], 
                                         GRB.LESS_EQUAL,
                                         quicksum(cvars) + 1)

        # symmetry-breaking constraints
        print('adding symmetry-breaking constraints')
        self.model.addConstr(self.node_vars[0][0], GRB.EQUAL, 1)
        for i in range(2, k):
            self.model.addConstr(quicksum([self.node_vars[i-1][j] for j in range(n_vertices)]),
                            GRB.LESS_EQUAL,
                            quicksum([self.node_vars[i][j] for j in range(n_vertices)]))
        
        # Objective function
        obj_expr = QuadExpr()
        for cluster in range(k):
            # TODO: figure out how to compute cluster_hardness
            cluster_K = self.model.addVar(lb=0.0, ub=100.0, vtype=GRB.INTEGER)
            self.model.addConstr(cluster_K == 
            quicksum([self.edge_vars[cluster][i] for i in range(self.n_edges)]))
            
            cluster_original_qubit = self.model.addVar(lb=0.0, ub=100.0, vtype=GRB.INTEGER)
            self.model.addConstr(cluster_original_qubit ==
            quicksum([self.node_vars[cluster][i] for i in range(self.n_vertices)]))
            
            cluster_cut_qubit = self.model.addVar(lb=0.0, ub=100.0, vtype=GRB.INTEGER)
            self.model.addConstr(cluster_cut_qubit ==
            quicksum([self.edge_vars[cluster][i] * self.node_vars[cluster][self.edges[i][1]]
            for i in range(self.n_edges)]))

            cluster_d = self.model.addVar(lb=0.0, ub=100.0, vtype=GRB.INTEGER)
            self.model.addConstr(cluster_d == cluster_original_qubit + cluster_cut_qubit)

            obj_expr.add(cluster_K)
            obj_expr.add(cluster_d*cluster_d)
        
        self.model.setObjective(obj_expr, GRB.MINIMIZE)
        self.model.update()
        self.model.params.OutputFlag = self.verbosity

    def solve(self):     
        try:
            self.model.optimize()
        except GurobiError:
            print(GurobiError.message)
        
        self.objective = None
        self.clusters = None
        self.optimal = (self.model.Status == GRB.OPTIMAL)
        self.runtime = self.model.Runtime
        self.node_count = self.model.nodecount
        self.mip_gap = self.model.mipgap
        self.objective = self.model.ObjVal
        
        if self.model.solcount > 0:
            clusters = []
            for i in range(self.k):
                cluster = []
                for j in range(self.n_vertices):
                    if abs(self.node_vars[i][j].x) > 1e-4:
                        cluster.append(j)
                clusters.append(cluster)
            self.clusters = clusters
    
    def print_stat(self):
        print('*'*200)
        print('MIP stats:')
        print('clusters:')
        print(self.clusters)

        print('node count:', self.node_count)
        print('mip gap:', self.mip_gap)
        print('objective value:', self.objective)
        print('runtime:', self.runtime)
        for i in range(self.k):
            print('cluster %d has edges:' % i)
            for j in range(self.n_edges):
                if abs(self.edge_vars[i][j].x) < 1e-4:
                    print(self.edges[j])


        if (self.optimal):
            print('OPTIMAL')
        else:
            print('NOT OPTIMAL')

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
    circ = gen_supremacy(3,3,8,'71230456')
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
                  node_ids=node_ids,
                  k=2,
                  hw_max_qubit=24)
    print('kwargs:')
    [print(x, kwargs[x]) for x in kwargs]

    m = Basic_Model(**kwargs)
    m.solve()
    m.print_stat()
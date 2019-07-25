# coding: utf-8

from __future__ import print_function, division
from gurobipy import Model, GRB, quicksum, LinExpr
import networkx as nx

class Basic_Model(object):
    def create_graph(self):
        G = nx.Graph()
        G.add_nodes_from(range(self.n_vertices))

        for v1, v2 in self.edges:
            G.add_edge(v1, v2) 
        self.graph = G
        self.node_sets = set()
        self.node_set_vars = dict()
        
    def connectivity_vars(self, cluster, v1, v2):
        assert((v1, v2) not in self.edges)

        connectivity_vars = []
        for path in nx.all_simple_paths(self.graph, v1, v2):
            node_set = tuple(sorted(path[1:-1]))
            n = len(node_set)
            
            if n == 1:
                node = node_set[0]
                cvar = self.mvars[cluster][node]
            else:            
                # check if the node set is new
                if not node_set in self.node_sets:
                    n = len(node_set)
                    for i in range(self.k):
                        var = self.model.addVar(lb=0.0, ub=1.0, vtype=GRB.BINARY)
                        ns_vars = [self.mvars[i][j] for j in node_set]
                        self.node_set_vars[(node_set, i)] = var
                        self.model.addConstr(quicksum(ns_vars) - n*var <= n-1)
                        self.model.addConstr(quicksum(ns_vars) - n*var >= 0)

                    self.node_sets.add(node_set)
                cvar = self.node_set_vars[(node_set, cluster)]

            connectivity_vars.append(cvar)
        
        return connectivity_vars        
        
    def __init__(self, n_vertices, edges, constraints, k, gamma, 
                 verbosity=0, 
                 symmetry_breaking=True,
                 overlap=False,
                 timeout=None):
        self.check_graph(n_vertices, edges)
        self.n_vertices = n_vertices
        self.edges = edges
        self.k = k
        self.verbosity = verbosity
        self.timeout = timeout
        self.create_graph()
        
        self.model = Model('graph_clustering')
        self.model.params.updatemode = 1
        
        self.mvars = []
        for i in range(k):
            cvars = []
            for j in range(n_vertices):
                v = self.model.addVar(lb=0.0, ub=1.0, vtype=GRB.BINARY)
                cvars.append(v)
            self.mvars.append(cvars)
 
        ineq_sense = GRB.GREATER_EQUAL if overlap else GRB.EQUAL
        # constraint: each vertex in exactly/at least one cluster
        for v in range(n_vertices):
            self.model.addConstr(quicksum([self.mvars[i][v] for i in range(k)]), 
                                          ineq_sense, 1)
            
        # connectivity constraints:
        for v1 in range(n_vertices):
            for v2 in range(v1+1, n_vertices):
                if (v1, v2) in self.edges: continue
                for i in range(k):
                    cvars = self.connectivity_vars(i, v1, v2)
                    self.model.addConstr(self.mvars[i][v1] + self.mvars[i][v2], 
                                         GRB.LESS_EQUAL,
                                         quicksum(cvars) + 1)
                    
        # symmetry-breaking constraints
        if symmetry_breaking:
            self.model.addConstr(self.mvars[0][0], GRB.EQUAL, 1)
            for i in range(2, k):
                self.model.addConstr(quicksum([self.mvars[i-1][j] for j in range(n_vertices)]),
                                GRB.LESS_EQUAL,
                                quicksum([self.mvars[i][j] for j in range(n_vertices)]))
        
        
        obj_expr = LinExpr()
        wsum = sum(w for (_, _, w) in constraints)
        gamma = gamma/wsum        
        # indicators for violation of cl constraints
        for (u, v, w) in constraints:
            for i in range(k):
                y = self.model.addVar(lb=0.0, ub=1.0, vtype=GRB.BINARY)
                self.model.addConstr(y >= self.mvars[i][u] + self.mvars[i][v] - 1)
                obj_expr.add(y, -w * gamma)
        
        # size of smallest cluster 
        s = self.model.addVar(lb=0.0, ub=n_vertices, vtype=GRB.INTEGER)
        for i in range(k):
            self.model.addConstr(s <= quicksum([self.mvars[i][v] for v in range(n_vertices)]))
        
        s_coef = 1/n_vertices if overlap else k/n_vertices
        obj_expr.add(s_coef * s)
        
        self.model.setObjective(obj_expr, GRB.MAXIMIZE)
        self.model.update()
        self.model.params.OutputFlag = self.verbosity
               
    def check_graph(self, n_vertices, edges):
        vertices = set([i for (i, _) in edges])
        vertices |= set([i for (_, i) in edges])
        assert(vertices == set(range(n_vertices)))
        for u, v in edges:
            assert(u < v)
            assert(u < n_vertices)
    
    def solve(self):
        if self.timeout:
            self.model.Params.TimeLimit = self.timeout        
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
                    if abs(self.mvars[i][j].x) > 1e-4:
                        cluster.append(j)
                clusters.append(cluster)
            self.clusters = clusters        
        
    def print_stat(self):
        pass



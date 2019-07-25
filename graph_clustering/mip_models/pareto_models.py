# coding: utf-8

from __future__ import print_function, division
from gurobipy import Model, GRB, quicksum, LinExpr, GurobiError
import networkx as nx
from networkx.algorithms.flow import edmonds_karp
from timeit import default_timer as timer

class Cut_Finder(object):
    def __init__(self, ingraph_nnodes, ingraph_edges):
        G = nx.DiGraph()
        G.add_nodes_from(range(2*ingraph_nnodes))

        for i in range(ingraph_nnodes):
            G.add_edge(2*i, 2*i+1)
            G.add_edge(2*i+1, 2*i)

        for v1, v2 in ingraph_edges:
            G.add_edge(v1*2+1, v2*2) 
            G.add_edge(v2*2+1, v1*2)
        
        self.G = G
        self.in_nodes = ingraph_nnodes
        self.in_edges = set(ingraph_edges)
        self.capacity_edges =  [(2*i, 2*i+1) for i in range(ingraph_nnodes)]
        self.capacity_edges += [(2*i+1, 2*i) for i in range(ingraph_nnodes)]
        
    def update_capacities(self, ingraph_capacities):
        edge_capacities = dict(zip(self.capacity_edges, ingraph_capacities * 2))
        nx.set_edge_attributes(self.G, 'capacity', edge_capacities)        
    
    def find_cutset(self, in1, in2):
        try:
            _, (reachable, non_reachable) = nx.minimum_cut(self.G, 
                                                           in1*2+1, in2*2,
                                                           flow_func=edmonds_karp)
        except nx.NetworkXUnbounded:
            print('unbounded flow for nodes %d and %d'%(in1, in2))
        cutset = set()
        for u, nbrs in ((n, self.G[n]) for n in reachable):
            cutset.update((u, v) for v in nbrs if v in non_reachable)
        return [i//2 for (i, _) in cutset]
    
    def get_cutsets(self, ingraph_capacities):
        self.update_capacities(ingraph_capacities)
        
        cutsets = []
        for in1 in range(self.in_nodes):
            for in2 in range(in1+1, self.in_nodes):
                if (in1, in2) in self.in_edges: continue
                if ingraph_capacities[in1] + ingraph_capacities[in2] <= 1: continue
                    
                cutset = self.find_cutset(in1, in2)
                cut_csum = sum(ingraph_capacities[i] for i in cutset)
                if cut_csum < ingraph_capacities[in1] + ingraph_capacities[in2] -1:
                    cutsets.append((in1, in2, cutset))
        return cutsets


def mincut_callback(model, where):

    if model._impcounter < 10 and where == GRB.Callback.MIPNODE:
        if model.cbGet(GRB.Callback.MIPNODE_NODCNT) != 0: return
            
        start = timer()
        
        relaxation_objval = model.cbGet(GRB.Callback.MIPNODE_OBJBND)
        if model._relobj is not None:
            if model._relobj == 0:
                if relaxation_objval == 0:
                    model._impcounter += 1
                else:
                    model._impcounter = 0
            else:
                imp = (model._relobj - relaxation_objval) / model._relobj
                if imp < 0.005: 
                    model._impcounter += 1
                else:
                    model._impcounter = 0
        model._relobj = relaxation_objval

        for i in range(model._k):
            capacities = model.cbGetNodeRel(model._vars[i])
            cutsets = model._cutfinder.get_cutsets(capacities)
            for (u, v, cutset) in cutsets:
                cutset_expr = quicksum(model._vars[i][j] for j in cutset)
                model.cbCut(cutset_expr >= model._vars[i][u] + model._vars[i][v] - 1)
            if model._single_cut and cutsets:
                break
        model._root_cuttime += timer() - start
        
    elif where == GRB.Callback.MIPSOL:
            
        start = timer()
        
        for i in range(model._k):
            capacities = model.cbGetSolution(model._vars[i])
            cutsets = model._cutfinder.get_cutsets(capacities)
            for (u, v, cutset) in cutsets:
                cutset_expr = quicksum(model._vars[i][j] for j in cutset)
                model.cbLazy(cutset_expr >= model._vars[i][u] + model._vars[i][v] - 1)
        model._tree_cuttime += timer() - start


class Bnc_Pareto(object):
    def __init__(self, n_vertices, edges, constraints, 
                 k, minsize,
                 verbosity=0, 
                 symmetry_breaking=True,
                 overlap=False,
                 single_cut=False,
                 timeout=None):
        self.check_graph(n_vertices, edges)
        self.n_vertices = n_vertices
        self.k = k
        self.verbosity = verbosity
        self.timeout = timeout
        
        model = Model('graph_clustering')
        
        mvars = []
        for i in range(k):
            cvars = []
            for j in range(n_vertices):
                v = model.addVar(lb=0.0, ub=1.0, vtype=GRB.BINARY)
                cvars.append(v)
            mvars.append(cvars)
        model.update()
            
        ineq_sense = GRB.GREATER_EQUAL if overlap else GRB.EQUAL
        # constraint: each vertex in exactly/at least one cluster
        for v in range(n_vertices):
            model.addConstr(quicksum([mvars[i][v] for i in range(k)]), 
                                     ineq_sense, 1)
                                         
        # symmetry-breaking constraints
        if symmetry_breaking:
            model.addConstr(mvars[0][0], GRB.EQUAL, 1)
            for i in range(2, k):
                model.addConstr(quicksum([mvars[i-1][j] for j in range(n_vertices)]) <=
                                quicksum([mvars[i][j] for j in range(n_vertices)]))

        # size constraint        
        for i in range(k):
            model.addConstr(
            quicksum([mvars[i][v] for v in range(n_vertices)]) >= minsize)

        
        obj_expr = LinExpr()
        # indicators for violation of cl constraints
        for (u, v, w) in constraints:
            for i in range(k):
                y = model.addVar(lb=0.0, ub=1.0, vtype=GRB.BINARY)
                model.update()
                model.addConstr(y >= mvars[i][u] + mvars[i][v] - 1)
                obj_expr.add(y, w)
        
        model.setObjective(obj_expr, GRB.MINIMIZE)
        model.params.OutputFlag = self.verbosity
        model.Params.PreCrush = 1
        model.Params.LazyConstraints = 1
        
        model._cutfinder = Cut_Finder(n_vertices, edges)
        model._vars = mvars
        model._k = k
        model._relobj = None
        model._impcounter = 0
        model._single_cut = single_cut
        
        # runtime information
        model._root_cuttime = 0
        model._tree_cuttime = 0
        
        self.model = model
               
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
            self.model.optimize(mincut_callback)
        except GurobiError:
            print(GurobiError.message)
        
        self.objective = None
        self.clusters = None
        self.optimal = (self.model.Status == GRB.OPTIMAL)
        self.runtime = self.model.Runtime
        self.node_count = self.model.nodecount
        self.mip_gap = self.model.mipgap
        
        if self.model.solcount > 0:
            self.objective = self.model.ObjVal
            clusters = []
            for i in range(self.k):
                cluster = []
                for j in range(self.n_vertices):
                    if abs(self.model._vars[i][j].x) > 1e-4:
                        cluster.append(j)
                clusters.append(cluster)
            self.clusters = clusters
        
    def print_stat(self):

        print('separation time in root: %f' %self.model._root_cuttime)
        print('separation time in tree: %f' %self.model._tree_cuttime)




class Basic_Pareto(object):
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
        
    def __init__(self, n_vertices, edges, constraints, 
                 k, minsize,
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
        self.model.params.OutputFlag = self.verbosity
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
        # size constraint        
        for i in range(k):
            self.model.addConstr(
            quicksum([self.mvars[i][v] for v in range(n_vertices)]) >= minsize)
        
        obj_expr = LinExpr()
        # indicators for violation of cl constraints
        for (u, v, w) in constraints:
            for i in range(k):
                y = self.model.addVar(lb=0.0, ub=1.0, vtype=GRB.BINARY)
                self.model.addConstr(y >= self.mvars[i][u] + self.mvars[i][v] - 1)
                obj_expr.add(y, w)
        
        self.model.setObjective(obj_expr, GRB.MINIMIZE)
        self.model.update()
               
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
        
        if self.model.solcount > 0:
            self.objective = self.model.ObjVal
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





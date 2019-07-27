from qiskit.dagcircuit.dagcircuit import DAGCircuit
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.tools.visualization import dag_drawer
from gurobipy import *
import networkx as nx
from qcg.generators import gen_supremacy
import randomized_searcher as r_s

class Basic_Model(object):
    def __init__(self, n_vertices, edges, node_ids, id_nodes, k, hw_max_qubit, verbosity=0):
        print('*'*200)
        print('Initializing MIP model')
        self.check_graph(n_vertices, edges)
        self.n_vertices = n_vertices
        self.edges = edges
        self.n_edges = len(edges)
        self.node_ids = node_ids
        self.id_nodes = id_nodes
        self.k = k
        self.hw_max_qubit = hw_max_qubit
        self.verbosity = verbosity
        self.create_graph()

        self.model = Model('cut_searching')
        self.model.params.updatemode = 1

        self.node_qubits = {}
        for node in self.node_ids:
            qargs = node.split(' ')
            num_in_qubits = 0
            for qarg in qargs:
                if int(qarg.split(']')[1]) == 0:
                    num_in_qubits += 1
            self.node_qubits[node] = num_in_qubits

        # Indicate if a node is in some cluster
        self.node_vars = []
        self.not_node_vars = []
        for i in range(k):
            cluster_vars = []
            not_cluster_vars = []
            for j in range(n_vertices):
                j_in_i = self.model.addVar(lb=0.0, ub=1.0, vtype=GRB.BINARY)
                j_not_in_i = self.model.addVar(lb=0.0, ub=1.0, vtype=GRB.BINARY)
                self.model.addConstr(j_not_in_i == 1-j_in_i)
                cluster_vars.append(j_in_i)
                not_cluster_vars.append(j_not_in_i)
            self.node_vars.append(cluster_vars)
            self.not_node_vars.append(not_cluster_vars)

        # Indicate if an edge has one and only one vertex in some cluster
        self.edge_vars = []
        for i in range(k):
            cluster_vars = []
            for j in range(self.n_edges):
                v = self.model.addVar(lb=0.0, ub=1.0, vtype=GRB.BINARY)
                cluster_vars.append(v)
            self.edge_vars.append(cluster_vars)
        
        # constraint: each vertex in exactly one cluster
        print('adding vertex non-overlapping constraint')
        for v in range(n_vertices):
            self.model.addConstr(quicksum([self.node_vars[i][v] for i in range(k)]), GRB.EQUAL, 1)
        
        # constraint: edge_var=1 indicates one and only one vertex of an edge is in cluster
        # edge_var[cluster][edge] = node_var[cluster][u] XOR node_var[cluster][v]
        print('adding cutting edges constraint')
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
        # TODO: check the implementations
        # TODO: add BnC method?
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
        # TODO: check the implementation
        print('adding symmetry-breaking constraints')
        self.model.addConstr(self.node_vars[0][0], GRB.EQUAL, 1)
        for i in range(2, k):
            self.model.addConstr(quicksum([self.node_vars[i-1][j] for j in range(n_vertices)]),
                            GRB.LESS_EQUAL,
                            quicksum([self.node_vars[i][j] for j in range(n_vertices)]))
        
        # Objective function
        print('adding objective')
        obj_expr = QuadExpr()
        for cluster in range(k):
            # TODO: figure out how to compute cluster_hardness
            # FIXME: upper bound should not be hardcoded
            cluster_K = self.model.addVar(lb=0.0, ub=100.0, vtype=GRB.INTEGER)
            self.model.addConstr(cluster_K == 
            quicksum([self.edge_vars[cluster][i] for i in range(self.n_edges)]))
            
            cluster_original_qubit = self.model.addVar(lb=0.0, ub=100.0, vtype=GRB.INTEGER)
            self.model.addConstr(cluster_original_qubit ==
            quicksum([self.node_qubits[id_nodes[i]]*self.node_vars[cluster][i]
            for i in range(self.n_vertices)]))
            
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
    
    def check_graph(self, n_vertices, edges):
        # 1. edges must include all vertices
        # 2. all u,v must be ordered and smaller than n_vertices
        vertices = set([i for (i, _) in edges])
        vertices |= set([i for (_, i) in edges])
        assert(vertices == set(range(n_vertices)))
        for u, v in edges:
            assert(u < v)
            assert(u < n_vertices)
    
    def create_graph(self):
        # Need the graph for connectivity checks
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
    
    def solve(self):
        print('*'*200)
        print('solving model')   
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
        print('MIQCP stats:')
        print('clusters:')
        print(self.clusters)

        print('node count:', self.node_count)
        print('mip gap:', self.mip_gap)
        print('objective value:', self.objective)
        print('runtime:', self.runtime)

        if (self.optimal):
            print('OPTIMAL')
        else:
            print('NOT OPTIMAL')

def read_circ(circ):
    dag = circuit_to_dag(circ)
    edges = []
    node_name_ids = {}
    id_node_names = {}
    node_ids = {}
    curr_node_id = 0
    qubit_gate_idx = {}
    for qubit in dag.qubits():
        qubit_gate_idx[qubit] = 0
    # print('initial qubit_gate_idx:', qubit_gate_idx)
    for vertex in dag.topological_op_nodes():
        if len(vertex.qargs) != 2:
            raise Exception('vertex does not have 2 qargs!')
        # print('qargs:', vertex.qargs)
        arg0, arg1 = vertex.qargs
        vertex_name = '%s[%d]%d %s[%d]%d' % (arg0[0].name, arg0[1],qubit_gate_idx[arg0],
                                             arg1[0].name, arg1[1],qubit_gate_idx[arg1])
        qubit_gate_idx[arg0] += 1
        qubit_gate_idx[arg1] += 1
        # print('vertex_name:', vertex_name)
        if vertex_name not in node_name_ids and id(vertex) not in node_ids:
            node_name_ids[vertex_name] = curr_node_id
            id_node_names[curr_node_id] = vertex_name
            node_ids[id(vertex)] = curr_node_id
            curr_node_id += 1
    
    for u, v, attr in dag.edges():
        if u.type == 'op' and v.type == 'op':
            u_id = node_ids[id(u)]
            v_id = node_ids[id(v)]
            edges.append((u_id, v_id))
         
    n_vertices = len(list(dag.topological_op_nodes()))
    return n_vertices, edges, node_name_ids, id_node_names

if __name__ == '__main__':
    circ = gen_supremacy(4,4,8,'71230456')
    stripped_circ = r_s.circ_stripping(circ)
    dag_drawer(circuit_to_dag(stripped_circ),filename='dag.pdf')
    n_vertices, edges, node_ids, id_nodes = read_circ(stripped_circ)
    kwargs = dict(n_vertices=n_vertices,
                  edges=edges,
                  node_ids=node_ids,
                  id_nodes=id_nodes,
                  k=2,
                  hw_max_qubit=24)
    print('*'*200)
    print('kwargs:')
    [print(x, kwargs[x],'\n') for x in kwargs]

    m = Basic_Model(**kwargs)
    m.solve()
    m.print_stat()
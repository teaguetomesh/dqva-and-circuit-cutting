from qiskit.dagcircuit.dagcircuit import DAGCircuit
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.tools.visualization import dag_drawer
from gurobipy import *
import networkx as nx
from qcg.generators import gen_supremacy
import randomized_searcher as r_s
import numpy as np
import cutter

class Basic_Model(object):
    def __init__(self, n_vertices, edges, node_ids, id_nodes, k, hw_max_qubit, verbose=False):
        if verbose:
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
        self.verbosity = 0

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

        # Indicate if a cluster contains an edge
        # self.has_edge = []
        # for i in range(k):
        #     cluster_vars = []
        #     for j in range(self.n_edges):
        #         v = self.model.addVar(lb=0.0, ub=1.0, vtype=GRB.BINARY)
        #         cluster_vars.append(v)
        #     self.has_edge.append(cluster_vars)
        
        # constraint: each vertex in exactly one cluster
        if verbose:
            print('adding vertex non-overlapping constraint')
        for v in range(n_vertices):
            self.model.addConstr(quicksum([self.node_vars[i][v] for i in range(k)]), GRB.EQUAL, 1)
        
        # constraint: edge_var=1 indicates one and only one vertex of an edge is in cluster
        # edge_var[cluster][edge] = node_var[cluster][u] XOR node_var[cluster][v]
        # has_edge[cluster][edge] = node_var[cluster][u] AND node_var[cluster][v]
        if verbose:
            print('adding edges and cluster constraint')
        for i in range(k):
            for e in range(self.n_edges):
                u, v = self.edges[e]
                u_node_var = self.node_vars[i][u]
                v_node_var = self.node_vars[i][v]
                self.model.addConstr(self.edge_vars[i][e] <= u_node_var+v_node_var)
                self.model.addConstr(self.edge_vars[i][e] >= u_node_var-v_node_var)
                self.model.addConstr(self.edge_vars[i][e] >= v_node_var-u_node_var)
                self.model.addConstr(self.edge_vars[i][e] <= 2-u_node_var-v_node_var)
                # not_u_node_var = self.not_node_vars[i][u]
                # not_v_node_var = self.not_node_vars[i][v]
                # tmp1 = self.model.addVar(lb=0.0, ub=1.0, vtype=GRB.BINARY)
                # self.model.addConstr(tmp1 == and_(u_node_var, not_v_node_var))
                # tmp2 = self.model.addVar(lb=0.0, ub=1.0, vtype=GRB.BINARY)
                # self.model.addConstr(tmp2 == and_(not_u_node_var, v_node_var))
                # self.model.addConstr(self.edge_vars[i][e] == or_(tmp1, tmp2))
                # self.model.addConstr(self.has_edge[i][e] == and_(u_node_var, v_node_var))

        # symmetry-breaking constraints
        if verbose:
            print('adding symmetry-breaking constraints')
        self.model.addConstr(self.node_vars[0][0], GRB.EQUAL, 1)
        for i in range(2, k):
            self.model.addConstr(quicksum([self.node_vars[i-1][j] for j in range(n_vertices)]),
                            GRB.LESS_EQUAL,
                            quicksum([self.node_vars[i][j] for j in range(n_vertices)]))
        
        # Objective function
        if verbose:
            print('adding objective')
        obj_expr = LinExpr()
        for cluster in range(k):
            # FIXME: upper bound on variables should not be hardcoded
            cluster_K = self.model.addVar(lb=0, ub=10, vtype=GRB.INTEGER, name='cluster_K_%d'%cluster)
            self.model.addConstr(cluster_K == 
            quicksum([self.edge_vars[cluster][i] for i in range(self.n_edges)]))
            
            cluster_original_qubit = self.model.addVar(lb=0, ub=100, vtype=GRB.INTEGER)
            self.model.addConstr(cluster_original_qubit ==
            quicksum([self.node_qubits[id_nodes[i]]*self.node_vars[cluster][i]
            for i in range(self.n_vertices)]))
            
            cluster_cut_qubit = self.model.addVar(lb=0, ub=100, vtype=GRB.INTEGER)
            self.model.addConstr(cluster_cut_qubit ==
            quicksum([self.edge_vars[cluster][i] * self.node_vars[cluster][self.edges[i][1]]
            for i in range(self.n_edges)]))

            cluster_d = self.model.addVar(lb=0, ub=self.hw_max_qubit, vtype=GRB.INTEGER, name='cluster_d_%d'%cluster)
            self.model.addConstr(cluster_d == cluster_original_qubit + cluster_cut_qubit)
            
            lb = 1
            ub = 3*10+self.hw_max_qubit
            ptx, ptf = self.pwl_exp(2,lb,ub)
            cluster_hardness_exponent = self.model.addVar(lb=lb,ub=ub,vtype=GRB.INTEGER)
            self.model.addConstr(cluster_hardness_exponent == 3*cluster_K + cluster_d)
            self.model.setPWLObj(cluster_hardness_exponent, ptx, ptf)

        self.model.update()
        self.model.params.OutputFlag = self.verbosity
    
    def pwl_exp(self, base, lb, ub):
        ptx = []
        ptf = []

        for i in range(lb,ub+1):
            ptx.append(i)
            ptf.append(np.power(base,(ptx[i-lb]/10)))
        return ptx, ptf
    
    def check_graph(self, n_vertices, edges):
        # 1. edges must include all vertices
        # 2. all u,v must be ordered and smaller than n_vertices
        vertices = set([i for (i, _) in edges])
        vertices |= set([i for (_, i) in edges])
        assert(vertices == set(range(n_vertices)))
        for u, v in edges:
            assert(u < v)
            assert(u < n_vertices)
    
    def solve(self):
        try:
            self.model.optimize()
        except GurobiError:
            print(GurobiError.message)
        
        if self.model.solcount > 0:
            self.objective = None
            self.clusters = None
            self.optimal = (self.model.Status == GRB.OPTIMAL)
            self.runtime = self.model.Runtime
            self.node_count = self.model.nodecount
            self.mip_gap = self.model.mipgap
            self.objective = self.model.ObjVal

            clusters = []
            for i in range(self.k):
                cluster = []
                for j in range(self.n_vertices):
                    if abs(self.node_vars[i][j].x) > 1e-4:
                        cluster.append(self.id_nodes[j])
                clusters.append(cluster)
            self.clusters = clusters

            cut_edges_idx = []
            cut_edges = []
            for i in range(self.k):
                for j in range(self.n_edges):
                    if abs(self.edge_vars[i][j].x) > 1e-4 and j not in cut_edges_idx:
                        cut_edges_idx.append(j)
                        u, v = self.edges[j]
                        cut_edges.append((self.id_nodes[u], self.id_nodes[v]))
            self.cut_edges = cut_edges
            return True
        else:
            return False
    
    def print_stat(self):
        print('*'*200)
        print('MIQCP stats:')
        print('splitting %d vertices %d edges graph into %d clusters. Max qubit = %d'%
        (self.n_vertices, self.n_edges,self.k,self.hw_max_qubit))
        print('model has %d variables, %d linear constraints,%d quadratic constraints, %d general constraints'
        % (self.model.NumVars,self.model.NumConstrs, self.model.NumQConstrs, self.model.NumGenConstrs))
        # [print('cluster %d\n'%i, x) for i, x in enumerate(self.clusters)]
        # print('edges to cut:')
        # print(self.cut_edges)

        # print('node count:', self.node_count)
        # print('mip gap:', self.mip_gap)
        print('objective value:', self.objective)
        print('runtime:', self.runtime)

        # for v in self.model.getVars():
        #     if 'cluster' in v.VarName:
        #         print('%s %g' % (v.VarName, v.X))

        for i in range(self.k):
            cluster_K = self.model.getVarByName('cluster_K_%d'%i)
            cluster_d = self.model.getVarByName('cluster_d_%d'%i)
            print('cluster %d, K = %d, d = %d' % 
            (i,cluster_K.X,cluster_d.X))

        if (self.optimal):
            print('OPTIMAL')
        else:
            print('NOT OPTIMAL')
        print('*'*200)

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

def cuts_parser(cuts, circ):
    dag = circuit_to_dag(circ)
    positions = []
    for position in cuts:
        source, dest = position
        source_qargs = [x[:len(x)-1] for x in source.split(' ')]
        dest_qargs = [x[:len(x)-1] for x in dest.split(' ')]
        qubit_cut = list(set(source_qargs).intersection(set(dest_qargs)))
        if len(qubit_cut)>1:
            raise Exception('one cut is cutting on multiple qubits')
        for x in source.split(' '):
            if x[:len(x)-1] == qubit_cut[0]:
                source_idx = int(x[len(x)-1])
        for x in dest.split(' '):
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
    positions = sorted(positions, reverse=True, key=lambda cut: cut[1])
    return positions

def find_cuts(circ, hw_max_qubit=20, verbose=False):
    ub=int(3*len(circ.qubits)/hw_max_qubit)
    if ub<2:
        min_objective = np.power(2,len(circ.qubits)/10)
        best_positions = []
        best_K = [0]
        best_d = [len(circ.qubits)]
        best_num_cluster=ub
        return min_objective, best_positions, best_K, best_d, best_num_cluster
    num_clusters = range(2,ub+1)
    stripped_circ = r_s.circ_stripping(circ)
    n_vertices, edges, node_ids, id_nodes = read_circ(stripped_circ)
    min_objective = float('inf')
    best_positions = None
    best_K = None
    best_d = None
    best_num_cluster = None
    for num_cluster in num_clusters:
        kwargs = dict(n_vertices=n_vertices,
                    edges=edges,
                    node_ids=node_ids,
                    id_nodes=id_nodes,
                    k=num_cluster,
                    hw_max_qubit=hw_max_qubit,
                    verbose=verbose)

        m = Basic_Model(**kwargs)
        feasible = m.solve()
        if not feasible:
            continue
        if verbose:
            m.print_stat()
        
        if m.objective < min_objective:
            best_num_cluster = num_cluster
            min_objective = m.objective
            best_positions = cuts_parser(m.cut_edges, circ)
            best_K = []
            best_d = []
            for i in range(m.k):
                cluster_K = m.model.getVarByName('cluster_K_%d'%i)
                cluster_d = m.model.getVarByName('cluster_d_%d'%i)
                best_K.append(cluster_K.X)
                best_d.append(cluster_d.X)

    return min_objective, best_positions, best_K, best_d, best_num_cluster

if __name__ == '__main__':
    circ = gen_supremacy(2,3,8,'71230456')
    stripped_circ = r_s.circ_stripping(circ)
    n_vertices, edges, node_ids, id_nodes = read_circ(stripped_circ)
    k=2
    hw_max_qubit=20
    kwargs = dict(n_vertices=n_vertices,
                  edges=edges,
                  node_ids=node_ids,
                  id_nodes=id_nodes,
                  k=k,
                  hw_max_qubit=hw_max_qubit)
    print('splitting %d vertices %d edges graph into %d clusters. Max qubit = %d'%
    (n_vertices, len(edges),k,hw_max_qubit))

    m = Basic_Model(**kwargs)
    m.solve()
    m.print_stat()

    print('verifying with cutter')
    positions = cuts_parser(m.cut_edges, circ)
    print('cut positions:', positions)
    fragments, complete_path_map, K, d = cutter.cut_circuit(circ, positions)
    print('%d fragments, %d cuts'%(len(fragments),len(positions)),'K =', K, 'd =', d)
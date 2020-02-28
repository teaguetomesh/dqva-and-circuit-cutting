from qiskit.dagcircuit.dagcircuit import DAGCircuit
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.tools.visualization import dag_drawer
from gurobipy import *
import networkx as nx
from qcg.generators import gen_supremacy, gen_hwea
import numpy as np
import math

class Basic_Model(object):
    def __init__(self, num_cuts, n_vertices, edges, vertex_ids, id_vertices, num_cluster, num_qubits):
        self.check_graph(n_vertices, edges)
        self.num_cuts = num_cuts
        self.n_vertices = n_vertices
        self.edges = edges
        self.n_edges = len(edges)
        self.vertex_ids = vertex_ids
        self.id_vertices = id_vertices
        self.num_cluster = num_cluster
        self.num_qubits = num_qubits

        self.model = Model('cut_searching')
        self.model.params.OutputFlag = 0

        self.vertex_weight = {}
        for node in self.vertex_ids:
            qargs = node.split(' ')
            num_in_qubits = 0
            for qarg in qargs:
                if int(qarg.split(']')[1]) == 0:
                    num_in_qubits += 1
            self.vertex_weight[node] = num_in_qubits

        # Indicate if a vertex is in some cluster
        self.vertex_y = []
        for i in range(num_cluster):
            cluster_y = []
            for j in range(n_vertices):
                j_in_i = self.model.addVar(lb=0.0, ub=1.0, vtype=GRB.BINARY)
                cluster_y.append(j_in_i)
            self.vertex_y.append(cluster_y)

        # Indicate if an edge has one and only one vertex in some cluster
        self.edge_x = []
        for i in range(num_cluster):
            cluster_x = []
            for j in range(self.n_edges):
                v = self.model.addVar(lb=0.0, ub=1.0, vtype=GRB.BINARY)
                cluster_x.append(v)
            self.edge_x.append(cluster_x)
        
        # constraint: each vertex in exactly one cluster
        for v in range(n_vertices):
            self.model.addConstr(quicksum([self.vertex_y[i][v] for i in range(num_cluster)]), GRB.EQUAL, 1)
        
        # constraint: edge_var=1 indicates one and only one vertex of an edge is in cluster
        # edge_var[cluster][edge] = node_var[cluster][u] XOR node_var[cluster][v]
        for i in range(num_cluster):
            for e in range(self.n_edges):
                u, v = self.edges[e]
                u_vertex_y = self.vertex_y[i][u]
                v_vertex_y = self.vertex_y[i][v]
                self.model.addConstr(self.edge_x[i][e] <= u_vertex_y+v_vertex_y)
                self.model.addConstr(self.edge_x[i][e] >= u_vertex_y-v_vertex_y)
                self.model.addConstr(self.edge_x[i][e] >= v_vertex_y-u_vertex_y)
                self.model.addConstr(self.edge_x[i][e] <= 2-u_vertex_y-v_vertex_y)

        # Better (but not best) symmetry-breaking constraints
        #   Force small-numbered vertices into small-numbered clusters:
        #     v0: in cluster 0
        #     v1: in c0 or c1
        #     v2: in c0 or c1 or c2
        #     ....
        for vertex in range(num_cluster):
            self.model.addConstr(quicksum([self.vertex_y[cluster][vertex] for cluster in range(vertex+1,num_cluster)]) == 0)
        
        self.model.addConstr(self.num_cuts == 
        quicksum(
            [self.edge_x[cluster][i] for i in range(self.n_edges) for cluster in range(num_cluster)]
            )/2)
        
        for cluster in range(num_cluster):
            cluster_original_qubit = self.model.addVar(lb=0, ub=self.num_qubits-1, vtype=GRB.INTEGER, name='cluster_input_%d'%cluster)
            self.model.addConstr(cluster_original_qubit ==
            quicksum([self.vertex_weight[id_vertices[i]]*self.vertex_y[cluster][i]
            for i in range(self.n_vertices)]))
            
            cluster_rho_qubits = self.model.addVar(lb=0, ub=self.num_qubits-1, vtype=GRB.INTEGER, name='cluster_rho_qubits_%d'%cluster)
            self.model.addConstr(cluster_rho_qubits ==
            quicksum([self.edge_x[cluster][i] * self.vertex_y[cluster][self.edges[i][1]]
            for i in range(self.n_edges)]))

            cluster_O_qubits = self.model.addVar(lb=0, ub=self.num_qubits-1, vtype=GRB.INTEGER, name='cluster_O_qubits_%d'%cluster)
            self.model.addConstr(cluster_O_qubits ==
            quicksum([self.edge_x[cluster][i] * self.vertex_y[cluster][self.edges[i][0]]
            for i in range(self.n_edges)]))

            cluster_d = self.model.addVar(lb=0.1, ub=self.num_qubits-1, vtype=GRB.INTEGER, name='cluster_d_%d'%cluster)
            self.model.addConstr(cluster_d == cluster_original_qubit + cluster_rho_qubits)
            
            lb = 0
            ub = self.num_qubits*np.log(2)*2
            ptx, ptf = self.pwl_exp(params=[1,1],lb=lb,ub=ub,weight=1)
            evaluator_cost_exponent = self.model.addVar(lb=lb,ub=ub,vtype=GRB.CONTINUOUS, name='evaluator_cost_exponent_%d'%cluster)
            self.model.addConstr(evaluator_cost_exponent == np.log(2)*(cluster_rho_qubits+cluster_d))
            self.model.setPWLObj(evaluator_cost_exponent, ptx, ptf)

        self.model.update()
    
    def pwl_exp(self, params, lb, ub, weight):
        # Piecewise linear approximation of w*p_0*e^(p_1*x)
        ptx = []
        ptf = []

        num_pt = 500

        for i in range(num_pt):
            x = (ub-lb)/(num_pt-1)*i+lb
            y = weight*params[0]*np.exp(params[1]*x)
            ptx.append(x)
            ptf.append(y)
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
        # print('solving for %d clusters'%self.num_cluster)
        # print('model has %d variables, %d linear constraints,%d quadratic constraints, %d general constraints'
        # % (self.model.NumVars,self.model.NumConstrs, self.model.NumQConstrs, self.model.NumGenConstrs))
        # try:
        #     self.model.optimize()
        # except GurobiError:
        #     print(GurobiError)
        #     print(GurobiError.message)
        self.model.optimize()
        
        if self.model.solcount > 0:
            self.objective = None
            self.clusters = None
            self.optimal = (self.model.Status == GRB.OPTIMAL)
            self.runtime = self.model.Runtime
            self.node_count = self.model.nodecount
            self.mip_gap = self.model.mipgap
            self.objective = self.model.ObjVal

            clusters = []
            for i in range(self.num_cluster):
                cluster = []
                for j in range(self.n_vertices):
                    if abs(self.vertex_y[i][j].x) > 1e-4:
                        cluster.append(self.id_vertices[j])
                clusters.append(cluster)
            self.clusters = clusters

            cut_edges_idx = []
            cut_edges = []
            for i in range(self.num_cluster):
                for j in range(self.n_edges):
                    if abs(self.edge_x[i][j].x) > 1e-4 and j not in cut_edges_idx:
                        cut_edges_idx.append(j)
                        u, v = self.edges[j]
                        cut_edges.append((self.id_vertices[u], self.id_vertices[v]))
            self.cut_edges = cut_edges
            return True
        else:
            # print('Infeasible')
            return False
    
    def print_stat(self):
        print('*'*20)
        print('MIQCP stats:')
        # print('node count:', self.node_count)
        # print('%d vertices %d edges graph. Max qubit = %d'%
        # (self.n_vertices, self.n_edges, self.cluster_max_qubit))
        print('%d cuts, %d clusters, max qubit = %d'%(len(self.cut_edges),self.num_cluster,self.cluster_max_qubit))

        evaluator_cost_verify = 0
        for i in range(self.num_cluster):
            cluster_input = self.model.getVarByName('cluster_input_%d'%i)
            cluster_rho_qubits = self.model.getVarByName('cluster_rho_qubits_%d'%i)
            cluster_O_qubits = self.model.getVarByName('cluster_O_qubits_%d'%i)
            cluster_d = self.model.getVarByName('cluster_d_%d'%i)
            evaluator_cost_verify += 2**cluster_rho_qubits.X
            print('cluster %d: original input = %.2f, \u03C1_qubits = %.2f, O_qubits = %.2f, d = %.2f' % 
            (i,cluster_input.X,cluster_rho_qubits.X,cluster_O_qubits.X,cluster_d.X))

        print('objective value = %.3e'%self.objective)
        print('manually calculated objective value: %.3e'%evaluator_cost_verify)
        # print('mip gap:', self.mip_gap)
        print('runtime:', self.runtime)

        if (self.optimal):
            print('OPTIMAL')
        else:
            print('NOT OPTIMAL')
        print('*'*20)

def read_circ(circ):
    dag = circuit_to_dag(circ)
    edges = []
    node_name_ids = {}
    id_node_names = {}
    vertex_ids = {}
    curr_node_id = 0
    qubit_gate_idx = {}
    for qubit in dag.qubits():
        qubit_gate_idx[qubit] = 0
    for vertex in dag.topological_op_nodes():
        if len(vertex.qargs) != 2:
            raise Exception('vertex does not have 2 qargs!')
        arg0, arg1 = vertex.qargs
        vertex_name = '%s[%d]%d %s[%d]%d' % (arg0.register.name, arg0.index,qubit_gate_idx[arg0],
                                                arg1.register.name, arg1.index,qubit_gate_idx[arg1])
        qubit_gate_idx[arg0] += 1
        qubit_gate_idx[arg1] += 1
        if vertex_name not in node_name_ids and id(vertex) not in vertex_ids:
            node_name_ids[vertex_name] = curr_node_id
            id_node_names[curr_node_id] = vertex_name
            vertex_ids[id(vertex)] = curr_node_id
            curr_node_id += 1

    for u, v, _ in dag.edges():
        if u.type == 'op' and v.type == 'op':
            u_id = vertex_ids[id(u)]
            v_id = vertex_ids[id(v)]
            edges.append((u_id, v_id))
            
    n_vertices = len(list(dag.topological_op_nodes()))
    return n_vertices, edges, node_name_ids, id_node_names

def cuts_parser(cuts, circ):
    dag = circuit_to_dag(circ)
    positions = []
    for position in cuts:
        source, dest = position
        source_qargs = [(x.split(']')[0]+']',int(x.split(']')[1])) for x in source.split(' ')]
        dest_qargs = [(x.split(']')[0]+']',int(x.split(']')[1])) for x in dest.split(' ')]
        qubit_cut = []
        for source_qarg in source_qargs:
            source_qubit, source_multi_Q_gate_idx = source_qarg
            for dest_qarg in dest_qargs:
                dest_qubit, dest_multi_Q_gate_idx = dest_qarg
                if source_qubit==dest_qubit and dest_multi_Q_gate_idx == source_multi_Q_gate_idx+1:
                    qubit_cut.append(source_qubit)
        if len(qubit_cut)>1:
            raise Exception('one cut is cutting on multiple qubits')
        for x in source.split(' '):
            if x.split(']')[0]+']' == qubit_cut[0]:
                source_idx = int(x.split(']')[1])
        for x in dest.split(' '):
            if x.split(']')[0]+']' == qubit_cut[0]:
                dest_idx = int(x.split(']')[1])
        multi_Q_gate_idx = max(source_idx, dest_idx)
        
        wire = None
        for qubit in circ.qubits:
            if qubit.register.name == qubit_cut[0].split('[')[0] and qubit.index == int(qubit_cut[0].split('[')[1].split(']')[0]):
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

def circ_stripping(circ):
    # Remove all single qubit gates in the circuit
    dag = circuit_to_dag(circ)
    stripped_dag = DAGCircuit()
    [stripped_dag.add_qreg(x) for x in circ.qregs]
    for vertex in dag.topological_op_nodes():
        if len(vertex.qargs) >= 2:
            stripped_dag.apply_operation_back(op=vertex.op, qargs=vertex.qargs)
    return dag_to_circuit(stripped_dag)

def find_cuts(circ, num_cuts):
    stripped_circ = circ_stripping(circ)
    n_vertices, edges, vertex_ids, id_vertices = read_circ(stripped_circ)
    num_qubits = len(circ.qubits)
    num_clusters = range(2,num_cuts+2) # 2 to K+1
    min_objective = float('inf')
    solution_dict = {}

    for num_cluster in num_clusters:
        
        kwargs = dict(num_cuts=num_cuts,
                    n_vertices=n_vertices,
                    edges=edges,
                    vertex_ids=vertex_ids,
                    id_vertices=id_vertices,
                    num_cluster=num_cluster,
                    num_qubits=num_qubits)

        m = Basic_Model(**kwargs)
        feasible = m.solve()
        if not feasible or m.objective>min_objective:
            continue
        else:
            min_objective = m.objective
            positions = cuts_parser(m.cut_edges, circ)
            num_rho_qubits = []
            num_O_qubits = []
            num_d_qubits = []
            for i in range(m.num_cluster):
                cluster_rho_qubits = m.model.getVarByName('cluster_rho_qubits_%d'%i)
                cluster_O_qubits = m.model.getVarByName('cluster_O_qubits_%d'%i)
                cluster_d = m.model.getVarByName('cluster_d_%d'%i)
                num_rho_qubits.append(cluster_rho_qubits.X)
                num_O_qubits.append(cluster_O_qubits.X)
                num_d_qubits.append(cluster_d.X)
            solution_dict = {'model':m,
            'circ':circ,
            'searcher_time':m.runtime,
            'num_rho_qubits':num_rho_qubits,
            'num_O_qubits':num_O_qubits,
            'num_d_qubits':num_d_qubits,
            'objective':m.objective,
            'positions':positions,
            'num_cluster':m.num_cluster}
    return solution_dict

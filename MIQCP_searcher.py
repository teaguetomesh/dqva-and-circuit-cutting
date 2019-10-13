from qiskit.dagcircuit.dagcircuit import DAGCircuit
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.tools.visualization import dag_drawer
from gurobipy import *
import networkx as nx
from qcg.generators import gen_supremacy, gen_hwea
import numpy as np
import cutter

class Basic_Model(object):
    def __init__(self, n_vertices, edges, node_ids, id_nodes, k, hw_max_qubit, evaluator_weight):
        self.check_graph(n_vertices, edges)
        self.n_vertices = n_vertices
        self.edges = edges
        self.n_edges = len(edges)
        self.node_ids = node_ids
        self.id_nodes = id_nodes
        self.k = k
        self.hw_max_qubit = hw_max_qubit
        self.evaluator_weight = evaluator_weight
        self.verbosity = 0

        self.model = Model('cut_searching')
        self.model.params.OutputFlag = 0
        # self.model.params.NodeLimit = 50
        # self.model.params.MIPGap = 0.5

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
        for i in range(k):
            cluster_vars = []
            for j in range(n_vertices):
                j_in_i = self.model.addVar(lb=0.0, ub=1.0, vtype=GRB.BINARY)
                cluster_vars.append(j_in_i)
            self.node_vars.append(cluster_vars)

        # Indicate if an edge has one and only one vertex in some cluster
        self.edge_vars = []
        for i in range(k):
            cluster_vars = []
            for j in range(self.n_edges):
                v = self.model.addVar(lb=0.0, ub=1.0, vtype=GRB.BINARY)
                cluster_vars.append(v)
            self.edge_vars.append(cluster_vars)
        
        # constraint: each vertex in exactly one cluster
        for v in range(n_vertices):
            self.model.addConstr(quicksum([self.node_vars[i][v] for i in range(k)]), GRB.EQUAL, 1)
        
        # constraint: edge_var=1 indicates one and only one vertex of an edge is in cluster
        # edge_var[cluster][edge] = node_var[cluster][u] XOR node_var[cluster][v]
        for i in range(k):
            for e in range(self.n_edges):
                u, v = self.edges[e]
                u_node_var = self.node_vars[i][u]
                v_node_var = self.node_vars[i][v]
                self.model.addConstr(self.edge_vars[i][e] <= u_node_var+v_node_var)
                self.model.addConstr(self.edge_vars[i][e] >= u_node_var-v_node_var)
                self.model.addConstr(self.edge_vars[i][e] >= v_node_var-u_node_var)
                self.model.addConstr(self.edge_vars[i][e] <= 2-u_node_var-v_node_var)

        # FIXME: cluster is allowed to have 0 qubits
        # symmetry-breaking constraints
        # TODO: this does not break all the symmetries
        # TODO: is this necessary?
        # self.model.addConstr(self.node_vars[0][0], GRB.EQUAL, 1)
        # for i in range(2, k):
        #     self.model.addConstr(quicksum([self.node_vars[i-1][j] for j in range(n_vertices)]),
        #                     GRB.LESS_EQUAL,
        #                     quicksum([self.node_vars[i][j] for j in range(n_vertices)]))
        
        # Objective function
        lb = 0
        ub = 50
        total_num_cuts = self.model.addVar(lb=lb, ub=ub, vtype=GRB.INTEGER, name='total_num_cuts')
        self.model.addConstr(total_num_cuts == 
        quicksum(
            [self.edge_vars[cluster][i] for i in range(self.n_edges) for cluster in range(k)]
            ))
        list_of_cluster_d = []
        for cluster in range(k):
            cluster_K = self.model.addVar(lb=0, ub=30, vtype=GRB.INTEGER, name='cluster_K_%d'%cluster)
            self.model.addConstr(cluster_K == 
            quicksum([self.edge_vars[cluster][i] for i in range(self.n_edges)]))
            
            cluster_original_qubit = self.model.addVar(lb=0, ub=self.hw_max_qubit, vtype=GRB.INTEGER, name='cluster_input_%d'%cluster)
            self.model.addConstr(cluster_original_qubit ==
            quicksum([self.node_qubits[id_nodes[i]]*self.node_vars[cluster][i]
            for i in range(self.n_vertices)]))
            # self.model.addConstr(cluster_original_qubit >= 0.1)
            
            cluster_rho_qubits = self.model.addVar(lb=0, ub=self.hw_max_qubit, vtype=GRB.INTEGER, name='cluster_rho_qubits_%d'%cluster)
            self.model.addConstr(cluster_rho_qubits ==
            quicksum([self.edge_vars[cluster][i] * self.node_vars[cluster][self.edges[i][1]]
            for i in range(self.n_edges)]))

            cluster_O_qubits = self.model.addVar(lb=0, ub=self.hw_max_qubit, vtype=GRB.INTEGER, name='cluster_O_qubits_%d'%cluster)
            self.model.addConstr(cluster_O_qubits ==
            quicksum([self.edge_vars[cluster][i] * self.node_vars[cluster][self.edges[i][0]]
            for i in range(self.n_edges)]))

            cluster_d = self.model.addVar(lb=0, ub=self.hw_max_qubit, vtype=GRB.INTEGER, name='cluster_d_%d'%cluster)
            self.model.addConstr(cluster_d == cluster_original_qubit + cluster_rho_qubits)
            list_of_cluster_d.append(cluster_d)
            
            lb = 0
            ub = 100
            ptx, ptf = self.pwl_exp(2,lb,ub,self.evaluator_weight)
            evaluator_hardness_exponent = self.model.addVar(lb=lb,ub=ub,vtype=GRB.CONTINUOUS, name='evaluator_cost_exponent_%d'%cluster)
            self.model.addConstr(evaluator_hardness_exponent == (np.log2(6)*cluster_rho_qubits + np.log2(3)*cluster_O_qubits))
            self.model.setPWLObj(evaluator_hardness_exponent, ptx, ptf)

            if cluster>0:
                lb = 0
                ub = 100
                uniter_hardness_exponent = self.model.addVar(lb=lb,ub=ub,vtype=GRB.CONTINUOUS, name='uniter_cost_exponent_%d'%cluster)
                self.model.addConstr(uniter_hardness_exponent == total_num_cuts+quicksum(list_of_cluster_d))
                ptx, ptf = self.pwl_exp(2,lb,ub,1-self.evaluator_weight)
                self.model.setPWLObj(uniter_hardness_exponent, ptx, ptf)

        self.model.update()
    
    def pwl_exp(self, base, lb, ub, coefficient=1):
        ptx = []
        ptf = []

        num_pt = 400

        for i in range(num_pt):
            x = (ub-lb)/(num_pt-1)*i+lb
            y = np.power(base,x)*coefficient
            ptx.append(x)
            ptf.append(y)
        # ptx.append(ub+1)
        # ptf.append(float('inf'))
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
        print('solving for %d clusters'%self.k)
        # print('model has %d variables, %d linear constraints,%d quadratic constraints, %d general constraints'
        # % (self.model.NumVars,self.model.NumConstrs, self.model.NumQConstrs, self.model.NumGenConstrs))
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
            print('Infeasible')
            return False
    
    def print_stat(self):
        print('MIQCP stats:')
        print('node count:', self.node_count)
        print('%d vertices %d edges graph. Max qubit = %d'%
        (self.n_vertices, self.n_edges, self.hw_max_qubit))
        print('%d cuts, %d clusters'%(len(self.cut_edges),self.k))

        evaluator_cost_verify = 0
        uniter_cost_verify = 0
        for i in range(self.k):
            cluster_input = self.model.getVarByName('cluster_input_%d'%i)
            cluster_rho_qubits = self.model.getVarByName('cluster_rho_qubits_%d'%i)
            cluster_O_qubits = self.model.getVarByName('cluster_O_qubits_%d'%i)
            cluster_d = self.model.getVarByName('cluster_d_%d'%i)
            cluster_K = self.model.getVarByName('cluster_K_%d'%i)
            evaluator_cost_verify += np.power(6,cluster_rho_qubits.X)*np.power(3,cluster_O_qubits.X)
            print('cluster %d: original input = %.2f, rho qubits = %.2f, O qubits = %.2f, d = %.2f, K = %.2f' % 
            (i,cluster_input.X,cluster_rho_qubits.X,cluster_O_qubits.X,cluster_d.X,cluster_K.X))
            if i>0:
                uniter_cost_exponent = self.model.getVarByName('uniter_cost_exponent_%d'%i)
                # print('uniter cost exponent = ',uniter_cost_exponent.X)
                uniter_cost_verify += np.power(2,uniter_cost_exponent.X)

        print('objective value:', self.objective)
        print('manually calculated objective value:', self.evaluator_weight*evaluator_cost_verify+(1-self.evaluator_weight)*uniter_cost_verify)
        print('mip gap:', self.mip_gap)
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
    for vertex in dag.topological_op_nodes():
        if len(vertex.qargs) != 2:
            raise Exception('vertex does not have 2 qargs!')
        arg0, arg1 = vertex.qargs
        vertex_name = '%s[%d]%d %s[%d]%d' % (arg0[0].name, arg0[1],qubit_gate_idx[arg0],
                                                arg1[0].name, arg1[1],qubit_gate_idx[arg1])
        qubit_gate_idx[arg0] += 1
        qubit_gate_idx[arg1] += 1
        if vertex_name not in node_name_ids and id(vertex) not in node_ids:
            node_name_ids[vertex_name] = curr_node_id
            id_node_names[curr_node_id] = vertex_name
            node_ids[id(vertex)] = curr_node_id
            curr_node_id += 1

    for u, v, _ in dag.edges():
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

def circ_stripping(circ):
    # Remove all single qubit gates in the circuit
    dag = circuit_to_dag(circ)
    stripped_dag = DAGCircuit()
    [stripped_dag.add_qreg(x) for x in circ.qregs]
    for vertex in dag.topological_op_nodes():
        if len(vertex.qargs) >= 2:
            stripped_dag.apply_operation_back(op=vertex.op, qargs=vertex.qargs)
    return dag_to_circuit(stripped_dag)

def find_cuts(circ, num_clusters = range(1,5), hw_max_qubit=20,evaluator_weight=1):
    min_objective = float('inf')
    best_positions = None
    best_ancilla = None
    best_d = None
    best_num_cluster = None
    best_model = None
    stripped_circ = circ_stripping(circ)
    n_vertices, edges, node_ids, id_nodes = read_circ(stripped_circ)

    for num_cluster in num_clusters:
        kwargs = dict(n_vertices=n_vertices,
                    edges=edges,
                    node_ids=node_ids,
                    id_nodes=id_nodes,
                    k=num_cluster,
                    hw_max_qubit=hw_max_qubit,
                    evaluator_weight=evaluator_weight)

        m = Basic_Model(**kwargs)
        feasible = m.solve()
        if not feasible:
            continue
        
        if m.objective < min_objective:
            best_num_cluster = num_cluster
            min_objective = m.objective
            best_positions = cuts_parser(m.cut_edges, circ)
            best_ancilla = []
            best_d = []
            best_model = m
            for i in range(m.k):
                cluster_rho_qubits = m.model.getVarByName('cluster_rho_qubits_%d'%i)
                cluster_d = m.model.getVarByName('cluster_d_%d'%i)
                best_ancilla.append(cluster_rho_qubits.X)
                best_d.append(cluster_d.X)

    return min_objective, best_positions, best_ancilla, best_d, best_num_cluster, best_model

if __name__ == '__main__':
    circ = gen_supremacy(4,4,8)
    hardness, positions, K, d, num_cluster, m = find_cuts(circ,num_clusters=[1,2,3,4],hw_max_qubit=15)
    m.print_stat()
    fragments, complete_path_map, K, d = cutter.cut_circuit(circ, positions)
    print('Testing in cutter:')
    [print(x, complete_path_map[x]) for x in complete_path_map]
    print(d)
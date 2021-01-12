import networkx as nx

def graph_from_file(fn):
    with open(fn, 'r') as f:
        edgelist = f.readline().split(',')
        edges = []
        for e in edgelist:
            if '(' in e:
                curedge = []
                curedge.append(int(e.strip('( ')))
            elif ')' in e:
                curedge.append(int(e.strip(') ')))
                edges.append(tuple(curedge))

        G = nx.Graph()
        G.add_edges_from(edges)
    return G

def square_graph():
    G = nx.Graph()
    G.add_nodes_from(range(4))
    edge_list = list(range(4)) + [0]
    print(edge_list)
    for i in range(len(edge_list)-1):
        G.add_edge(edge_list[i], edge_list[i+1])
    return G

def bowtie_graph():
    G = nx.Graph()
    G.add_nodes_from(range(7))
    edge_list1 = list(range(4)) + [0]
    edge_list2 = list(range(3,7)) + [3]
    print(edge_list1)
    print(edge_list2)
    for edge_list in [edge_list1, edge_list2]:
        for i in range(len(edge_list)-1):
            G.add_edge(edge_list[i], edge_list[i+1])
    return G

def test_graph(n, p):
    G1 = nx.erdos_renyi_graph(n, p)
    G2 = nx.erdos_renyi_graph(n, p)

    # Make a combined graph using the two subgraphs
    G = nx.Graph()

    # Add nodes and edges from G1
    G.add_nodes_from(G1.nodes)
    G.add_edges_from(G1.edges)

    # Add nodes and edges from G2
    offset = len(G1.nodes)

    g2_nodes = [n+offset for n in G2.nodes]
    G.add_nodes_from(g2_nodes)

    g2_edges = [(n1+offset, n2+offset) for n1, n2 in G2.edges]
    G.add_edges_from(g2_edges)

    # Connect the two subgraphs
    G.add_edge(list(G1.nodes)[-1], list(G2.nodes)[0]+offset)

    return G

def ring_graph(n):
    G = nx.Graph()
    G.add_nodes_from(list(range(n)))
    edges = [(i, i+1) for i in range(n-1)] + [(n-1, 0)]
    G.add_edges_from(edges)
    return G

def view_partition(partition, G):
    node_colors = []
    for node in G.nodes:
        if node in partition[0]:
            node_colors.append('gold')
        else:
            node_colors.append('lightblue')

    edge_colors = []
    for edge in G.edges:
        if (edge[0] in partition[0] and edge[1] in partition[1]) or \
           (edge[0] in partition[1] and edge[1] in partition[0]):
            edge_colors.append('red')
        else:
            edge_colors.append('black')
    nx.draw_spring(G, with_labels=True, node_color=node_colors, edge_color=edge_colors)

def get_subgraphs(G, partition):
    subgraphs = []
    cut_edges = []
    all_edges = G.edges
    for subgraph_nodes in partition:
        subG = nx.Graph()
        subG.add_nodes_from(subgraph_nodes)

        for v1, v2 in all_edges:
            if v1 in subgraph_nodes and v2 in subgraph_nodes:
                subG.add_edge(v1, v2)
            if v1 in subgraph_nodes and v2 not in subgraph_nodes:
                cut_edges.append((v1, v2))

        subgraphs.append(subG)

    return subgraphs, cut_edges

def is_indset(bitstr, G):
    nodes = list(G.nodes)
    ind_set = []
    for idx, bit in enumerate(reversed(bitstr)):
        if bit == '1':
            cur_neighbors = list(G.neighbors(idx))
            for node in ind_set:
                if node in cur_neighbors:
                    return False
            else:
                ind_set.append(idx)
    return True

import glob
import networkx as nx

dirs = glob.glob('N*_p*')

for folder in dirs:
    print(folder)
    n = int(folder.split('_')[0][1:])
    p = int(folder.split('_')[1][1:]) / 100
    print('Nodes: {}, probability: {}'.format(n, p))

    count = 0
    while count < 50:
        G = nx.generators.random_graphs.erdos_renyi_graph(n, p)
        if nx.is_connected(G):
            count += 1
            edges = list(G.edges())

            with open(folder+'/G{}.txt'.format(count), 'w') as fn:
                edgestr = ''.join(['{}, '.format(e) for e in edges])
                edgestr = edgestr.strip(', ')
                fn.write(edgestr)


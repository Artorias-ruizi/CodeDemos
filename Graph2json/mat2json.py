import json

import scipy.io as sio
import numpy as np
import networkx as nx
from networkx.readwrite import json_graph


def load_data_for_Wang(
        mat_path='0SingleObjectiveExperiments\PSO_CR\Data\BA_n200_deg5_ori\data_BA_n200_deg5_dir_nets.mat'):
    mat = sio.loadmat(mat_path)
    # mat = h5py.File(mat_path)  # v7.3
    g_ori = mat['g_ori']
    mat = mat['data']
    LENNETS = len(mat)
    for i in range(LENNETS):
        # 一定要区分Graph和DiGraph
        tmp_G = nx.DiGraph(mat[i, 0]['adj'][0, 0])
        with open(
                'D:\ProgramData\PycharmProject\MOEA\\0SingleObjectiveExperiments\PSO_CNN_CR\Data\BA_n200_deg5_ori\BA_n200_deg5_reduce_r_ind_' + str(
                    i + 1), 'w') as f:
            for edge in tmp_G.edges:
                # print(edge)
                f.write(f'{edge[0]} {edge[1]}\n')
            f.close()
    g_ori = nx.DiGraph(g_ori[0, 0]['adj'][0, 0])
    with open(
            'D:\ProgramData\PycharmProject\MOEA\\0SingleObjectiveExperiments\PSO_CNN_CR\Data\BA_n200_deg5_ori\BA_n200_deg5_reduce_r_ind_0',
            'w') as f:
        for edge in g_ori.edges:
            # print(edge)
            f.write(f'{edge[0]} {edge[1]}\n')


if __name__ == '__main__':
    mat = sio.loadmat('./data_BA_n200_deg5_dir_nets.mat')
    mat = mat['data']
    adj = mat[0, 0]['adj'][0, 0]
    adj = adj.todense()

    # save net to json
    G = json_graph.node_link_data(nx.DiGraph(adj))
    G = json.dumps(G)
    with open('test.json', mode='w') as f:
        f.write(G)
        f.close()

    # load net from json

    G_data = json.load(open('test.json'))
    G = json_graph.node_link_graph(G_data, directed=True)
    adj1 = nx.adjacency_matrix(G)
    print(adj1)

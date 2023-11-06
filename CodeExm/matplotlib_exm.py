import networkx as nx
import matplotlib as mpl
import matplotlib.pyplot as plt

params = {'legend.fontsize': 'large',
          'figure.figsize': (5.0, 5.0),
          'axes.labelsize': 'x-large',
          'axes.titlesize': 'xx-large',
          'xtick.labelsize': 'large',
          'ytick.labelsize': 'large'}
mpl.rcParams.update(params)

if __name__ == '__main__':
    g = nx.erdos_renyi_graph(10, 0.25)

    nx.draw(g, with_labels=True, node_size=1500, node_color="skyblue", node_shape="s", alpha=0.5, linewidths=10)
    plt.show()

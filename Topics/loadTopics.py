import networkx as nx
import matplotlib.pyplot as plt


if __name__ == '__main__':
    G = nx.read_edgelist("withouttime/line 10 of Shanghai-Metro pileup", create_using=nx.DiGraph())
    print(f"Graph type:{G}")
    nx.draw(G, with_labels=True)
    plt.show()
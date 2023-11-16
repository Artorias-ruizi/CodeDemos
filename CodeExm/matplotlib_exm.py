import networkx as nx
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

params = {
          'font.family': 'Times New Roman',
          'figure.figsize': (5.0, 5.0),
          'axes.labelsize': 'x-large',
          'axes.titlesize': 'xx-large',
          'xtick.labelsize': 'large',
          'ytick.labelsize': 'large',
          'legend.loc': 'lower left',
          'legend.fontsize': 'large'
          }
mpl.rcParams.update(params)

if __name__ == '__main__':
    pass
    # x = np.linspace(0, 2 * np.pi, 100)
    # y1 = np.sin(x)
    # y2 = np.cos(x)
    # plt.xlabel('x')
    # plt.ylabel('y')
    # plt.title('Sin and Cos Functions')
    # plt.plot(x, y1, 'ro--', label='Sin')
    # plt.plot(x, y2, 'bo--', label='Cos')
    # plt.legend()
    # plt.show()

    fig, axs = plt.subplots(nrows=2, ncols=3)
    axs[0, 0].plot([1, 2, 3], [4, 5, 6])
    axs[0, 1].scatter([1, 2, 3], [4, 5, 6])
    axs[0, 2].hist([1, 1, 2, 2, 2, 3, 3, 3, 3, 3])
    axs[1, 0].bar([1, 2, 3], [4, 5, 6])
    axs[1, 1].pie([1, 2, 3])

    plt.legend()
    plt.show()
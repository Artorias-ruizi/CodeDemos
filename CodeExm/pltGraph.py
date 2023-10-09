import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd


def pltGraph1():

    # Build a dataframe with your connections
    df = pd.DataFrame({'from': ['A', 'B', 'C', 'A'], 'to': ['D', 'A', 'E', 'C']})

    # Build your graph 建立表格
    G = nx.from_pandas_edgelist(df, 'from', 'to')

    # Graph with Custom nodes: 自定义表格
    # with_labels是否显示标签，node_size节点大小，node_color节点颜色，node_shape节点形状，alpha透明度，linewidths线条宽度
    nx.draw(G, with_labels=True, node_size=1500, node_color="skyblue", node_shape="s", alpha=0.5, linewidths=10)
    plt.show()

def pltGraph2():
    G = nx.Graph()

    G.add_edge('a', 'b', weight=0.6)
    G.add_edge('a', 'c', weight=0.2)
    G.add_edge('c', 'd', weight=0.1)
    G.add_edge('c', 'e', weight=0.7)
    G.add_edge('c', 'f', weight=0.9)
    G.add_edge('a', 'd', weight=0.3)

    elarge = [(u, v) for (u, v, d) in G.edges(data=True) if d['weight'] > 0.5]
    esmall = [(u, v) for (u, v, d) in G.edges(data=True) if d['weight'] <= 0.5]

    pos = nx.spring_layout(G)  # positions for all nodes

    # nodes
    nx.draw_networkx_nodes(G, pos, node_size=700)

    # edges
    nx.draw_networkx_edges(G, pos, edgelist=elarge, width=6)
    nx.draw_networkx_edges(G, pos, edgelist=esmall, width=6, alpha=0.5, edge_color='b', style='dashed')

    # labels
    nx.draw_networkx_labels(G, pos, font_size=20, font_family='sans-serif')

    plt.axis('off')
    plt.show()

def pltGraph3():
    # Build a dataframe with your connections
    df = pd.DataFrame({'from': ['A', 'B', 'C', 'A'], 'to': ['D', 'A', 'E', 'C']})

    # Build your graph
    G = nx.from_pandas_edgelist(df, 'from', 'to')

    # Custom the edges:
    # font_size标签字体大小，font_color标签字体颜色,font_weight字体形式
    nx.draw(G, with_labels=True, node_size=1500, font_size=25, font_color="yellow", font_weight="bold")
    plt.show()

def pltGraph4():
    # Build a dataframe with your connections
    df = pd.DataFrame({'from': ['A', 'B', 'C', 'A'], 'to': ['D', 'A', 'E', 'C']})
    # Build your graph
    G = nx.from_pandas_edgelist(df, 'from', 'to')
    # Chart with Custom edges:
    # width边线条宽，edge_color边线条颜色
    nx.draw(G, with_labels=True, width=10, edge_color="skyblue", style="solid")
    plt.show()

def pltGraph5():
    # Build a dataframe with your connections
    df = pd.DataFrame({'from': ['A', 'B', 'C', 'A'], 'to': ['D', 'A', 'E', 'C']})

    # Build your graph
    G = nx.from_pandas_edgelist(df, 'from', 'to')

    # All together we can do something fancy
    nx.draw(G, with_labels=True, node_size=1500, node_color="skyblue", node_shape="o", alpha=0.5, linewidths=4,
            font_size=25, font_color="grey", font_weight="bold", width=2, edge_color="grey")
    plt.show()

def pltGraph6Layout():
    # Build a dataframe with your connections
    df = pd.DataFrame({'from': ['A', 'B', 'C', 'A', 'E', 'F', 'E', 'G', 'G', 'D', 'F'],
                       'to': ['D', 'A', 'E', 'C', 'A', 'F', 'G', 'D', 'B', 'G', 'C']})
    # Build your graph
    G = nx.from_pandas_edgelist(df, 'from', 'to')

    # Fruchterman Reingold Fruchterman Reingold引导布局算法布局
    nx.draw(G, with_labels=True, node_size=1500, node_color="skyblue", pos=nx.fruchterman_reingold_layout(G))
    plt.title("fruchterman_reingold")
    plt.show()
    # # Circular 环形布局
    # nx.draw(G, with_labels=True, node_size=1500, node_color="skyblue", pos=nx.circular_layout(G))
    # plt.title("circular")
    # # Random 随机布局
    # nx.draw(G, with_labels=True, node_size=1500, node_color="skyblue", pos=nx.random_layout(G))
    # plt.title("random")
    # # Spectral 光谱式布局
    # nx.draw(G, with_labels=True, node_size=1500, node_color="skyblue", pos=nx.spectral_layout(G))
    # plt.title("spectral")
    # # Spring 跳跃式布局
    # nx.draw(G, with_labels=True, node_size=1500, node_color="skyblue", pos=nx.spring_layout(G))
    # plt.title("spring")

def pltGraph7Color():
    # Build a dataframe with your connections
    df = pd.DataFrame({'from': ['A', 'B', 'C', 'A'], 'to': ['D', 'A', 'E', 'C']})

    # And a data frame with characteristics for your nodes
    carac = pd.DataFrame(
        {'ID': ['A', 'B', 'C', 'D', 'E'], 'myvalue': ['group1', 'group1', 'group2', 'group3', 'group3']})

    # Build your graph
    # 建立图
    G = nx.from_pandas_edgelist(df, 'from', 'to', create_using=nx.Graph())

    # The order of the node for networkX is the following order:
    # 打印节点顺序
    G.nodes()
    # Thus, we cannot give directly the 'myvalue' column to netowrkX, we need to arrange the order!

    # Here is the tricky part: I need to reorder carac to assign the good color to each node
    carac = carac.set_index('ID')
    # 根据节点顺序设定值
    carac = carac.reindex(G.nodes())

    # And I need to transform my categorical column in a numerical value: group1->1, group2->2...
    # 设定类别
    carac['myvalue'] = pd.Categorical(carac['myvalue'])
    carac['myvalue'].cat.codes

    # Custom the nodes:
    nx.draw(G, with_labels=True, node_color=carac['myvalue'].cat.codes, cmap=plt.cm.Set1, node_size=1500)
    plt.show()

def pltGraph8EdgeColor():
    # Build a dataframe with your connections
    # value设定链接值
    df = pd.DataFrame({'from': ['A', 'B', 'C', 'A'], 'to': ['D', 'A', 'E', 'C'], 'value': [1, 10, 5, 5]})

    # Build your graph
    G = nx.from_pandas_edgelist(df, 'from', 'to', create_using=nx.Graph())

    # Custom the nodes:
    # edge_color设置边的颜色
    nx.draw(G, with_labels=True, node_color='skyblue', node_size=1500, edge_color=df['value'], width=10.0,
            edge_cmap=plt.cm.Blues)
    plt.show()

def pltGraph9EdgeColor2():
    ## 类别型 categorical
    # libraries
    import pandas as pd
    import numpy as np
    import networkx as nx
    import matplotlib.pyplot as plt

    # Build a dataframe with your connections
    # value设置类型
    df = pd.DataFrame(
        {'from': ['A', 'B', 'C', 'A'], 'to': ['D', 'A', 'E', 'C'], 'value': ['typeA', 'typeA', 'typeB', 'typeB']})
    df

    # And I need to transform my categorical column in a numerical value typeA->1, typeB->2...
    # 转换为类别
    df['value'] = pd.Categorical(df['value'])
    df['value'].cat.codes

    # Build your graph
    G = nx.from_pandas_edgelist(df, 'from', 'to', create_using=nx.Graph())

    # Custom the nodes:
    nx.draw(G, with_labels=True, node_color='skyblue', node_size=1500, edge_color=df['value'].cat.codes, width=10.0,
            edge_cmap=plt.cm.Set2)
    plt.show()

def pltMLP():
    import matplotlib.pyplot as plt
    import networkx as nx
    left, right, bottom, top, layer_sizes = .1, .9, .1, .9, [4, 7, 7, 2]
    # 网络离上下左右的距离
    # layter_sizes可以自己调整
    import random
    G = nx.Graph()
    v_spacing = (top - bottom) / float(max(layer_sizes))
    h_spacing = (right - left) / float(len(layer_sizes) - 1)
    node_count = 0
    for i, v in enumerate(layer_sizes):
        layer_top = v_spacing * (v - 1) / 2. + (top + bottom) / 2.
        for j in range(v):
            G.add_node(node_count, pos=(left + i * h_spacing, layer_top - j * v_spacing))
            node_count += 1
    # 这上面的数字调整我想了好半天，汗
    for x, (left_nodes, right_nodes) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        for i in range(left_nodes):
            for j in range(right_nodes):
                G.add_edge(i + sum(layer_sizes[:x]), j + sum(layer_sizes[:x + 1]))
                # 慢慢研究吧
    pos = nx.get_node_attributes(G, 'pos')
    # 把每个节点中的位置pos信息导出来
    nx.draw(G, pos,
            node_color=range(node_count),
            with_labels=True,
            node_size=200,
            edge_color=[random.random() for i in range(len(G.edges))],
            width=3,
            cmap=plt.cm.Dark2,  # matplotlib的调色板，可以搜搜，很多颜色呢
            edge_cmap=plt.cm.Blues
            )
    plt.show()
if __name__ == '__main__':
    pltGraph9EdgeColor2()
import networkx as nx
import matplotlib.pyplot as plt

def viz_graph(filelist, pairs):
    # Create a directed graph
    G = nx.DiGraph()

    # Add nodes
    for i, file in enumerate(filelist):
        file = file.split("/")[-1][:-4]
        G.add_node(i, label=file)

    # Add edges
    for pair in pairs:
        src = filelist.index(pair[0]["path"])
        dst = filelist.index(pair[1]["path"])
        G.add_edge(src, dst)

    # Draw the graph
    pos = nx.spring_layout(G)
    labels = nx.get_node_attributes(G, 'label')
    nx.draw(G, pos, with_labels=True, labels=labels, node_size=3000, node_color='skyblue', font_size=10, font_color='black', font_weight='bold', arrows=True)
    plt.show()
"""
Here, we want to take in the Grid() object and fully visualize it!
I'm envisioning
- different symbols for various types of nodes:
  - solar, wind, hydro, nuclear, battery storage, 
    consumer vs. corporation, distributed system
- color / bound microgrids together?
- add a map of the city to the background?
- color code the edges based on how much voltage?
- would be insane if we could visualize a live simulation of the grid, as it runs...
"""

"""
networkx is the basic graph library, with hundreds of graph algorithms implemented 
- https://networkx.org/documentation/stable/reference/algorithms/index.html
"""
import networkx as nx
import matplotlib as mpl
import matplotlib.pyplot as plt

def test_networkx():
    G = nx.Graph()

    nodes = [(1, {'type': 'solar'}), (2, {'type': 'wind'}), (3, {'type': 'hydro'})]
    edges = [(1, 2, {'voltage': 10}), (2, 3, {'voltage': 20})]

    G.add_nodes_from(nodes)
    G.add_edges_from(edges)

    nx.draw(G, with_labels=True)
    plt.show()

    return G


"""
PyVis is a python wrapper for VisJS
interactive graph visualizations as a local html file
"""
from pyvis.network import Network
import os
# save local PyVis visualizations to htmls directory


def test_pyvis():
    net = Network()  # create graph object

    # add nodes
    net.add_nodes(
        [1, 2, 3, 4, 5],  # node ids
        label=['Node #1', 'Node #2', 'Node #3', 'Node #4', 'Node #5'],  # node labels
        # node titles (display on mouse hover)
        title=['Main node', 'Just node', 'Just node', 'Just node', 'Node with self-loop'],
        color=['#d47415', '#22b512', '#42adf5', '#4a21b0', '#e627a3']  # node colors (HEX)
    )
    # add list of edges, same as in the previous example
    net.add_edges([(1, 2), (1, 3), (2, 3), (2, 4), (3, 5), (5, 5)])

   # toggle physics simulation when dragging vertices to reshape grid
    net.toggle_physics(True) 
    # show editable physics simulation options
    net.show_buttons(filter_=['physics'])  # show only physics simulation

    net.show('graph.html')    # save visualization in 'graph.html'

    return net

if __name__ == "__main__":
    G = test_networkx()
    #P = test_pyvis()

    # can also turn networkx graphs into pyvis graphs!
    nt = Network()
    nt.from_nx(G)
    nt.show('nx.html')
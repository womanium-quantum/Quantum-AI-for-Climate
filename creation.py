import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

"""
FIRST THING TO DO:
Create some object-oriented structure to store an electrical grid

networkx might have a better implementation of a graph.... but bear with me

Grid(): complete electrical grid object, with the following structure:
- edges: dict[(nodeId, nodeId): Edge()]
- vertices: dict[nodeId: Node()]
- numNodes: int (nodeIds are 0, 1, 2, ..., numNodes-1) and are updated automatically

TODO:
- how do we add edges, vertices? how to best intuitively create grid?
- can we create grid from existing grid data?
- first, we need an algorithm to propogate voltage from the producers to a consumers 
in the graph... how to do this?
"""

class Node:
    """ need to store things like:
     physical location?
     - consumer, producer, or both? 
     - maybe just units of power consumption/production?
     - but how do we store how this changes day to day, month to month? 
        - link to some external object contained in grid with info? prob not best to store directly in node """
    def __init__(self, nodeId, x, y):
        self.nodeId = nodeId
        self.x = x
        self.y = y
        self.voltage = 0.0


    def __str__(self):
        return f"Node {self.nodeId} at ({self.x}, {self.y})"

    def __repr__(self):
        return f"Node {self.nodeId} at ({self.x}, {self.y})"


class Edge:
    def __init__(self, edgeId: int, node1: Node, node2: Node,
                 capacity: 100.0):
        self.edgeId = edgeId
        self.node1 = node1
        self.node2 = node2
        # automatically compute edge distance from nodes?
        self.distance = np.sqrt((node1.x - node2.x)**2 + (node1.y - node2.y)**2)

        self.capacity = capacity   # in kW
        # flow always defined from node1 to node2
        # how to handle flow in both directions?
        self.flow = node1.voltage - node2.voltage

        # need to calculate power loss in edge
        # need to account for voltage at node1, node2, make flow 



    def __str__(self):
        return f"Edge {self.edgeId} between {self.node1} and {self.node2}"

    def __repr__(self):
        return f"Edge {self.edgeId} between {self.node1} and {self.node2}"

class Grid:
    def __init__(self):
        self.edges = {}
        self.vertices = {}
        self.numNodes = 0

    def addNode(self, x, y):
        node = Node(self.numNodes, x, y)
        self.vertices[self.numNodes] = node
        self.numNodes += 1
        return node

    def addEdge(self, node1, node2):
        edge = Edge(len(self.edges), node1, node2)
        self.edges[(node1.nodeId, node2.nodeId)] = edge
        self.propogateVoltage()   # update voltages of all nodes?
        return edge
    
    def propogateVoltage(self):
        # need to propogate voltage from producers to consumers
        # how to do this?
        # I would start with BFS from both connected nodes... can only update 
        # between the two nodes
        pass

    def __str__(self):
        return f"Grid with {len(self.vertices)} nodes and {len(self.edges)} edges"

    def __repr__(self):
        return f"Grid with {len(self.vertices)} nodes and {len(self.edges)} edges"

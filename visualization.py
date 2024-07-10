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

import networkx as nx
from creation import Grid, Node, Edge


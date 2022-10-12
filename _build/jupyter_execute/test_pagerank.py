#!/usr/bin/env python
# coding: utf-8

# In[1]:


import itertools
import random

import networkx as nx
from matplotlib import pyplot as plt


nodes = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']

# Generate Networkx Graph
G = nx.Graph()
G.add_nodes_from(nodes)

# randomly determine vertices
for (node1, node2) in itertools.combinations(nodes, 2):
    if random.random() < 0.5:
        G.add_edge(node1, node2)
        print(G)

# Draw generated graph
# nx.draw_networkx(G, pos=nx.circular_layout(G), with_labels=True)

print(list(G))
# Compute Page Rank
# pr = nx.pagerank(G, alpha=0.85)

# plt.show()


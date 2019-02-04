#%%
# Here we can use network x to create a -- network. This isn't
# exactly how a neural network works in practice, but it is a 
# great way to create a visualization you can modify in code

#%%
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import math

#%%
# Building a graph with network X, a neural network
# consists of three basic kinds of nodes 
# - inputs
# - activations, these are the connections between all nodes
# - outputs, this is how you tell what your network did

#%%
dense = nx.Graph()
inputs = {i: (0, i) for i in range(0, 5)}
activations = {i+100: (1, i) for i in range(0, 5)}
outputs= {i+1000: (2, i) for i in range(0, 2)}
all = {**inputs, **activations, **outputs}
# and now -- fully connected, every input talks to every 
# activation  -- this is the classic neural network
for input in inputs:
    for activation in activations:
        dense.add_edge(input, activation)
for activation in activations:
    for output in outputs:
        dense.add_edge(activation, output)
nx.draw_networkx_nodes(dense, all, 
    nodelist=all.keys(), node_color='b')
nx.draw_networkx_edges(dense, all, edge_color='w')
axes = plt.axis('off')

#%%
# in practice, these graphs are represented as tensors at each
# layer and are connected via operations, such as a tensor
# product which mathematically connects nodes via
# multiplication and addition
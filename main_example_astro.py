#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Main program for the RDIgraph analyzer

Created on June 18 2018

@author: Jesús Cid Sueiro

# For profiling: kernprof -lv mainRDIgraph.py ...

"""

import pathlib
import matplotlib.pyplot as plt
import numpy as np

# Local imports
from rdigraphs.sgtaskmanager import SgTaskManager

# #######################
# Configurable parameters
# #######################

# Set to True if you want to regenerate the graph data
reset_graph = False
plot_graph = True

# Select project:
path2project = pathlib.Path('..') / 'projects' / f'stars_s02'
path2source = pathlib.Path('..') / 'datasets'
Gw = 'starw'

params = {'select_all': True}

# Graph of windows
sim = 'ncosine'          # Similarity measure
n_epn = 17               # Number of edges per node
cd_algorithm = 'leiden'  # Community detection algorithm
nw_layout = 500           # Number of iterations for the layout

# Graph of signals
th = 0.04   # 0.16             # Threshold for filtering edges
order = 1             # Order of the transductive graph
ns_layout = 10000       # Number of iterations for the layout

# Select the attribute to build the secondary graph
signal_attribute = 'signal'

# #########################
# Launch supergraph project
# #########################

# Open task manager
paths2data = {}
tm = SgTaskManager(path2project, paths2data, path2source)
paths2data = {'graphs': path2project / 'graphs',
              'bigraphs': path2project / 'bigraphs'}

# Load or create project
if not reset_graph and path2project.is_dir():
    tm.load()
    tm.show_SuperGraph()
else:
    tm.create()
    tm.setup()

# ###########################
# Similarity graph of windows
# ###########################

# Get list of the available graphs with saved attributes
graphs = tm.get_graphs_with_features()
# Generate graph of windows (with no edges) if it does not exist
if Gw not in graphs:
    tm.import_snode_from_table(Gw, n0=0, sampling_factor=1, params=params)

# Path to the graph
path2Gw = paths2data['graphs'] / Gw

# Compute similarity graph
if reset_graph or Gw not in graphs:
    tm.blocksize = 10_000
    tm.infer_sim_graph(path2Gw, sim, n0=0, n_epn=n_epn)

# Connected components
if 'cc' not in tm.SG.get_attributes(Gw):
    tm.SG.detectCommunities(Gw, alg='cc', ncmax=None, comm_label='cc', seed=43)

    # Take the largest connected components
    num_cc = 1
    tm.SG.sub_snode_by_threshold(
        Gw, 'cc', num_cc - 1, bound='upper', sampleT=True)

# Detect communities.
if cd_algorithm not in tm.SG.get_attributes(Gw):
    tm.SG.detectCommunities(
        Gw, alg=cd_algorithm, ncmax=None, comm_label=cd_algorithm, seed=43)

# ##########
# Plot graph
# ##########

# Attribute to define the color of the nodes
color_att = cd_algorithm

# Computing graph layout
print(f'Computing layout for graph {Gw} with attribute {color_att}')
if reset_graph:
    tm.SG.graph_layout(Gw, color_att, gravity=1, alg='fa2',
                       num_iterations=nw_layout)
if plot_graph:
    tm.SG.display_graph(
            Gw, color_att, size_att=None, base_node_size=None,
            edge_width=None, show_labels=None, path=None)

# The window graph is complete. Save supergraph.
tm.SG.save_supergraph()

# ##################
# Transductive graph
# ##################

# Bipartite graph windows-signals
# tm.inferBGfromA(path2Gw, signal_attribute, t_label=None, e_label=None)
BGws = "windows_2_signals"  # Name of the bipartite graph
Gs = "signals"              # Name of the secondary graph
tm.SG.snode_from_atts(Gw, signal_attribute, target=Gs, e_label=BGws,
                      save_T=True)

# Transductive graph
tm.SG.transduce(BGws, n=order, normalize=True)

# ############################
# Centrality measures
# ############################

local_metrics = ['centrality', 'degree', 'betweenness', 'pageRank']  # 
                 # 'cluster_coef']
# closeness and katz not included because they are too slow (katz raises
# an error, too).

# Compute local metrics
for metric in local_metrics:
    print(f"-- -- Metric: {metric}")
    tm.SG.local_snode_analysis(Gs, parameter=metric)

# ##########################
# Strongly connected cluster
# ##########################

# The data show that there exist a cluster with strong connections. We will
# detect this cluster by filtering weak edges and extracting the largest CC.

# Remove weak edges, that maybe characteristic of noisy signals
tm.SG.filter_edges_from_snode(Gs, th)

# Take the largest connected components
tm.SG.detectCommunities(Gs, alg='cc', ncmax=None, comm_label='cc')
num_cc = 1
tm.SG.sub_snode_by_threshold(Gs, 'cc', num_cc - 1, bound='upper', sampleT=True)
tm.SG.detectCommunities(Gs, alg=cd_algorithm, ncmax=None,
                        comm_label=cd_algorithm, seed=43)

# Plot transductive graph
# tm.graph_layout(C, attribute)
tm.SG.graph_layout(Gs, cd_algorithm, gravity=1, alg='fa2',
                   num_iterations=ns_layout)
tm.SG.filter_edges_from_snode(Gs, 2*th)
tm.SG.display_graph(Gs, color_att, size_att='pageRank', base_node_size=4000,
                    edge_width=None, show_labels=None, path=None)

tm.SG.save_supergraph()

breakpoint()









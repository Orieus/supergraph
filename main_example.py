#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Main program for the RDIgraph analyzer

Created on June 18 2018

@author: Jesús Cid Sueiro

# For profiling: kernprof -lv mainRDIgraph.py ...

"""

import pathlib
# import matplotlib.pyplot as plt
import numpy as np

# Local imports
from rdigraphs.sgtaskmanager import SgTaskManager

# #######################
# Configurable parameters
# #######################

# Set tu True if you want to regenerate the graph data
reset_graph = True

# Select project:
n = 26
path2project = pathlib.Path('..') / 'projects' / f'cordisAI_{n}topics'
path2source = pathlib.Path('..') / 'datasets'
G = 'Cordis_Kwds3_AI_topics'
params = {'select_all': True, 'n_topics': f'topics{n}'}

# Graph generation
sim = 'BC'
n_epn = 10 if n == 26 else 13
# Community detection
# algorithm = 'louvain'
algorithm = 'leiden'

# Graph of communities
order = 1
# Threshold to sparsify the community graph
th = 0.016 if n == 26 else 0.011

# #################
# Open task manager
# #################

paths2data = {'topicmodels': path2source / 'topic_models',
              'agents': path2source / 'agents',
              'ACL_models': path2source / 'ACL_models'}
tm = SgTaskManager(path2project, paths2data, path2source)

paths2data = {'graphs': path2project / 'graphs',
              'bigraphs': path2project / 'bigraphs',
              'topicmodels': path2source / 'topic_models',
              'agents': path2source / 'agents'}

# ######################
# Load or create project
# ######################

if not reset_graph and path2project.is_dir():
    tm.load()
    tm.show_SuperGraph()
else:
    tm.create()
    tm.setup()

# ############
# Create graph
# ############

# Import nodes and attributes
graphs = tm.get_graphs_with_features()
if G not in graphs:
    tm.import_snode_from_table(
        G, n0=0, sampling_factor=1, params=params)

# Path to the graph
path2G = paths2data['graphs'] / G

# Compute similarity graph
tm.infer_sim_graph(path2G, sim, n0=0, n_epn=n_epn)

# Community detection
tm.detectCommunities('cc', path2G, comm_label=None)
tm.detectCommunities(algorithm, path2G, comm_label=None)

# ##########
# Plot graph
# ##########
attribute = algorithm
tm.graph_layout(G, attribute)
tm.display_graph(G, attribute)

tm.SG.activate_snode(G)
coms = tm.SG.snodes[G].df_nodes.leiden.tolist()
Tmean = tm.SG.snodes[G].T.mean(axis=0)
T2mean = (tm.SG.snodes[G].T**2).mean(axis=0)
Tstd = np.sqrt(T2mean - Tmean**2)
flabels = tm.SG.snodes[G].metadata['feature_labels']

# Plot feature distributions
# plt.figure()
# plt.bar(flabels, Tmean)
# plt.xlabel('Features')
# plt.ylabel('Weight')
# plt.title('Average feature weight')
# plt.xticks(rotation=90)

# plt.figure()
# plt.bar(flabels, Tstd)
# plt.xlabel('Features')
# plt.ylabel('Weight')
# plt.title('Feature weight stds')
# plt.xticks(rotation=90)
# plt.show()

# ###############
# Bipartite graph
# ###############
tm.inferBGfromA(path2G, attribute, t_label=None, e_label=None)

# ##################
# Transductive graph
# ##################
GC = tm.SG.get_sedges()[0]
path2GC = paths2data['bigraphs'] / GC
tm.transduce(path2GC, order)
C = f"{algorithm}_{G}"

path2C = paths2data['graphs'] / C
tm.filter_edges(path2C, th)
tm.detectCommunities(algorithm, path2C, comm_label=None)

# Label nodes
tm.SG.label_nodes_from_features(C, att='tag', thp=2.5)

# #######################
# Plot transductive graph
# #######################
tm.graph_layout(C, attribute)
tm.display_graph(C, attribute)

# ##############################
# Export graphs to parquet files
# ##############################
tm.export_2_parquet(path2G)
tm.export_2_parquet(path2C)

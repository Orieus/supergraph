#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Jesús Cid Sueiro
"""

import pathlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp

# Local imports
from rdigraphs.sgtaskmanager import SgTaskManager

# #########################
# Scope-specific parameters
# #########################

# Select scope:
scope = 'AI'  # 'AI' or 'cancer'

# Dictionary of scope-specific parameters
param_dict = {
    'AI': {'n_topics': 25, 
           'run': 2,
           'ns_G': 0.4,
           'ew_G': 0.04},
    'cancer': {'n_topics': 40,
               'run': 8,
               'ns_G': 0.2,
               'ew_G': 0.02}
    }

# Take variable values from the dictionary
n_topics = param_dict[scope]['n_topics']
run = param_dict[scope]['run']
ns_G = param_dict[scope]['ns_G']  # Node size in the graph
ew_G = param_dict[scope]['ew_G']  # Edge width in the graph

# ################
# Other parameters
# ################

# Name of the folder containing the dataset
table_name = f'cordis_{scope}_test_size_0.0_ntopics_{n_topics}_run_{run}'

# Paths to data
project_name = f'cordis{scope}_{n_topics}topics'
path2project = pathlib.Path('..') / 'projects' / project_name
path2source = pathlib.Path('..') / 'datasets'

#  Set to True to recompute the similarity graph even if it already exists.
reset_graph = True

# Similarity graph
G = f'Cordis{scope}{n_topics}'
params = {'select_all': True, 'n_topics': f'topics{n_topics}'}
sim = 'BC'
n_epn = None     # None to read the value from the metadata
n_iter_layout = 100

# Graph of communities
cd_alg = 'leiden'
# Threshold to sparsify the community graph
th = 0.016 if n_topics == 26 else 0.011

# ######################
# Load or create project
# #######################

paths2data = {}
tm = SgTaskManager(path2project, paths2data, path2source, keep_active=True)
if not reset_graph and path2project.is_dir():
    tm.load()
    tm.show_SuperGraph()
else:
    tm.create()
    tm.setup()

# ################
# Similarity graph
# ################

# Import nodes and attributes
graphs = tm.SG.get_snodes_with_features()
if G not in graphs:
    tm.import_snode_from_npz(table_name, label=G, n0=0, params=params)

# Take n_epn from the metadata of the dataset, if not given
if n_epn is None:
    if 'min_epn_4_1cc' not in tm.DM.metadata:
        raise ValueError("The metadata does not contain the minimum epn for a "
                         "unique CC. Please, set the value of n_epn manually.")
    n_epn = tm.DM.metadata['min_epn_4_1cc'][sim]
    print(f"-- -- Minimum epn for a unique CC: {n_epn}")

# Compute similarity graph
n_nodes = tm.SG.snodes[G].n_nodes
n_edges = int(n_epn * n_nodes)
tm.SG.computeSimGraph(G, n_edges=n_edges, similarity=sim, g=1,
                      blocksize=tm.blocksize, useGPU=tm.useGPU)

# Community detection
tm.SG.detectCommunities(G, alg='cc', comm_label=None)
tm.SG.detectCommunities(G, alg=cd_alg, comm_label=None)

# ##########
# Plot graph
# ##########

# Graph layout
tm.SG.graph_layout(G, gravity=40, alg='fa2', num_iterations=n_iter_layout,
                   attribute=cd_alg)
# Graph display
color_att = cd_alg
att_2_rgb = tm.SG.display_graph(
    G, color_att,size_att=None, base_node_size=ns_G,
    edge_width=ew_G, show_labels=None)

coms = tm.SG.snodes[G].df_nodes.leiden.tolist()
if isinstance(tm.SG.snodes[G].T, sp.csr_matrix):
    Tmean = tm.SG.snodes[G].T.mean(axis=0).A
    T2mean = (tm.SG.snodes[G].T.power(2)).mean(axis=0).A
else:
    Tmean = tm.SG.snodes[G].T.mean(axis=0)
    T2mean = (tm.SG.snodes[G].T**2).mean(axis=0)
Tstd = np.sqrt(T2mean - Tmean**2)
flabels = tm.SG.snodes[G].metadata['feature_labels']

# ##################
# Transductive graph
# ##################

# Bipartite graph
attribute = cd_alg
t_label = f"{attribute}_{G}"
e_label = f"{G}_2_{t_label}"
tm.SG.snode_from_atts(
    G, attribute, target=t_label,e_label=e_label, save_T=True)

# Transductive graph
GC = tm.SG.get_sedges()[0]
tm.SG.transduce(GC)

# Community detection
C = f"{cd_alg}_{G}"
tm.SG.filter_edges_from_snode(C, th)
tm.SG.detectCommunities(C, alg=cd_alg, comm_label=None)

# Label nodes
tm.SG.label_nodes_from_features(C, att='tag', thp=2.5)

# Plot transductive graph
tm.SG.graph_layout(
    C, gravity=40, alg='fr', num_iterations=100, attribute=attribute)
color_att = attribute
att_2_rgb2 = tm.SG.display_graph(
    C, color_att, size_att=None, base_node_size=300, edge_width=1,
    show_labels=None)


# ##############################
# Export graphs to parquet files
# ##############################
# path2G = tm.SG.path2snodes / G
# path2C = tm.SG.path2snodes / C
# tm.export_2_parquet(path2G)
# tm.export_2_parquet(path2C)

# #######################
# Analyze EU projects
# #######################

# Get community labels from the snodes of the community graph
tm.SG.activate_snode(C)
comm_names = tm.SG.snodes[C].df_nodes.tag.tolist()
# Some name changes to reduce the length of the labels
comm_names = [name.replace(' and ', ' & ') for name in comm_names]
comm_names = [name.replace(',', ' ,\n ') for name in comm_names]
comm_names = [name.replace('Natural Language Processing', 'NLP')
              for name in comm_names]
comm_names = [name.replace('Environmental', 'Environm.')
              for name in comm_names]
comm_names = [name.replace('Reinforcement', 'Reinf.') for name in comm_names]
comm_names = [name.replace('Assessment', 'Assessm.') for name in comm_names]
comm_names = [name.replace('Neural Networks', 'NN') for name in comm_names]

# Compute a dictionary with the frequency of each community
comm_dict = dict(zip(tm.SG.snodes[C].df_nodes.Id.tolist(), comm_names))
eu_labels = [comm_dict[str(i)] for i in coms]
freq = {name: 0 for name in comm_names}
for label in eu_labels:
    freq[label] += 1
print(freq)

# Sort dictionary by decreasing values
freqs = list(freq.values())
colors = att_2_rgb.values()
sorted_tuples = sorted(zip(freqs, comm_names, colors), reverse=True)
sorted_freqs, sorted_comm_names, sorted_colors = zip(*sorted_tuples)

# Plot a pie chart with the frequency of each community
plt.figure(figsize=(10, 6))
plt.pie(sorted_freqs, labels=sorted_comm_names, autopct='%1.1f%%',
        colors=sorted_colors)
plt.title('Projects by community')
plt.show(block=False)

# Save figure in the following path
tm.SG.activate_snode(G)
path = tm.SG.snodes[G].path2graph / 'eu_coms.png'
plt.savefig(path)

"""
# #######################
# Analyze spanish projects
# #######################

# Detect Spanish projects in graph G.
isES = (tm.SG.snodes[G].df_nodes['coordinatorCountry'] == 'ES').tolist()
tm.SG.add_snode_attributes(G, 'is_spanish', isES)

# Display the graph highlighting Spanish projects
path = tm.SG.snodes[G].path2graph / 'is_spanish.png'
tm.display_graph(G, 'is_spanish', path=path)

# Get communities of Spanish projects
tm.SG.activate_snode(G)
coms = tm.SG.snodes[G].df_nodes.leiden.tolist()
isES = tm.SG.snodes[G].df_nodes.is_spanish.tolist()
comsES = [coms[i] for i in range(len(coms)) if isES[i]]

# Get community labels from the snodes of the community graph
tm.SG.activate_snode(C)

# Compute a dictionary with the frequency of each community
# in es_labels
es_labels = [comm_dict[str(i)] for i in comsES]
freq = {name: 0 for name in comm_names}
for label in es_labels:
    freq[label] += 1
print(freq)

# Sort dictionary by decreasing values
freqs = list(freq.values())
sorted_tuples = sorted(zip(freqs, comm_names, colors), reverse=True)
sorted_freqs, sorted_comm_names, sorted_colors = zip(*sorted_tuples)

# Plot a pie chart with the frequency of each community
plt.figure(figsize=(10, 6))
plt.pie(sorted_freqs, labels=sorted_comm_names, autopct='%1.1f%%',
        colors=sorted_colors)
plt.title('Spanish projects by community')
plt.show(block=False)

# Save figure in the following path
path = tm.SG.snodes[G].path2graph / 'es_coms.png'
plt.savefig(path)

# ##############################
# Evolution of communities
# ##############################

# Select columns of starting dates and communities
tm.SG.activate_snode(G)
# df_cy = tm.SG.snodes[G].df_nodes.loc[:, ['leiden', 'startDate', 'totalCost']]
df_cy = tm.SG.snodes[G].df_nodes.loc[:, ['leiden', 'startDate',
                                         'ecMaxContribution']]

# Keep rows without nan values only
ind = [i for i, x in enumerate(df_cy['startDate']) if isinstance(x, str)]
df_cy = df_cy.iloc[ind]

# Column 'startDate' contains dates as strings. Take first 4 characters (year)
comms = df_cy.leiden.tolist()
years = df_cy.startDate.tolist()
years = [x[:4] for x in years]
# budgets = df_cy.totalCost.tolist()
budgets = df_cy.ecMaxContribution.tolist()

# Maping years to indices
period = sorted(list(set(years)))
year_2_ind = dict(zip(period, list(range(len(period)))))

# Compute array communities times years
num_comms = len(set(comms))
num_years = len(period)
X = np.zeros((num_comms, num_years))
Y = np.zeros((num_comms, num_years))
for i in range(len(comms)):
    X[comms[i], year_2_ind[years[i]]] += 1
    bi = float(budgets[i].replace(',', '.')) / 1_000_000
    Y[comms[i], year_2_ind[years[i]]] += bi

# Save X and Y into xlsx files
dfX = pd.DataFrame(X, columns=period, index=comm_names)
dfY = pd.DataFrame(Y, columns=period, index=comm_names)
dfX.to_excel(tm.SG.snodes[G].path2graph / 'evolution.xlsx')
dfY.to_excel(tm.SG.snodes[G].path2graph / 'funding_evolution.xlsx')

# Make a linear plot of the evolution of communities
plt.figure(figsize=(10, 6))
plt.stackplot(period, X, labels=comm_names, alpha=0.8, colors=colors)
plt.xlabel('Years')
plt.ylabel('Number of projects')
plt.title('Evolution of communities')
# place the legend outside the plot
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
# Make more space on the right side to fit the legend
plt.subplots_adjust(right=0.55)
# Rotate xticks
plt.xticks(rotation=90)
# Save figure in a file named 'evolution.png'
path = tm.SG.snodes[G].path2graph / 'evolution.png'
plt.savefig(path)
plt.show(block=False)

# Make a linear plot of the evolution of communities
plt.figure(figsize=(10, 6))
plt.stackplot(period, Y, labels=comm_names, alpha=0.8, colors=colors)
plt.xlabel('Years')
plt.ylabel('Budget (mill. €)')
plt.title('Evolution of communities')

# place the legend outside the plot
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
# Make more space on the right side to fit the legend
plt.subplots_adjust(right=0.55)
# Rotate xticks
plt.xticks(rotation=90)
# Save figure in a file named 'evolution.png'
path = tm.SG.snodes[G].path2graph / 'funding_evolution.png'
plt.savefig(path)
plt.show(block=False)

"""

# ############################
# Centrality measures
# ############################

local_metrics = ['centrality', 'degree', 'betweenness', 'closeness',
                 'pageRank', 'cluster_coef']
# katz not included because it raises an error.

# Compute local metrics
for metric in local_metrics:
    print(f"-- -- Metric: {metric}")
    tm.SG.local_snode_analysis(G, parameter=metric)

tm.SG.save_supergraph()






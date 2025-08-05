#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""
A supergraph for the analysis of data from the Newton telescope.
Created on Oct, 15, 2024
@author: Jesús Cid
"""

# Imports
import pathlib
# Local imports
from astro_lib.astro import Astro

# #######################
# Configurable parameters
# #######################
# Set to True if you want to delete any previous graph and start from scratch
# Set to False to use the existing graphs or layouts.
reset_window_graph = False
reset_window_layout = False
reset_window_display = True
reset_signal_graph = False
reset_signal_layout = False

# Source data folder AND name of the graph of windows
# Gw = 'stars'
# Gw = 'starsVanillaXsd015Z05'  
# Gw = 'starsVanillaXsd015Z40'  
# Gw = 'starwVanillaZ05'  
# Gw = 'starwVanillaZ10'
# Gw = 'starwVanillaZ20'
# Gw = 'starwVanillaZ40'
# Gw = 'star_time_z_5'  # Name of the graph of windows
Gw = 'stars2025-07-17-162801'

# Name of the graph project.
project_name = 'stars20250717gauss'   # f'{Gw}_80'

# Names of the graphs
Gs = "sigmax"            # Name of the graph of signals
BGws = f"{Gw}_2_{Gs}"    # Name of the bipartite graph windows-signals

# This is to take all attributes from the input data
params = {'select_all': True}

# ################
# Graph of windows
sim = 'Gauss'          # Similarity measure
# sim = 'ncosine'

# Number of edges per node
# If None, the epn is set as the minimum number to get a unique connected
# component. This value is read from the metadata of the dataset.
n_epn = 140             # Number of edges per node
cd_algorithm = 'leiden'  # Community detection algorithm
# color_att_w = 'leiden'   # Attribute to define the color of the nodes
color_att_w = 'Label'   # Attribute to define the color of the nodes
color_att_w = 'Type'   # Attribute to define the color of the nodes
nw_layout = 300          # Number of iterations for the layout (~200 is fine)
edge_ratio_w = 0.2      # Ratio of edges to display from the graph of windows
# No. of connected components (CC)
#   if n, only nodes from the n largest CCs are kept: 
#   1 means all CCs are kept (the largest index is the largest CC)
ncc_w = 1
base_node_size_w = None
size_att_w = {'Label': {'Interesting': 2000,
                        'NOT interesting': 1000,
                        'n.a.': 2000}}
size_att_w = {'Type': {x: 2000 for x in [
    'n.a.', 'bump', 'background', 'increase', 'decrease', 'unknown',
    'doubleExponential', 'modulation', 'periodic']}}
# size_att_w = {'Subtype': {x: 2000 for x in [
#     'n.a.', 'highEnergies', 'modulation', 'midEnergies', 'increase', 'flare',
#     'withBackground', 'cataclismic']}}

# ################
# Graph of signals

# Dictionary of cases
case = {"sigsum": {"s_metric": "sum_sum", "th": 0.02,},
        "sigmax": {"s_metric": "max_max", "th": 0.85},
        "sigmap": {"s_metric": "lin_sum", "th": 0.05}}
s_metric = case[Gs]['s_metric']  # Similarity metric 
th = case[Gs]['th']              # Threshold for filtering edges
order = 1                        # Order of the transductive graph
ns_layout = 500                  # Number of iterations for the layout
# No. of connected components (CC)
#     if n, only nodes from the n largest CCs are kept: 
#     1 means all CCs are kept, the largest index is the largest CC)
ncc_s = 1
edge_ratio_s = 0.1  # Ratio of edges to display from the graph of signals

# Name of the attribute in the dataset that identifies the signal
signal_attribute = 'signal'
color_att_s = 'leiden'  # Attribute to define the color of the nodes
size_att_s = 'pageRank'

# ##########
# Main paths

# This should not be modified if you follow the "standard" file structure.
# Path fo the project folder
path2project = pathlib.Path('..') / 'projects' / project_name
# Path to the source data folder.
path2source = pathlib.Path('..') / 'datasets'

# #########################
# Launch supergraph project
# #########################

# Print a starting message in big letters
print("\n\n\n\n\n")
print("#############################################")
print("##                                         ##")
print("##     SUPERGRAPH FOR ASTROPHYSICS         ##")
print("##                                         ##")
print("#############################################")
print("\n")


# Create a new Astro object to manage the graph generation
starship = Astro(path2project, path2source,
                 reset_window_graph=reset_window_graph)

# ###########################
# Similarity graph of windows
# ###########################

# Generate graph of windows (with no edges) if it does not exist
if reset_window_graph or Gw not in starship.list_of_graphs:
    starship.compute_window_simgraph(
        Gw, params=params, n_epn=n_epn, sim=sim, ncc_w=ncc_w,
        cd_algorithm=cd_algorithm, take_largest_cc=True)
starship.tm.SG.describe(Gw)

# Display the graph of windows
if reset_window_layout:
    starship.layout_window_graph(
        Gw, nw_layout=nw_layout, save_gexf=False, color_att_w=color_att_w)

    # Saving here is safe, as computing the graph and the layout takes time...
    starship.save_supergraph()

if reset_window_display:
    att_2_rgb = starship.display_window_graph(
        Gw, color_att_w=color_att_w, edge_ratio=edge_ratio_w,
        base_node_size=base_node_size_w, size_att=size_att_w)
    print(att_2_rgb)

# The graph of windows is ready. Save supergraph.
starship.save_supergraph()

# #####################
# Graph of trajectories
# #####################

# Duplicate the graph of windows to create a graph of trajectories, chaining
# the nodes of the same star. 
Gw2 = 'trajectories'
if Gw2 not in starship.list_of_graphs:
    print(f"Creating graph of trajectories {Gw2} from {Gw}")
    starship.compute_trajectory_graph(Gw, Gw2, color_att_w=color_att_w)

# #####################################
# Graph of signals (transductive graph)
# #####################################

if reset_signal_graph or Gs not in starship.list_of_graphs:

    # Compute the graph of signals
    starship.compute_signal_graph(
        Gw, Gs, BGws, signal_attribute=signal_attribute, s_metric=s_metric,
        order=order)
    
    # Compute centrality measures
    starship.centralities(Gs)
    # Identify communities of highly connected signals
    starship.communities(Gs, cd_algorithm=cd_algorithm)
    # This takes a lot of computation, save the super graph.
    starship.save_supergraph()

# Display signal graph
if reset_signal_graph or reset_signal_layout:
    att_2_rgb_s = starship.display_signal_graph(
        Gs, color_att_s, size_att=size_att_s, n_iter=ns_layout,
        save_gexf=False, edge_ratio=edge_ratio_s)

# Analyze types of signals
if color_att_s == 'main_type':
    starship.analyze_types_of_signals(Gs, att_2_rgb_s)

# This takes a lot of computation, save the super graph.
starship.save_supergraph()

breakpoint()

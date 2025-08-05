# Importe the sgtask manager
from rdigraphs.sgtaskmanager import SgTaskManager

import matplotlib.pyplot as plt


def calculate_sum(a, b):
    """Calculate the sum of two numbers."""
    return a + b

class Astro:
    """
    A class to manage the analysis of astronomical data using a supergraph.
    """

    def __init__(self, path2project, path2source, reset_window_graph=True):
        """
        Initialize the Astro object.

        Parameters:
        -----------
        path2project : pathlib.Path
            Path to the project directory.
        path2source : pathlib.Path
            Path to the source data directory.
        reset_window_graph : bool, optional
            If True, the window graph will be reset and recomputed.
            Default is True.
        """

        # Open task manager.
        paths2data = {}
        tm = SgTaskManager(path2project, paths2data, path2source,
                           keep_active=True)

        self.tm = tm
        self.reset_window_graph = reset_window_graph
        self.path2project = path2project

        # Load or create project
        if not self.reset_window_graph and self.path2project.is_dir():
            self.tm.load()
            self.tm.SG.describe()     # This shows a summary of the current supergraph
        else:
            self.tm.create()
            self.tm.setup()

        return

    @property
    def list_of_graphs(self):
        """Return the list of graphs available in the supergraph."""
        return self.tm.SG.get_snodes()
        
    def compute_window_simgraph(
            self, Gw, params={}, n_epn=None, sim='ncosine', ncc_w=1,
            cd_algorithm='leiden', take_largest_cc=True):

        """ 
        Compute the similarity graph of windows.

        Parameters:
        -----------
        Gw : str
            Name of the graph to compute.
        params : dict
            Parameters for the similarity computation.
        n_epn : int, optional
            Number of edges per node in the similarity graph. If not provided,
            it will be taken from the metadata of the dataset.
        sim : str, optional
            Similarity measure to use. Default is 'ncosine'.
        ncc_w : int, optional
            Number of connected components to keep in the graph. Default is 1.
        cd_algorithm : str, optional
            Community detection algorithm to use. Default is 'leiden'.
        take_largest_cc : bool, optional
            If True, only the largest connected components will be kept in the
            graph. Default is True.
        """

        print("\n#############################################")
        print("##     Graph of windows")

        self.Gw = Gw

        self.tm.import_snode_from_table(
            Gw, n0=0, sampling_factor=1, params=params)

        # Take n_epn from the metadata of the dataset, if not given
        if n_epn is None:
            if 'min_epn_4_1cc' not in self.tm.DM.metadata:
                raise ValueError(
                    "The metadata does not contain the minimum epn for a "
                    "unique CC. Please, set the value of n_epn manually.")
            n_epn = self.tm.DM.metadata['min_epn_4_1cc'][sim]
            print(f"-- -- Minimum epn for a unique CC: {n_epn}")

        # Compute similarity graph
        n_edges = int(n_epn * self.tm.SG.snodes[Gw].n_nodes)  # Target #edges
        self.tm.SG.computeSimGraph(
            Gw, n_edges=n_edges, similarity=sim, g=1, blocksize=10_000,
            useGPU=False, verbose=False)
        
        # Compute connected components
        self.tm.SG.detectCommunities(Gw, alg='cc', ncmax=None, comm_label='cc')

        # Take the largest connected components
        if take_largest_cc:
            self.tm.SG.sub_snode_by_threshold(Gw, 'cc', ncc_w - 1,
                                              bound='upper', sampleT=True)

        # Detect communities.
        if cd_algorithm not in self.tm.SG.get_attributes(Gw):
            self.tm.SG.detectCommunities(
                Gw, alg=cd_algorithm, ncmax=None, comm_label=cd_algorithm,
                seed=43)

        return

    def layout_window_graph(self, Gw, nw_layout=300, save_gexf=True,
                            color_att_w=None):
        """
        Computes the graph layout in a 2D space.

        Parameters
        ----------
        Gw : str
            Name of the graph of windows to display.
        nw_layout : int, optional
            Number of iterations for the layout computation. Default is 300.
        save_gexf : bool, optional
            If True, the graph will be saved in GEXF format. Default is True.
        color_att_w : str, optional
            Attribute to define the color of the nodes. Only used if save_gexf
            is True. Default is None
        """

        # Compute graph layout
        print(f'Computing layout for graph {Gw}')
        self.tm.SG.graph_layout(
            Gw, gravity=1, alg='fa2', num_iterations=nw_layout,
            save_gexf=save_gexf, attribute=color_att_w)

        return

    def display_window_graph(self, Gw, color_att_w='', size_att=None,
                            base_node_size=None, edge_ratio=None):
        """
        Display the graph of windows.

        Parameters
        ----------
        Gw : str
            Name of the graph of windows to display.
        color_att_w : str, optional
            Attribute to define the color of the nodes. Default is an empty string.
        edge_ratio : float, optional
            If not None, the number of edges shown in the graph is this portion
            of the total number of edges. Only the edges with higher weights
            are kept. This is useful to display large graphs. If None, no
            reduction is applied. 
        """

        # Compute graph layout
        print(f'Displaying graph {Gw} with attribute {color_att_w}')

        # Display the graph into a png file
        if 'x' in self.tm.SG.get_attributes(Gw):
            att_2_rgb = self.tm.SG.display_graph(
                Gw, color_att_w, size_att=size_att,
                base_node_size=base_node_size, edge_width=0.01,
                edge_ratio=edge_ratio)
        else:
            att_2_rgb = None

        return att_2_rgb

    def compute_trajectory_graph(self, Gw, Gw2, color_att_w=''):
        """
        Compute the graph of trajectories from the window graph.
        
        Parameters:
        -----------
        Gw : str
            Name of the graph of windows.
        Gw2 : str
            Name of the graph of trajectories to be created.
        """

        print("\n#############################################")
        print("##     Graph of trajectories")

        # Generate the graph of trajectories
        self.tm.SG.duplicate_snode(Gw, Gw2)
        self.tm.SG.snodes[Gw2].remove_all_edges()
        self.tm.SG.chain_by_attributes(Gw2, 'signal', '# window')

        # Display the graph of trajectories in a png file
        if 'x' in self.tm.SG.get_attributes(Gw2):
            _ = self.tm.SG.display_graph(
                Gw2, color_att_w, size_att=None, base_node_size=None,
                edge_width=0.01)

        return 

    def compute_signal_graph(self, Gw, Gs, BGws, signal_attribute,
                             s_metric, order=1):
        """
        Compute the graph of signals from the window graph.

        Parameters
        ----------
        Gw : str
            Name of the graph of windows.
        Gs : str
            Name of the graph of signals to be created.
        BGws : str
            Name of the bipartite graph between windows and signals.
        signal_attribute : str
            Attribute from Gw that identifies the signals.
        s_metric : str
            Similarity metric to use for the signal graph.
        order : int, optional
            Order of the transductive graph. Default is 1.
        """

        print("\n#############################################")
        print("##     Graph of signals")

        # Create the bipartite graph windows --> signals,
        # and the graph of signals
        self.tm.SG.snode_from_atts(
            Gw, signal_attribute, target=Gs, e_label=BGws, save_T=False)

        # Transductive graph
        self.tm.SG.transduce(BGws, n=order, normalize=True, method=s_metric)

        # Add signal attribute to the nodes of the transductive graph
        self.tm.SG.activate_snode(Gw)
        df_types = self.tm.SG.snodes[Gw].df_nodes.loc[:, ['signal', 'main_type']]
        # Remove repeated rows in the dataframe
        df_types = df_types.drop_duplicates()

        self.tm.SG.add_snode_attributes(Gs, 'signal', df_types,
                                        fill_value='unknown')
        
        return

    def centralities(self, Gs):
        """
        Compute centrality measures for the graph of signals.

        Parameters:
        -----------
        Gs : str
            Name of the graph of signals.
        """

        # Compute centrality measures
        local_metrics = ['centrality', 'degree', 'betweenness', 'pageRank']
        for metric in local_metrics:
            print(f"-- -- Metric: {metric}")
            try:
                self.tm.SG.local_snode_analysis(Gs, parameter=metric)
            except Exception as e:
                print(f"Error computing {metric}: {e}")

        return

    def communities(self, G, cd_algorithm='leiden'):
        """
        Detect communities in the graph. This can be used for any graph
        (signals, windows, etc.).

        Parameters:
        -----------
        G : str
            Name of the graph.
        cd_algorithm : str, optional
            Community detection algorithm to use. Default is 'leiden'.
        """

        # To remove weak edges, that maybe characteristic of noisy signals
        # self.tm.SG.filter_edges_from_snode(G, th)
        # To take the largest connected components
        # self.tm.SG.detectCommunities(G, alg='cc', ncmax=None, comm_label='cc')
        # self.tm.SG.sub_snode_by_threshold(G, 'cc', ncc_s-1, bound='upper',
        #                                   sampleT=True)

        # Detect communities in the graph
        self.tm.SG.detectCommunities(
            G, alg=cd_algorithm, ncmax=None, comm_label=cd_algorithm, seed=43)

        return

    def display_signal_graph(self, Gs, color_att, size_att=None, n_iter=100,
                             save_gexf=True, edge_ratio=None):
        """
        Display the graph of signals.

        Parameters
        ----------
        Gs : str
            Name of the graph of signals to display.
        color_att : str
            Attribute to define the color of the nodes.
        size_att : str, optional
            Attribute to define the size of the nodes. Default is None.
        n_iter : int, optional
            Number of iterations for the layout computation. Default is 100
        save_gexf : bool, optional
            If True, the graph will be saved in GEXF format. Default is True
        edge_ratio : float, optional
            If not None, the number of edges shown in the graph is this portion
            of the total number of edges. Only the edges with higher weights
            are kept. This is useful to display large graphs. If None, no
            reduction is applied.
        """

        # Compute node coordinates
        self.tm.SG.graph_layout(
            Gs, gravity=1, alg='fa2', num_iterations=n_iter,
            save_gexf=save_gexf, attribute=color_att)

        # To plot the graph of signals in a png file
        att_2_rgb = self.tm.SG.display_graph(
            Gs, color_att, size_att=size_att, base_node_size=40000,
            edge_width=0.01, show_labels=None, path=None, edge_ratio=edge_ratio)
        
        return att_2_rgb

    def analyze_types_of_signals(self, Gs, att_2_rgb):
        """Analyze the types of signals in the graph.

        Parameters
        ----------
        Gs : str
            Name of the graph of signals.
        att_2_rgb : dict
            Dictionary mapping attributes to RGB colors.
        """

        # Get community labels from the snodes of the community graph
        self.tm.SG.activate_snode(Gs)
        unique_comm_names = list(att_2_rgb.keys())
        comm_names = tm.SG.snodes[Gs].df_nodes.main_type.tolist()
        # Compute a dictionary with the frequency of each community
        comm_dict = dict(zip(tm.SG.snodes[Gs].df_nodes.Id.tolist(), comm_names))

        freq = {name: 0 for name in unique_comm_names}
        for label in comm_names:
            freq[label] += 1
        print(freq)

        # Sort dictionary by decreasing values
        freqs = list(freq.values())
        colors = att_2_rgb.values()
        sorted_tuples = sorted(zip(freqs, unique_comm_names, colors), 
                               reverse=True)
        sorted_freqs, sorted_comm_names, sorted_colors = zip(*sorted_tuples)

        # Plot a pie chart with the frequency of each community
        plt.figure(figsize=(10, 6))
        plt.pie(sorted_freqs, labels=sorted_comm_names, autopct='%1.1f%%',
                colors=sorted_colors)
        plt.title('Star types by community')
        plt.show(block=False)

        # Save figure in the following path
        self.tm.SG.activate_snode(Gs)
        path = self.tm.SG.snodes[Gs].path2graph / 'pie_chart.png'
        plt.savefig(path)

    def save_supergraph(self):
        """
        Save the current supergraph to the project directory.
        """

        self.tm.SG.save_supergraph()

        return

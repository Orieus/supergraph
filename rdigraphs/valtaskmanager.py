import os
# import shutil
# import _pickle as pickle

# import yaml

import numpy as np
import scipy.sparse as scsp
import pandas as pd
# from scipy.sparse import csr_matrix
# import colored
import time
import logging

# # import configparser
from collections import Counter
from copy import copy

# Set matplotib parameters to allow remote executions
# import matplotlib
# matplotlib.use("Agg")

# Imports for the community label extraction
# Maybe this does not need to be after matplotlib.use()
import gensim.corpora.bleicorpus as blei
from sklearn.preprocessing import normalize

# # Local imports
from rdigraphs.sgtaskmanager import SgTaskManager
from rdigraphs.sim_graph.sim_graph import SimGraph
from rdigraphs.community_plus.community_plus import CommunityPlus

import matplotlib.pyplot as plt


class ValTaskManager(SgTaskManager):
    """
    Task Manager for the RDIgraph analyzer.

    The behavior of this class depends on the state of the project, in
    dictionary self.state, with the followin entries:

    - 'isProject'   : If True, project created. Metadata variables loaded
    - 'configReady' : If True, config file succesfully loaded. Datamanager
                      activated.
    """

    def __init__(self, path2project, paths2data):
        """
        Initializes the LabTaskManager object

        Parameters
        ----------
        path2project : str
            Path to the graph processing project
        paths2data : dict
            Paths to data sources
        """

        super().__init__(path2project, paths2data)

        return

    def get_ref_graphs(self, db):
        """
        Returns the availabel refrence graphs for the given database
        """

        path2graphs = os.path.join(self.path2project, self.f_struct['snodes'])
        corpus_label = {'Co': 'Crunch4Ever', 'Pu': 'S24Ever'}[db]
        graphs = [x for x in os.listdir(path2graphs)
                  if corpus_label in x]

        return graphs

    def compute_reference_graph(self, db):
        """
        Computes a reference graph for the topic models

        Parameters
        ----------
        db : str {Pu, Co}
            Data source.
        """

        # Configurable parameters:
        n_nodes_db = 100_000        # Size of the initial set of nodes
        n_nodes_target = 20_000     # Target no. of nodes
        epn = 100                   # Target number of edges per node
        logging.info(f'-- --  {epn} edges per node')

        # Parameters specific of the selected corpus:
        corpus_label = {'Co': 'Crunch4Ever', 'Pu': 'S24Ever'}[db]
        # The column in the database containing the node identifiers
        ref_col = {'Co': 'companyID', 'Pu': 'paperID'}[db]

        # Path to topic models
        logging.info('-- Reading nodes from the topic model folders')
        # Path to the file containing the metadata, including the node names,
        # which is common to all models
        path2nodenames = (f'/home/jarenas/github/TM4Ever/TM4Ever20200221/'
                          f'corpus/{corpus_label}/{corpus_label}_metadata.csv')
        path2models = '/home/jarenas/github/TM4Ever/TM4Ever20200221/models/tm'

        # Get an arbitrary topic model to get the list of nodes
        folder_names = os.listdir(path2models)
        folder_names = [x for x in folder_names if x[:4] == corpus_label[:4]]
        fpath = os.path.join(path2models, folder_names[0])
        df_nodes_tm = self.DM.readCoordsFromFile(
            fpath=fpath, sparse=True, path2nodenames=path2nodenames,
            ref_col=ref_col)[1]
        nodes_tm = df_nodes_tm[ref_col].tolist()
        logging.info(f'-- -- {len(nodes_tm)} nodes in the topic models')

        # Read db table
        if db == 'Co':

            # #############################
            # REFERENCE GRAPH FOR COMPANIES
            # #############################

            # Load all nodes and ist category attribute from database
            logging.info('-- Reading nodes from database')
            df = self.DM.SQL['Co'].readDBtable(
                'CompanyCat', limit=None, selectOptions=None,
                filterOptions=None, orderOptions=None)
            # df = self.DM.SQL['Co'].readDBtable(
            #     'CompanyGroupCat', limit=None, selectOptions=None,
            #     filterOptions=None, orderOptions=None)
            att = 'categoryID'

            # Original dataset
            nodes_db = sorted(list(set(df['companyID'])))
            n_nodes_db = len(nodes_db)
            # Inverse index of nodes
            node_db2ind = dict(zip(nodes_db, range(n_nodes_db)))
            logging.info(f'-- -- Dataset loaded with {n_nodes_db} nodes and ' +
                         f'{len(df)} attribute assignments')

            row_inds = [node_db2ind[x] for x in df['companyID']]  # .tolist()]
            col_inds = df[att].tolist()
            data = [1] * len(row_inds)
            T = scsp.csr_matrix((data, (row_inds, col_inds)))
            T = scsp.csr_matrix(T / np.sum(T, axis=1))

            # ########################
            # Compute similarity graph

            # Create datagraph with the full feature matrix
            graph_name = f'{corpus_label}_ref'
            self.SG.makeSuperNode(label=graph_name, nodes=nodes_db, T=T)
            # Take nodes that are in the topic models only
            self.SG.sub_snode(graph_name, nodes_tm, sampleT=True)
            # Subsample graph for the validation analysis
            breakpoint()
            n_gnodes = n_nodes_db // 10
            self.SG.sub_snode(graph_name, n_gnodes, sampleT=True)

            # Compute similarity graph based on attributes
            n_edges = epn * n_nodes_db
            self.SG.computeSimGraph(
                graph_name, n_edges=n_edges, n_gnodes=n_gnodes,
                similarity='He2', g=1, rescale=self.rescale,
                blocksize=self.blocksize, useGPU=self.useGPU, tmp_folder=None,
                save_every=20_000_000, verbose=False)

        elif db == 'Pu':

            # #############################
            # REFERENCE GRAPH FOR COMPANIES
            # #############################

            logging.info('-- Reading nodes from database')
            # Table: S2papers
            # ---- Attributes: paperID, S2paperID, title, lowertitle,
            #      paperAbstract, entities, fieldsOfStudy, s2PdfUrl, pdfUrls,
            #      year, journalVolume, journalPages, isDBLP, isMedline, doi,
            #      doiUrl, pmid, langid, LEMAS
            # ---- No. of rows: 13126503
            # Table: citations
            # ---- Attributes: citationID, S2paperID1, S2paperID2
            # ---- No. of rows: 86260982
            df_nodes_db = self.DM.SQL['Pu'].readDBtable(
                'S2papers', limit=None, selectOptions='paperID, S2paperID',
                filterOptions=None, orderOptions=None)
            logging.info(f'-- -- {len(df_nodes_db)} nodes in the database')

            df_nodes_db = df_nodes_db[df_nodes_db['paperID'].isin(nodes_tm)]
            df_nodes_db = df_nodes_db.sample(n=n_nodes_db, random_state=3)
            logging.info(f'-- -- Random sample of {n_nodes_db} nodes selected')

            # Read edges from database
            logging.info('-- Reading citation data from database')
            df_edges = self.DM.SQL['Pu'].readDBtable(
                'citations', limit=None, filterOptions=None,
                orderOptions=None)
            logging.info(f'-- -- {len(df_edges)} edges in the database')

            # Select edges with a citing paper in the list of nodes only.
            nodes_S2 = df_nodes_db['S2paperID'].tolist()
            df_edges = df_edges[df_edges['S2paperID1'].isin(nodes_S2)]
            edges_orig = df_edges['S2paperID1'].tolist()
            edges_dest = df_edges['S2paperID2'].tolist()
            n_edges = len(edges_orig)
            logging.info(f'-- -- {n_edges} edges from the selected nodes')

            # and, viceversa: select nodes with a cite in df_edges.
            # This is the definite list of nodes for the large reference graph
            df_nodes_db = (
                df_nodes_db[df_nodes_db['S2paperID'].isin(edges_orig)])
            nodes = df_nodes_db['paperID'].tolist()
            # This is the list of S2 ids used for the edges
            nodes_S2 = df_nodes_db['S2paperID'].tolist()
            n_nodes = len(nodes)
            logging.info(
                f'-- -- Selecting {n_nodes} nodes with at least one reference')

            # Get list of unique cited papers
            atts = list(set(df_edges['S2paperID2']))
            n_atts = len(atts)

            # Inverse index of nodes
            nodeS2_2ind = dict(zip(nodes_S2, range(n_nodes)))
            att2ind = dict(zip(atts, range(n_atts)))
            logging.info(
                f'-- Dataset loaded with {n_nodes} nodes, {n_atts} ' +
                f'attributes and {n_edges} attribute assignments')

            row_inds = [nodeS2_2ind[x] for x in edges_orig]  # .tolist()]
            col_inds = [att2ind[x] for x in edges_dest]  # .tolist()]

            # Compute normalized feature matrix
            data = [1] * len(row_inds)
            T = scsp.csr_matrix((data, (row_inds, col_inds)))
            sumT = scsp.diags(1 / T.sum(axis=1).A.ravel())
            T = scsp.csr_matrix(sumT * T)

            # ########################
            # Compute similarity graph

            # Create datagraph with the full feature matrix
            graph_name = f'{corpus_label}_ref'
            self.SG.makeSuperNode(label=graph_name, nodes=nodes, T=T)
            # self.SG.sub_snode(graph_name, n_nodes_target, ylabel=graph_name,
            #                   sampleT=True)

            n_edges = epn * n_nodes
            self.SG.computeSimGraph(
                graph_name, n_edges=n_edges, n_gnodes=None, similarity='He2',
                g=1, rescale=self.rescale, blocksize=self.blocksize,
                useGPU=self.useGPU, tmp_folder=None, save_every=20_000_000,
                verbose=False)

        else:
            logging.error('-- Unknown db')

        # #####################
        # SHOW AND SAVE RESULTS

        # Log some results
        # md = dg.metadata['graph']
        md = self.SG.snodes[graph_name].metadata
        logging.info(f"-- -- Number of nodes: {md['nodes']['n_nodes']}")
        logging.info(f"-- -- Number of edges: {md['edges']['n_edges']}")
        logging.info(f"-- -- Average neighbors per node: " +
                     f"{md['edges']['neighbors_per_sampled_node']}")
        logging.info(f"-- -- Density of the similarity graph: " +
                     f"{100 * md['edges']['density']} %")

        self.SG.detectCommunities(graph_name, alg='cc', comm_label='CC')

        cc = self.SG.snodes[graph_name].df_nodes['CC'].tolist()
        rr = Counter(cc)
        logging.info(f"-- Largest connected component with {rr[0]} nodes")
        ylabel = f'{corpus_label}_ref_{n_nodes_target}'
        self.SG.sub_snode_by_value(graph_name, 'CC', 0)

        self.SG.sub_snode(graph_name, n_nodes_target, ylabel=ylabel)

        # Save graph: nodes and edges
        self.SG.save_supergraph()

        # Reset snode. This is to save memory.
        self._deactivate()

        return

    def _validate_topic_models(self, path2models, path2nodenames, corpus,
                               label_RG, spf, radius=None, epn=None,
                               cd_algs=[], metrics=[], ref_col='corpusid',
                               drop_snodes=True):
        """
        Analyzes the influence of the topic model and the similarity graph
        parameters on the quality of the graph.

        The similarity graph is validated using a citations graph.

        Parameters
        ----------
        path2models : str
            Path specifying the class of models to validate.
        path2nodenames : str
            Path to the file containing the node names
        corpus : str
            Name of the corpus
        label_RG : str
            Name of the reference graph
        spf : float
            Sampling factor. To reduce the corpus size.
        radius : float or None, optional (default=None)
            Distance threshold for the sparse graph generation.
            If None, epn must be specified
        epn : float or None, optional (default=None)
            Target number of edges per node in the sparse graph.
            If None then a value of radius must be specified
        cd_algs : list of str, optional (default=[])
            Community detection algorithms to validate
        metrics : list of str, optional (default=[])
            Metrics to evaluate for community comparison.
        col_ref : str, optional (default='corpusid')
            Name of the column in the metadata files containing the node names

        Notes
        -----
            radius and epn are mutually exclusive parameters: one and
            only one must be specified.
        """

        # ################
        # Corpus selection

        # ##########
        # Parameters

        if radius is None and epn is None:
            logging.error("-- -- One of radius or epn parameters mus be " +
                          "specified")
        elif radius is not None and epn is not None:
            logging.error("-- -- Only one radius or epn parameters must be " +
                          "specified")

        # #####################
        # Paths to topic models

        # Models to analyze. The name of each model is also the name of the
        # folder that contains it.
        models = [f for f in os.listdir(path2models)
                  if f.split('_')[0] == corpus and 'interval' in f]

        # #######################
        # Load co-citations graph

        # Load co-citations graph from DB or from a file in the supergraph
        logging.info('-- -- Loading reference graph...')
        if label_RG not in self.SG.metagraph.nodes:
            logging.error("-- Reference graph does not exist. Use " +
                          "self.compute_reference_graph() first")
            exit()

        # If the selected reference subgraph already exists, just load it
        # from file.
        self.SG.activate_snode(label_RG)
        nodes_ref = self.SG.snodes[label_RG].nodes
        spf = 1
        n_gnodes = int(spf * len(nodes_ref))

        # # List of indices of the selected nodes. This will be used to
        # # select the topic vectors in the topic matrix
        # np.random.seed(3)
        # idx = sorted(
        #     np.random.choice(range(n_gnodes), n_gnodes, replace=False))
        # nodes_ref = [nodes_ref[i] for i in idx]

        # # Sample reference graph, selecting nodes in nodes_ref only
        # self.SG.sub_snode(label_RG, nodes_ref, ylabel=None, sampleT=True,
        #                   save_T=True)

        # Compute communities for the scitation graph
        if len(cd_algs) > 0:
            for alg in cd_algs:
                self.SG.snodes[label_RG].detectCommunities(
                    alg=alg, label=alg)
            self.SG.save_supergraph()

        # #######################
        # Model-by-model analysis

        # Initialize output variables
        n_topics, n_nodes, n_edges, comp_time = [], [], [], []
        cc, cmax, rmax, c_rel, RG_sim = [], [], [], [], []
        alpha, interval, n_run = [], [], []
        cd_metric = {}
        for alg in cd_algs:
            cd_metric[alg] = {'nc': [], 'lc': [], 'lc_rel': []}
            for m in metrics:
                cd_metric[alg][m] = []
        extended_metrics = metrics + ['nc', 'lc', 'lc_rel']

        for i, model in enumerate(models):

            logging.info(f"-- Model {i} ({model}) out of {len(models)}")

            # Select topic model
            path = os.path.join(path2models, model)

            # Take metadata information about the model from the model name
            model_metadata = model.split('_')
            alpha_i = model_metadata[4]
            interval_i = model_metadata[6]
            n_run_i = model_metadata[8]

            # #####################
            # Document-topic matrix

            # Load doc-topic matrix for the whole corpus
            data, df_nodes = self.DM.readCoordsFromFile(
                fpath=path, sparse=True, path2nodenames=path2nodenames,
                ref_col=ref_col)
            T = data['thetas']                     # Doc-topic matrix
            nodes_model = df_nodes[ref_col].astype(str).tolist()
            # nodes = df_nodes[ref_col].tolist()     # List of node ids

            # ###########
            # Subsampling

            # Create graph
            self.SG.makeSuperNode(label=model, nodes=nodes_model, T=T)
            # Take the samples in the reference graph only
            self.SG.sub_snode(model, nodes_ref, sampleT=True)

            t0 = time.time()
            n_edges_t = int(epn * n_gnodes)
            self.SG.computeSimGraph(
                model, R=radius, n_edges=n_edges_t, n_gnodes=n_gnodes,
                similarity='He2', verbose=False)
            dt = time.time() - t0

            # Compute connected components
            self.SG.snodes[model].detectCommunities(alg='cc', label='cc')

            # Get output parameters
            md = self.SG.snodes[model].metadata
            n_topics.append(T.shape[1])
            n_nodes.append(n_gnodes)
            n_edges.append(md['edges']['n_edges'])
            rmax.append(md['edges']['R'])
            cc.append(md['communities']['cc']['n_communities'])
            cmax.append(md['communities']['cc']['largest_comm'])
            c_rel.append(md['communities']['cc']['largest_comm'] / n_gnodes)
            comp_time.append(dt)
            alpha.append(alpha_i)
            interval.append(interval_i)
            n_run.append(n_run_i)

            # Compute similarity with the citations graph
            score = self.SG.cosine_sim(label_RG, model)
            RG_sim.append(score)

            # Compute communities for the semantic graphs
            for alg in cd_algs:
                # Compute community
                self.SG.snodes[model].detectCommunities(alg=alg, label=alg)

                md = self.SG.snodes[model].metadata
                md_alg = md['communities'][alg]    # To abbreviate
                cd_metric[alg]['nc'].append(md_alg['n_communities'])
                cd_metric[alg]['lc'].append(md_alg['largest_comm'])
                cd_metric[alg]['lc_rel'].append(
                    md_alg['largest_comm'] / n_gnodes)

                # Compare communities with those of the scitations graph
                comm1 = self.SG.snodes[label_RG].df_nodes[alg].tolist()
                comm2 = self.SG.snodes[model].df_nodes[alg].tolist()
                edges = self.SG.snodes[label_RG].edge_ids
                weights = self.SG.snodes[label_RG].weights
                CD = CommunityPlus()
                for m in metrics:
                    if m in ['modularity', 'coverage', 'performance']:
                        # Community vs graph metric
                        score = CD.community_metric(edges, weights, comm2, m)
                    else:
                        # Community vs community metric
                        score = CD.compare_communities(comm1, comm2, method=m)
                    cd_metric[alg][m].append(score)

            # Remove snode, because it is no longer required
            if drop_snodes:
                self.SG.drop_snode(model)

        # ############
        # Save results

        # Sort result variables by number of topics
        # We need to save the original ordering of the number of topics to
        # sort the cd metrics afterwards.
        n_topics_0 = copy(n_topics)
        (n_topics, models, n_nodes, n_edges, comp_time, cc, cmax, rmax, c_rel,
            comp_time, RG_sim, alpha, interval, n_run) = tuple(zip(*sorted(
                zip(n_topics, models, n_nodes, n_edges, comp_time, cc, cmax,
                    rmax, c_rel, comp_time, RG_sim, alpha, interval,
                    n_run))))

        # Compute communities for the reference graph
        for alg in cd_algs:
            for m in metrics:
                (aux, cd_metric[alg][m]) = tuple(zip(*sorted(zip(
                    n_topics_0, cd_metric[alg][m]))))

        # Create summary table
        df = pd.DataFrame({'Model': models,
                           'No. of topics': n_topics,
                           'alpha': alpha,
                           'interval': interval,
                           'n_run': n_run,
                           'Radius': rmax,
                           'Time': comp_time,
                           'Number of nodes': n_nodes,
                           'Number of edges': n_edges,
                           'Connected components': cc,
                           'Largest component': cmax,
                           'Relative max CC': c_rel,
                           'Ref. graph similarity': RG_sim})

        # Add community metrics
        for alg in cd_algs:
            for m in extended_metrics:
                col = f"{alg}_{m}"
                df[col] = cd_metric[alg][m]

        print("Summary of results:")
        print(df)

        # Save summary table
        preffix = f'{corpus}_{n_gnodes}_{epn}'
        fname = f'{preffix}.xls'
        out_dir = os.path.join(self.path2project, self.f_struct['output'],
                               'validation', preffix, 'results')
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        out_path = os.path.join(out_dir, fname)
        df.to_excel(out_path)

        return

    def validate_topic_models(self, corpus, ref_graph=None):
        """
        Analyzes the influence of the topic model on te quality of the
        similarity graphs

        The similarity graph is validated using a citations graph.

        Parameters
        ----------
        corpus : str {'Pu', 'Co'}
            Corpus (Pu: Semantic Scholar, or Co: Crunchbase data)
        """

        corpus_label = {'Co': 'Crunch4Ever', 'Pu': 'S24Ever'}[corpus]
        ref_col = {'Co': 'companyID', 'Pu': 'paperID'}[corpus]

        path2nodenames = (f'/home/jarenas/github/TM4Ever/TM4Ever20200221/'
                          f'corpus/{corpus_label}/{corpus_label}_metadata.csv')
        path2models = '/home/jarenas/github/TM4Ever/TM4Ever20200221/models/tm'

        # Parameters
        p = self.global_parameters['validate_all_models']
        spf = 1    # p['spf']       # Sampling factor
        epn = 20                    # Target number of edges per node
        print(f"Number of edges per node: {epn}")

        # Validate modesl, one by one..
        self._validate_topic_models(
            path2models, path2nodenames, corpus_label, ref_graph, spf,
            epn=epn, ref_col=ref_col)

        return

    def show_validation_results_e(self, path):
        """
        Shows the results of the topic model validation in
        self.validate_topic_models()

        Parameters
        ----------
        path: str
            Path to data
        """

        # ###############
        # Read data files
        # ###############

        # Read the file names in the folder containing the xls reports
        data_dir = os.path.join(path, 'results')
        data_files = sorted(os.listdir(data_dir))

        # Read all result files
        # sim, rescale, n_nodes, n_edges, ig, tm_class = [], [], [], [], [], []
        n_nodes, epn = [], []
        alpha, n_run, interval = [], [], []

        params, df_dict = {}, {}
        # fname_struct = ['corpus', 'sim', 'rescale', 'n_nodes', 'n_edges',
        #                 'ig', 'tm_class']
        for f in data_files:
            if f.endswith('.xls'):
                fname = os.path.splitext(f)[0]
                fname_parts = fname.split('_')

                # Read parameters from the file name
                n_nodes_f = fname_parts[1]
                epn_f = fname_parts[2]

                n_nodes.append(n_nodes_f)
                epn.append(epn_f)

                # Read report from file
                fpath = os.path.join(data_dir, f)
                df_dict[fname] = pd.read_excel(fpath)
                params[fname] = {'n': n_nodes_f, 'e': epn_f}

                alpha += df_dict[fname]['alpha'].tolist()
                n_run += df_dict[fname]['n_run'].tolist()
                interval += df_dict[fname]['interval'].tolist()

        # ############
        # Plot results
        # ############

        # Ordered list of parameters.
        n_nodes = sorted(list(set(n_nodes)))
        epn = sorted(list(set(epn)))
        alpha = sorted(list(set(alpha)))
        n_run = sorted(list(set(n_run)))
        interval = sorted(list(set(interval)))

        # Get the list of all possiblo dataframe columns to visualize
        cols = []
        for name, df in df_dict.items():
            cols += df.columns.tolist()

        # Remove columns that will not become y-coordinates
        cols = list(set(cols) - {'Unnamed: 0', 'Model', 'No. of topics',
                                 'Number of nodes'})

        # Dictionary of abreviations (for the file names)
        cols2plot = list(set(cols) - {'alpha', 'interval', 'n_run'})
        abbreviation = {x: x for x in cols2plot}
        abbreviation.update({'Radius': 'Rad',
                             'Time': 't',
                             'Number of edges': 'ne',
                             'Connected components': 'cc',
                             'Largest component': 'lc',
                             'Relative max CC': 'rc',
                             'Ref. graph similarity': 'cs'})

        # The following nested loop is aimed to make multiple plots from xls
        # files inside the same directory. It is actually not needed if all
        # xls files in the given folder have the same parameter values.
        for n in n_nodes:
            for e in epn:

                fnames = [x for x in df_dict if
                          params[x]['n'] == n and params[x]['e'] == e]

                if len(fnames) == 0:
                    continue

                for var, ab in abbreviation.items():

                    # #################################################
                    # Plot figure for the current value of e, n and var
                    fig, ax = plt.subplots()
                    print(fnames)
                    for fname in fnames:
                        df_f = df_dict[fname]
                        for a in alpha:
                            df_a = df_f[df_f['alpha'] == a]
                            for i in interval:
                                df = df_a[df_a['interval'] == i]
                                x = df['No. of topics']
                                y = df[var]

                                base_line, = ax.plot(x, y, '.')

                                df_av = df.groupby('No. of topics').mean()
                                x = df_av.index
                                y = df_av[var]
                                ax.plot(x, y, '.-', label=f'a={a}, i={i}',
                                        color=base_line.get_color())

                    ax.set_xlabel('No. of topics')
                    ax.set_ylabel(var)
                    ax.set_title(f'Nodes: {n}, Edges per node: {e}, ')
                    ax.legend()
                    ax.grid()
                    plt.show(block=False)

                    # Save figure
                    out_dir = os.path.join(path, 'figs')
                    tag = '_'.join(fname_parts[0:5])
                    fname = f'{tag}_{ab}.png'
                    if not os.path.exists(out_dir):
                        os.makedirs(out_dir)
                    out_path = os.path.join(out_dir, fname)
                    plt.savefig(out_path)

        return

    def _plot_results(self, x, y, xlabel, ylabel, fpath, xscale='log',
                      yscale='log'):
        """
        Generic plot for the data dictionaries generated by the graph
        analysis

        Parameters
        ----------
        x : dict
            A dictionary of lists of arrays of x values
        y : dict
            A dictionary of arrays of y values
        xlabel : dict
            A dictionary of x-labels
        ylabel : dict
            A dictionary of y-labels
        fpath : str
            Path to the output file
        xscale : str {'log', 'lin'}, optional (default='log')
            Scale of the x coordinate (log or lin)
        yscale : str {'log', 'lin'}, optional (default='log')
            Scale of the y coordinate (log or lin)
        """

        plt.figure()

        # A quick one-line to sort the models in a natural order (this has
        # an influence on the order of the legend plot)
        sorted_models = [z[1] for z in sorted([(int(k.split('_')[-1]), k)
                                               for k in x])]
        for model in sorted_models:
            if xscale == 'log' and yscale == 'log':
                plt.loglog(x[model], y[model], '.:', label=model)
            elif xscale == 'log' and yscale == 'lin':
                plt.semilogx(x[model], y[model], '.:', label=model)
            elif xscale == 'lin' and yscale == 'log':
                plt.semilogy(x[model], y[model], '.:', label=model)
            else:
                plt.plot(x[model], y[model], '.:', label=model)

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend()
        plt.show(block=False)
        plt.savefig(fpath)

        return

    def get_source_info(self):
        """
        Extracts basic information about the doc-topic matrices in the source
        folders.

        For each matrix, plot the distribution of the number of nonzero topics
        """

        folders = os.listdir(self.path2tm)
        p2p = self.path2project   # This is just to abbreviate
        out_dir = os.path.join(p2p, self.f_struct['output'], 'source_info')
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        models, n_items, n_topics, n_nz, r_nz = [], [], [], [], []

        # Explore all source folders contatinig a file names modelo_sparse.npz
        for folder in folders:

            # Load data matrix
            path = os.path.join(self.path2tm, folder)
            fpath = os.path.join(path, 'modelo_sparse.npz')

            if os.path.isfile(fpath):
                # Load corpus data
                logging.info(f'---- Loading model {folder}')
                data, df_nodes = self.DM.readCoordsFromFile(
                    path, sparse=True, ref_col='corpusid')
                T = data['thetas']
                # nodes = df_nodes['corpusid'].tolist()

                # Keep metadata
                models.append(folder)
                n_items.append(T.shape[0])
                n_topics.append(T.shape[1])
                n_nz.append(T.count_nonzero())
                r_nz.append(100 * n_nz[-1] / n_items[-1] / n_topics[-1])

                # Plot distribution of the number of nonzero topics
                num_topics = (T > 0).sum(axis=1).T.tolist()[0]
                nt_hist = Counter(num_topics)

                plt.figure()
                plt.stem(nt_hist.keys(), nt_hist.values())
                plt.xlabel('Number of nonzero topics')
                plt.ylabel('Number of documents')
                plt.title(folder)
                plt.show(block=False)
                fpath = os.path.join(out_dir, f'ntopics_hist_{folder}.png')
                plt.savefig(fpath)
            else:
                logging.info(f'---- Model {folder} not available')

        # Create summary table
        df = pd.DataFrame({'Name': models,
                           'Topics': n_topics,
                           'Items': n_items,
                           'Non-zeros': n_nz,
                           'Ratio nonzeros (%)': r_nz})

        # Save summary table
        out_path = os.path.join(p2p, self.f_struct['output'],
                                'source_models.xls')
        df.to_excel(out_path)

        return

    def get_equivalent_classes(self, corpus):
        """
        Extracts basic information about the doc-topic matrices in the source
        folders.

        For each matrix, plot the distribution of the number of nonzero topics

        Parameters
        ----------
        corpus : str
            Name of the corpus
        """

        # Parameters
        fname = 'modelo_sparse.npz'     # Name of topic model files
        out_fname = f'equivalence_classes_{corpus}.xls'

        # ########################
        # Location of data sources
        # ########################

        # Prepare ouput folder
        p2p = self.path2project   # This is just to abbreviate
        out_dir = os.path.join(p2p, self.f_struct['output'], 'source_info',
                               'equiv_classes')
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        # Get the parameter of the selected collection of topic models
        if corpus == 'noACL':
            root_dir = self.path2tm
            alg_level = False
            path2nodenames = None
            ref_col = 'corpusid'
        elif corpus == 'ACL':
            root_dir = self.path2ACL
            alg_level = True
            path2nodenames = os.path.join(self.path2ACL, 'corpus', 'ACL',
                                          'allmetadata.csv')
            ref_col = 'ACLid'

        # Get paths to all folders containing a topic model file
        paths = []
        for dir_name, subdirs, files in os.walk(root_dir, topdown=False):
            if fname in files:
                paths.append(dir_name)

        # #################################
        # Computation of equivalent classes
        # #################################

        models, algs, n_nodes, n_topics, sparsity = [], [], [], [], []
        n_1links, n_clusters, n_preclusters = [], [], []

        # Explore all source folders contatinig a file names modelo_sparse.npz
        for path in paths:

            parent, folder = os.path.split(path)
            if alg_level:
                parent, alg = os.path.split(parent)
                model = f'{folder}_{alg}'
            else:
                model = folder
                alg = None

            # Load data matrix
            logging.info(f'-- Equivalent classes for model {model}')

            # Load corpus data
            data, df_nodes = self.DM.readCoordsFromFile(
                path, sparse=True, ref_col=ref_col,
                path2nodenames=path2nodenames)
            T = data['thetas']
            # nodes = df_nodes[ref_col].tolist()

            # Plot distribution of the number of nonzero topics
            logging.info(f'-- -- Computing classes')

            # Compute equivalent classes
            sg = SimGraph(T)
            sg.cluster_equivalent_nodes()

            models.append(model)
            algs.append(alg)
            n_nodes.append(T.shape[0])
            n_topics.append(T.shape[1])
            sparsity.append(1 - T.count_nonzero() / np.prod(T.shape))
            n_1links.append(sg.n_edges)
            n_preclusters.append(sg.n_preclusters)
            n_clusters.append(sg.n_clusters)

            # Show main results
            print(f"-- -- Number of nodes = {n_nodes}")
            print(f"-- -- Clusters = {n_clusters}")
            print(f"-- -- Unit_links = {n_1links}")

        # #################################
        # Computation of equivalent classes
        # #################################

        # Create summary table
        df = pd.DataFrame({'Model': models,
                           'Algorithm': algs,
                           'Nodes': n_nodes,
                           'Topics': n_topics,
                           'Sparsity': sparsity,
                           'Equivalent classes': n_clusters,
                           'Distinct nonzero patterns': n_preclusters,
                           'Unit links': n_1links})
        print(df)

        # Save summary table of equivalence classes
        out_path = os.path.join(out_dir, out_fname)
        df.to_excel(out_path)

        return

    def show_equivalent_classes(self):
        """
        Show equivalent classes
        """

        # ###############
        # Read data files
        # ###############

        # Read the file names in the folder containing the xls reports
        p2p = self.path2project   # This is just to abbreviate
        data_dir = os.path.join(p2p, self.f_struct['output'], 'source_info',
                                'equiv_classes')
        fname = f'equivalence_classes_ACL.xls'

        # Read report from file
        fpath = os.path.join(data_dir, fname)
        df = pd.read_excel(fpath)

        # Select datat o plot
        keys = ['Equivalent classes', 'Unit links', 'Sparsity',
                'Distinct nonzero patterns']
        nfigs = len(keys)

        # Create figure handlers
        fig, ax = [None] * nfigs, [None] * nfigs
        for i in range(nfigs):
            fig[i], ax[i] = plt.subplots()

        # Plot results
        algorithms = sorted(list(set(df['Algorithm'].tolist())))
        for a in algorithms:
            df_a = df[df['Algorithm'] == a].sort_values('Topics')
            nt = df_a['Topics']

            for i, key in enumerate(keys):
                y = df_a[key]
                ax[i].plot(nt, y, '.:', label=a)
                ax[i].set_xlabel('Topics')
                ax[i].set_ylabel(key)

        # Complete and save figures
        for i, key in enumerate(keys):
            ax[i].legend()
            ax[i].grid()
            key_ = key.replace(' ', '_')
            fig[i].savefig(os.path.join(data_dir, f'{key_}.png'))

        plt.show(block=False)

        return

    def analyze_radius(self, corpus):
        """
        This method analyzes the generation of similarity (semantic) graphs.
        More specifically, it explores the variation of several graph
        parameters as a function of the radius, for a fixed number of nodes,
        for a given corpus and for several topic models (from the same corpus)

        The radius is the bound on the JS distance used to sparsify the graph.
        The number of nodes is a subsample of the total number of nodes.

        Parameters
        ----------
        corpus : str
            Corpus. All topic models based on this corpus will be will be
            analyzed.
        """

        # ##########
        # Parameters

        n_gnodes = 5000      # Number of nodes
        Rmax = 1.00001       # Maximum radius of the similarity graph
        sim = 'He2->JS'      # Similarity measure

        # Select all models for the given corpus:
        # The mixed corpus is excluded
        folders = os.listdir(self.path2tm)
        models = [f for f in folders if f.split('_')[0] == corpus and
                  f != 'FECYT_AI_SCOPUS_AI_PATSTAT_AI_120']

        # Output folder
        p2p = self.path2project   # This is just to abbreviate
        out_dir = os.path.join(p2p, self.f_struct['output'], 'thresholding')
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        radius, comp_time, n_edges, density, cc, cmax = {}, {}, {}, {}, {}, {}

        # #######################
        # Model-by-model analysis
        for model in models:

            # Select topic model
            path = os.path.join(self.path2tm, model)

            # #####################
            # Document-topic matrix

            # Load doc-topic matrix for the whole corpus
            logging.info(f'-- Loading data matrix from {path}')
            data, df_nodes = self.DM.readCoordsFromFile(
                path, sparse=True, ref_col='corpusid')
            T = data['thetas']
            nodes = df_nodes['corpusid'].tolist()

            # Take a random sample with n_gnodes
            np.random.seed(3)    # For comparative purposes (provisional)
            idx = np.random.choice(range(len(nodes)), n_gnodes, replace=False)
            T = T[idx]
            nodes = [nodes[i] for i in idx]

            # ########################
            # Compute similarity graph

            comp_time[model], n_edges[model], density[model] = [], [], []
            cc[model], cmax[model] = [], []

            radius[model] = np.linspace(0.1, Rmax, num=19)
            self.SG.makeSuperNode(label=model, nodes=nodes, T=T)

            # Compute similarity graph for each value of the radius.
            for R in radius[model]:
                print(f"-- R = {R}")

                # Create datagraph with the full feature matrix
                t0 = time.time()
                self.SG.computeSimGraph(model, R=R, n_gnodes=n_gnodes,
                                        similarity=sim, verbose=False)
                comp_time[model].append(time.time() - t0)

                # Compute connected components
                self.SG.snodes[model].detectCommunities(alg='cc', label='cc')

                # Get output parameters
                md = self.SG.snodes[model].metadata
                n_edges[model].append(md['edges']['n_edges'])
                density[model].append(100 * md['edges']['density'])
                cc[model].append(md['communities']['cc']['n_communities'])
                cmax[model].append(md['communities']['cc']['largest_comm'])

            # ############
            # Save results

            # Create summary table
            df = pd.DataFrame({'Radius': radius[model],
                               'Time': comp_time[model],
                               'Number of edges': n_edges[model],
                               'Density (%)': density[model],
                               'Connected components': cc[model],
                               'Largest component': cmax[model]})
            print(df)

            # Save summary table
            fname = f'radius_effect_{model}.xls'
            out_path = os.path.join(out_dir, fname)
            df.to_excel(out_path)

        # Plot results
        xlabel = 'Radius'

        fpath = os.path.join(out_dir, f'{corpus}_comp_time.png')
        self._plot_results(
            radius, comp_time, xlabel, 'Time', fpath, xscale='lin')

        fpath = os.path.join(out_dir, f'{corpus}_density.png')
        self._plot_results(
            radius, density, xlabel, 'Density', fpath, xscale='lin')

        fpath = os.path.join(out_dir, f'{corpus}_n_edges.png')
        self._plot_results(
            radius, n_edges, xlabel, 'Number of edges', fpath, xscale='lin')

        fpath = os.path.join(out_dir, f'{corpus}_CC.png')
        self._plot_results(
            radius, cc, xlabel, 'Number of connected components', fpath,
            xscale='lin')

        fpath = os.path.join(out_dir, f'{corpus}_maxCC.png')
        self._plot_results(
            radius, cmax, xlabel, 'Largest connected component', fpath,
            xscale='lin')

        return

    def _analyze_sampling(self, corpus, n_links=None, max_nodes=None, tag=""):
        """
        This method analyzes the effect of sampling for a fixed number of
        edges.

        Several graph parameters are computed for the given corpus, for
        different topic models and for a given number of target links, as a
        function of the sampling factor (i.e. the number of nodes)

        It uses the SimGraph class to compute the graphs. Since the number of
        edges cannot be fixed (only the number of nodes and the radius can
        be can be set in advance) we proceed in two steps: in a first step, we
        compute a denser graph, with a large number of nodes, and, based on it,
        we compute the Radius required to get the desired number of edges.

        UPDATE: The current version of the computeSimGraph() method from class
        simgraph admits entering the number of edges. The code below could be
        simplifying taking advantage of this new feature.

        Parameters
        ----------
        corpus : str
            Corpus. All topic models based on this corpus will be analyzed
        n_links : int
            Target number of links. If None, the output graph should be fully
            connected
        max_nodes : int
            Maximum number of nodes. If the corpus has more nodes, it is
            subsampled.
        tag : str
            Label identifier for the output file names.
        """

        # ##########
        # Parameters
        sim = 'He2->JS'       # Similarity measure
        fully_connected = n_links is None

        # Select all models for the given corpus:
        folders = os.listdir(self.path2tm)
        models = [f for f in folders if f.split('_')[0] == corpus and
                  f != 'FECYT_AI_SCOPUS_AI_PATSTAT_AI_120']

        comp_time, n_edges, cc, cmax, rmax, n_nodes = {}, {}, {}, {}, {}, {}
        c_rel = {}

        # #######################
        # Model-by-model analysis
        for model in models:

            # Select topic model
            path = os.path.join(self.path2tm, model)
            p2p = self.path2project   # This is just to abbreviate

            # #####################
            # Document-topic matrix

            # Load doc-topic matrix for the whole corpus
            logging.info(f'-- Loading data matrix from {path}')
            data, df_nodes = self.DM.readCoordsFromFile(
                path, sparse=True, ref_col='corpusid')
            T = data['thetas']
            nodes = df_nodes['corpusid'].tolist()

            if max_nodes is not None:
                max_nodes = min(max_nodes, T.shape[0])
                # Data sampling
                np.random.seed(3)    # For comparative purposes (provisional)
                # Randomize node ordering
                idx = np.random.choice(
                    range(max_nodes), max_nodes, replace=False)
                T = T[idx]
                nodes = [nodes[i] for i in idx]
            n_tot = len(nodes)

            # Select the range of sampling factors
            # spf = [0.2, 0.4, 0.6, 0.8, 1.0]
            spf = np.logspace(-9, 0, num=20, base=2)
            comp_time[model], n_edges[model], n_nodes[model] = [], [], []
            cc[model], cmax[model], rmax[model], c_rel[model] = [], [], [], []
            radius = 1

            for r in spf:

                # Select the reduced data matrix
                n_gnodes = int(r * n_tot)
                # Compute minimal sampling factor.
                if fully_connected:
                    n_links = n_gnodes * (n_gnodes - 1) // 2

                if n_links <= n_gnodes * (n_gnodes - 1) // 2:
                    # We arrive here only if the number of nodes is enough to
                    # get n_links
                    Tr = T[:n_gnodes]
                    nodes_r = nodes[:n_gnodes]
                    print(
                        f"-- -- Sampling up to {n_gnodes} and {n_links} links")

                    # First Trial: since we do not know the value of R required
                    # to get n_links, we compute a denser network first:
                    self.SG.makeSuperNode(label=model, nodes=nodes_r, T=Tr)
                    t0 = time.time()
                    self.SG.computeSimGraph(
                        model, R=radius, n_gnodes=n_gnodes, similarity=sim,
                        verbose=False)
                    print(f"-- -- Tentative simgraph in {time.time()-t0} secs")

                    # Sort weights and get the weight ranked n_links...
                    lw = len(self.SG.snodes[model].weights)
                    w = sorted(
                        list(zip(self.SG.snodes[model].weights, range(lw))),
                        reverse=True)
                    w = w[:n_links]
                    w_min = w[-1][0]

                    # Compute the threshold required to get the target links
                    # This is the radius that should provide n_links edges.
                    radius = np.sqrt(1 - w_min) * radius
                    if radius == 0:
                        logging.warning("-- -- All selected links, weight 1")
                    radius = max(1e-20, radius)

                    # Compute graph with the target number of links
                    t0 = time.time()
                    self.SG.computeSimGraph(model, R=radius, n_gnodes=n_gnodes,
                                            similarity=sim, verbose=False)
                    dt = time.time() - t0
                    comp_time[model].append(dt)
                    print(f"-- -- Simgraph in {dt} seconds")

                    # Compute connected components
                    self.SG.snodes[model].detectCommunities(
                        alg='cc', label='cc')

                    # Get output parameters
                    md = self.SG.snodes[model].metadata
                    rmax[model].append(radius)
                    n_nodes[model].append(n_gnodes)
                    n_edges[model].append(md['edges']['n_edges'])
                    cc[model].append(md['communities']['cc']['n_communities'])
                    cmax[model].append(md['communities']['cc']['largest_comm'])
                    c_rel[model].append(
                        md['communities']['cc']['largest_comm'] / n_gnodes)

                    # Remove snode, because it is no longer required
                    self.SG.drop_snode(model)

            # ############
            # Save results

            # Create summary table
            df = pd.DataFrame({'Radius': rmax[model],
                               'Time': comp_time[model],
                               'Number of nodes': n_nodes[model],
                               'Number of edges': n_edges[model],
                               'Connected components': cc[model],
                               'Largest component': cmax[model],
                               'Relative max CC': c_rel[model]})
            print(df)

            # Save summary table
            fname = f'{tag}_{model}.xls'
            out_dir = os.path.join(p2p, self.f_struct['output'], 'sampling')
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            out_path = os.path.join(out_dir, fname)
            df.to_excel(out_path)

        # Plot results
        xlabel = 'Number of nodes'
        fpath = os.path.join(out_dir, f'{tag}_{corpus}_comp_time.png')
        self._plot_results(n_nodes, comp_time, xlabel, 'Time', fpath)

        fpath = os.path.join(out_dir, f'{tag}_{corpus}_CC.png')
        self._plot_results(
            n_nodes, cc, xlabel, 'Number of connected components', fpath)

        fpath = os.path.join(out_dir, f'{tag}_{corpus}_maxCC.png')
        self._plot_results(
            n_nodes, cmax, xlabel, 'Largest connected component', fpath)

        fpath = os.path.join(out_dir, f'{tag}_{corpus}_relCC.png')
        self._plot_results(
            n_nodes, c_rel, xlabel, 'Relative size of the largest CC', fpath)

        fpath = os.path.join(out_dir, f'{tag}_{corpus}_rmax.png')
        self._plot_results(
            n_nodes, rmax, xlabel, 'Radius', fpath, yscale='lin')

        return

    def analyze_sampling(self, corpus):
        """
        Analyze sampling for a given corpus

        Parameters
        ----------
        corpus : str
            Corpus. All topic models based on this corpus will be will be
            analyzed.
        """

        self._analyze_sampling(corpus, n_links=400_000, tag='sampling',
                               max_nodes=200_000)

        return

    def _validate_topic_models_old(self, path2models, d, spf, rescale=True,
                                   radius=None, n_edges_t=None, g=1,
                                   cd_algs=[], metrics=[]):
        """
        Analyzes the influence of the topic model and the similarity graph
        parameters on the quality of the graph.

        The similarity graph is validated using a citations graph.

        Parameters
        ----------
        path2models : str
            Path specifying the class of models to validate.
        d : float
            Distance measure (JS, He or l1)
        spf : float
            Sampling factor. To reduce the corpus size.
        rescale : bool, optional (default=True)
            If True, it rescales similarity weights to span from 0 to 1.
        radius : float or None, optional (default=None)
            Distance threshold for the sparse graph generation.
            If None then an interger n_edges_t must be specified
        n_edges_t : int or None, optional (default=None)
            Target number of edges of the sparse graph.
            If None then a value of radius must be specified
        g : float, optional (default=1)
            Exponent for the conversion distances-weights
        cd_algs : list of str, optional (default=[])
            Community detection algorithms to validate
        metrics : list of str, optional (default=[])
            Metrics to evaluate for community comparison.

        Notes
        -----
            radius and n_edges_t are mutually exclusive parameters: one and
            only one must be specified.
        """

        # ################
        # Corpus selection

        # Corpus and model specification
        corpus = 'ACL'
        # Name of the column in metadata files containing the node identifiers
        ref_col = 'ACLid'

        # ##########
        # Parameters

        if radius is None and n_edges_t is None:
            logging.error("-- -- One of radius or n_edges_t parameters must " +
                          "be specified")
        elif radius is not None and n_edges_t is not None:
            logging.error("-- -- Only one radius or n_edges_t parameters " +
                          "must be specified")

        # Read the type of topic models from the path name.
        tm_class = os.path.split(path2models)[-1]

        # Take the similarity measure corresponding to the selected distance
        smap = {'JS': 'He2->JS', 'He': 'He2', 'l1': 'l1'}
        sim = smap[d]

        # #####################
        # Paths to topic models

        # Path to the folder containing the node names (same for all)
        path2nodenames = os.path.join(
            self.path2ACL, 'corpus/ACL/allmetadata.csv')
        # Path to the folder containing the folders with the topic models
        folders = os.listdir(path2models)

        # #######################
        # Load co-citations graph

        # Load co-citations graph from DB or from a file in the supergraph
        logging.info('-- -- Loading scitations graph...')
        label_cocit = f'cocit_spf{int(1000*spf)}'

        if label_cocit in self.SG.metagraph.nodes:
            # If the selected cocitation subgraph already exists, just load it
            # from file.
            self.SG.activate_snode(label_cocit)
            nodes_ref = self.SG.snodes[label_cocit].nodes
            n_gnodes = len(nodes_ref)

            # List of indices of the selected nodes. This will be used to
            # select the topic vectors in the topic matrix
            np.random.seed(3)
            idx = sorted(
                np.random.choice(range(n_gnodes), n_gnodes, replace=False))
        else:
            if 'cocitations' not in self.SG.metagraph.nodes:
                # Load co-citations graph from database.
                self.import_co_citations_graph()
            # Activate cocitations graph
            # (This is necessary no matter if self.import_co_citations_graph()
            # was invoked, because self.import_co_citations_graph deactivates
            # the graph, and we need re-activation to reload from file)
            self.SG.activate_snode('cocitations')

            # Read list of nodes.
            # The nodes in the citations graph and in the topic models are not
            # exactly the same. We will construct a version of the citation
            # graph with the nodes selected from the topic models.

            # Read list of nodes from the topic models
            df_nodes_tm = pd.read_csv(path2nodenames, usecols=[ref_col])
            n_tm = len(df_nodes_tm)
            # Sub-sample the list.
            # This is done because the complete list of nodes may be too large
            # to build dense or almost dense graphs
            n_gnodes = int(spf * n_tm)   # Target number of nodes
            np.random.seed(3)
            idx = sorted(
                np.random.choice(range(n_gnodes), n_gnodes, replace=False))
            nodes_ref = df_nodes_tm.iloc[idx][ref_col].tolist()

            # Subsample the cocitations graph
            self.SG.sub_snode('cocitations', nodes_ref, ylabel=label_cocit)
            self.SG.deactivate_snode('cocitations')

        # df_edges_cg = self.SG.snodes[label_cocit].df_edges

        # Compute communities for the scitation graph
        for alg in cd_algs:
            self.SG.snodes[label_cocit].detectCommunities(alg=alg, label=alg)
        self.SG.save_supergraph()

        # #######################
        # Model-by-model analysis

        # Models to analyze
        models = [f for f in folders if f.split('_')[0] == corpus]

        # Initialize output variables
        n_topics, n_nodes, n_edges, comp_time = [], [], [], []
        cc, cmax, rmax, c_rel, cocit_sim = [], [], [], [], []
        cd_metric = {}
        for alg in cd_algs:
            cd_metric[alg] = {'nc': [], 'lc': [], 'lc_rel': []}
            for m in metrics:
                cd_metric[alg][m] = []
        extended_metrics = metrics + ['nc', 'lc', 'lc_rel']

        for i, model in enumerate(models):

            logging.info(f"-- Model {i} ({model}) out of {len(models)}")

            # Select topic model
            path = os.path.join(path2models, model)

            # #####################
            # Document-topic matrix

            # Load doc-topic matrix for the whole corpus
            data, df_nodes = self.DM.readCoordsFromFile(
                fpath=path, sparse=True, path2nodenames=path2nodenames,
                ref_col=ref_col)
            T = data['thetas']                     # Doc-topic matrix
            # nodes = df_nodes[ref_col].tolist()     # List of node ids

            # ###########
            # Subsampling

            # Note that idx is the same for all models.
            T = T[idx]

            # Compute similarity graph
            self.SG.makeSuperNode(label=model, nodes=nodes_ref, T=T)
            t0 = time.time()
            self.SG.computeSimGraph(
                model, R=radius, n_edges=n_edges_t, n_gnodes=n_gnodes,
                similarity=sim, g=g, rescale=rescale, verbose=False)
            dt = time.time() - t0

            # Compute connected components
            self.SG.snodes[model].detectCommunities(alg='cc', label='cc')

            # Get output parameters
            md = self.SG.snodes[model].metadata
            n_topics.append(T.shape[1])
            n_nodes.append(n_gnodes)
            n_edges.append(md['edges']['n_edges'])
            rmax.append(md['edges']['R'])
            cc.append(md['communities']['cc']['n_communities'])
            cmax.append(md['communities']['cc']['largest_comm'])
            c_rel.append(md['communities']['cc']['largest_comm'] / n_gnodes)
            comp_time.append(dt)

            # Compute similarity with the citations graph
            score = self.SG.cosine_sim(label_cocit, model)
            cocit_sim.append(score)

            # Compute communities for the semantic graphs
            for alg in cd_algs:
                # Compute community
                self.SG.snodes[model].detectCommunities(alg=alg, label=alg)

                md = self.SG.snodes[model].metadata
                cd_metric[alg]['nc'].append(
                    md['communities'][alg]['n_communities'])
                cd_metric[alg]['lc'].append(
                    md['communities'][alg]['largest_comm'])
                cd_metric[alg]['lc_rel'].append(
                    md['communities'][alg]['largest_comm'] / n_gnodes)

                # Compare communities with those of the scitations graph
                comm1 = self.SG.snodes[label_cocit].df_nodes[alg].tolist()
                comm2 = self.SG.snodes[model].df_nodes[alg].tolist()
                edges = self.SG.snodes[label_cocit].edge_ids
                weights = self.SG.snodes[label_cocit].weights
                CD = CommunityPlus()
                for m in metrics:
                    if m in ['modularity', 'coverage', 'performance']:
                        # Community vs graph metric
                        score = CD.community_metric(edges, weights, comm2, m)
                    else:
                        # Community vs community metric
                        score = CD.compare_communities(comm1, comm2, method=m)

                    cd_metric[alg][m].append(score)

            # Remove snode, because it is no longer required
            self.SG.drop_snode(model)

        # ############
        # Save results

        # Sort result variables by number of topics
        # We need to save the original ordeiring of the number of topics to
        # sort the cd metrics afterwards.
        n_topics_0 = copy(n_topics)
        (n_topics, models, n_nodes, n_edges, comp_time, cc, cmax, rmax, c_rel,
            comp_time, cocit_sim) = tuple(zip(*sorted(zip(
                n_topics, models, n_nodes, n_edges, comp_time, cc, cmax, rmax,
                c_rel, comp_time, cocit_sim))))

        # Compute communities for the scitation graph
        for alg in cd_algs:
            for m in metrics:
                (aux, cd_metric[alg][m]) = tuple(zip(*sorted(zip(
                    n_topics_0, cd_metric[alg][m]))))

        # Create summary table
        df = pd.DataFrame({'Model': models,
                           'No. of topics': n_topics,
                           'Radius': rmax,
                           'Time': comp_time,
                           'Number of nodes': n_nodes,
                           'Number of edges': n_edges,
                           'Connected components': cc,
                           'Largest component': cmax,
                           'Relative max CC': c_rel,
                           'Cocitations similarity': cocit_sim})

        # Add community metrics
        for alg in cd_algs:
            for m in extended_metrics:
                col = f"{alg}_{m}"
                df[col] = cd_metric[alg][m]

        print("Summary of results:")
        print(df)

        # Save summary table
        if rescale:
            preffix = f'{corpus}_{sim}_R_{n_gnodes}_{n_edges_t}_ig{int(1/g)}'
        else:
            preffix = f'{corpus}_{sim}_U_{n_gnodes}_{n_edges_t}_ig{int(1/g)}'

        fname = f'{preffix}_{tm_class}.xls'
        out_dir = os.path.join(self.path2project, self.f_struct['output'],
                               'validation', preffix, 'results')
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        out_path = os.path.join(out_dir, fname)
        df.to_excel(out_path)

        return

    def validate_topic_models_old(self, path2models):
        """
        Calls to self._validate_topic_models for a specific value of the radius
        and the sampling factor.

        Parameters
        ----------
        path2models : str
            Path specifying the class of models to validate.
        """

        # ##########
        # Parameters
        d = 'JS'   # Similarity measure
        radius = 1.0001   # Radius threshold for the graph generation
        spf = 0.3         # Sampling factor
        rescale = True

        self._validate_topic_models(path2models, d, spf, radius=radius,
                                    rescale=rescale)

        return

    def validate_all_models_cd(self, d):
        """
        Analyzes the influence of the topic model and the similarity graph
        parameters on the quality of the community structures

        The similarity graph is validated using a citations graph.

        Parameters
        ----------
        d : str
            Similarity measure: options are: JS, He(llinger) or l1.
        """

        # Parameters
        # cd_algs = ['louvain', 'fastgreedy', 'walktrap', 'infomap',
        #            'labelprop']
        cd_algs = ['louvain', 'walktrap', 'labelprop', 'fastgreedy']
        metrics = ['vi', 'nmi', 'rand', 'adjusted_rand', 'split-join',
                   'modularity', 'coverage', 'performance']

        p = self.global_parameters['validate_all_models']
        spf = p['spf']              # Sampling factor
        rescale = p['rescale']      # True to map similarities to [0, 1]
        n_edges_t = p['n_edges_t']  # Target number of edges
        g = p['g']                  # Exponent of the dist-sim transformation
        print(f"Number of edges: {n_edges_t}")

        # Data folder
        folder = os.path.join(self.path2ACL, 'models', 'tm')
        tm_classes = [x for x in os.listdir(folder)
                      if os.path.isdir(os.path.join(folder, x))]

        # Validate modesl, one by one..
        for tmc in tm_classes:
            logging.info(
                f'-- Validating models of type {tmc} with {d} similarity')

            path2models = os.path.join(folder, tmc)
            out_dir = self._validate_topic_models(
                path2models, d, spf, rescale=rescale, n_edges_t=n_edges_t,
                g=g, cd_algs=cd_algs, metrics=metrics)

        return

    def show_validation_results_old(self, path):
        """
        Shows the results of the topic model validation in
        self.validate_topic_models()

        Parameters
        ----------
        path: str
            Path to data
        """

        # ###############
        # Read data files
        # ###############

        # Read the file names in the folder containing the xls reports
        data_dir = os.path.join(path, 'results')
        data_files = sorted(os.listdir(data_dir))

        # Read all result files
        sim, rescale, n_nodes, n_edges, ig, tm_class = [], [], [], [], [], []
        params, df_dict = {}, {}
        # fname_struct = ['corpus', 'sim', 'rescale', 'n_nodes', 'n_edges',
        #                 'ig', 'tm_class']
        for f in data_files:
            if f.endswith('.xls'):
                fname = os.path.splitext(f)[0]
                fname_parts = fname.split('_')

                # Read parameters from the file name
                sim_f = fname_parts[1]
                rescale_f = fname_parts[2]
                n_nodes_f = fname_parts[3]
                n_edges_f = fname_parts[4]
                ig_f = fname_parts[5]
                tm_class_f = '_'.join(fname_parts[6:])

                sim.append(sim_f)
                rescale.append(rescale_f)
                n_nodes.append(n_nodes_f)
                n_edges.append(n_edges_f)
                ig.append(ig_f)
                tm_class.append(tm_class_f)

                # Read report from file
                fpath = os.path.join(data_dir, f)
                df_dict[fname] = pd.read_excel(fpath)
                params[fname] = {
                    'tmc': tm_class_f, 'n': n_nodes_f, 'e': n_edges_f,
                    's': sim_f, 'r': rescale_f, 'i': ig_f}

        # ############
        # Plot results
        # ############

        # Ordered list of parameters.
        sim = sorted(list(set(sim)))
        rescale = sorted(list(set(rescale)))
        n_nodes = sorted(list(set(n_nodes)))
        n_edges = sorted(list(set(n_edges)))
        ig = sorted(list(set(ig)))
        tm_class = sorted(list(set(tm_class)))

        # Get the list of all possiblo dataframe columns to visualize
        cols = []
        for name, df in df_dict.items():
            cols += df.columns.tolist()

        # Remove columns that will not become y-coordinates
        cols = list(set(cols) - {'Unnamed: 0', 'Model', 'No. of topics',
                                 'Number of nodes'})

        # Dictionary of abreviations (for the file names)
        abbreviation = {x: x for x in cols}
        abbreviation.update({'Radius': 'Rad',
                             'Time': 't',
                             'Number of edges': 'ne',
                             'Connected components': 'cc',
                             'Largest component': 'lc',
                             'Relative max CC': 'rc',
                             'Cocitations similarity': 'cs'})

        # The following nested loop is aimed to make multiple plots from xls
        # files inside the same directory. It is actually not needed if all
        # xls files in the given folder have the same parameter values.
        for n in n_nodes:
            for e in n_edges:
                for s in sim:
                    for r in rescale:
                        for i in ig:

                            fnames = [x for x in df_dict if
                                      params[x]['n'] == n and
                                      params[x]['e'] == e and
                                      params[x]['s'] == s and
                                      params[x]['r'] == r and
                                      params[x]['i'] == i]

                            if len(fnames) > 0:

                                for var, ab in abbreviation.items():

                                    fig, ax = plt.subplots()
                                    print(fnames)
                                    # Plot all models with r and n:
                                    for fname in fnames:

                                        tmc = params[fname]['tmc']
                                        x = df_dict[fname]['No. of topics']
                                        y = df_dict[fname][var]
                                        ax.plot(x, y, '.:', label=tmc)

                                    ax.set_xlabel('No. of topics')
                                    ax.set_ylabel(var)
                                    ax.set_title(
                                        f'Nodes: {n}, Edges: {e}, Sim: {s}, '
                                        f'Params: {r + i[-1]}')
                                    ax.legend()
                                    ax.grid()
                                    plt.show(block=False)

                                    # Save figure
                                    out_dir = os.path.join(path, 'figs')
                                    tag = '_'.join(fname_parts[0:5])
                                    fname = f'{tag}_{ab}.png'
                                    if not os.path.exists(out_dir):
                                        os.makedirs(out_dir)
                                    out_path = os.path.join(out_dir, fname)
                                    plt.savefig(out_path)

        return

    def compute_citation_centralities(self, path, n=200):
        """
        Computes all centralities for a given graph, and save in output file
        the top n nodes by centrality

        Parameters
        ----------
        path : str
            Path to snode
        """

        # Create graph object
        logging.info(f'-- Loading GRAPH from {path}')
        label = path.split(os.path.sep)[-1]

        metrics = ['degree', 'centrality', 'betweenness', 'cluster_coef',
                   'pageRank']

        for parameter in metrics:
            # Local graph analysis
            self.SG.local_snode_analysis(label, parameter=parameter)

            # ref_name = self.SG.snodes[label].REF
            df = self.SG.snodes[label].df_nodes.sort_values(
                parameter, axis=0, ascending=False)

            print(f"-- -- Top ranked nodes in {label} by {parameter}")
            print(df.head(10))

            # Save the top n rows
            path2old = os.path.join(self.path2project, self.f_struct['output'],
                                    f'topNaN_{label}_{parameter}.xlsx')
            df.iloc[:n].to_excel(path2old, index=False, encoding='utf-8')
            print(f"-- -- Top {n} saved in {path2old}")

            # Save the top n
            path2out = os.path.join(self.path2project, self.f_struct['output'],
                                    f'top_{label}_{parameter}.xlsx')

            # Remove rows with NaNs. This is aimed for citation graphs
            if 'eid' in df.columns:
                df = df[~pd.isna(df['eid'])]
            df.iloc[:n].to_excel(path2out, index=False, encoding='utf-8')
            print(f"-- -- Top {n} saved in {path2out}")

        # Save
        self.SG.save_supergraph()
        self.SG.deactivate()

        return

    def show_all_citation_centralities(self, path, n=400):
        """
        Shows all available local parameters for a given graph, and save them
        in output files containing the top n nodes for each parameter

        Parameters
        ----------
        path : str
            Path to snode
        """

        # Activate selected snode
        logging.info(f'-- Loading GRAPH from {path}')
        label = path.split(os.path.sep)[-1]
        self.SG.activate_snode(label)
        # ref_name = self.SG.snodes[label].REF

        metrics = ['degree', 'centrality', 'betweenness', 'cluster_coef',
                   'pageRank', 'katz']

        for parameter in metrics:

            if parameter in self.SG.snodes[label].df_nodes.columns:

                logging.info(
                    f"-- -- Computing top nodes in {label} by {parameter}")
                # Sort nodes by parameter value
                df = self.SG.snodes[label].df_nodes.sort_values(
                    parameter, axis=0, ascending=False)

                # Remove rows with NaNs. This is aimed for citation graphs
                if 'eid' in df.columns:
                    df = df[~pd.isna(df['eid'])]

                if parameter == 'cluster_coef':
                    if ('abs_in_degree' in df.columns and 'abs_out_degree' in
                            df.columns):
                        # Filter nodes (only for cluster_coef)
                        logging.info(f"-- -- Filtering nodes with few links")
                        df = df[df['abs_in_degree'] +
                                df['abs_out_degree'] >= 10]
                    else:
                        logging.info(
                            f"No in- and out- degree columns available to " +
                            "filter nodes with few links")

                df.rename(columns={"title": "Título", "pub_year": "Año",
                                   "citation_count": "Citas"}, inplace=True)

                print(df.head(10))

                # Save the top n
                path2out = os.path.join(
                    self.path2project, self.f_struct['output'],
                    f'top_{label}_{parameter}.xlsx')

                df.iloc[:n].to_excel(path2out, index=True, encoding='utf-8')
                print(f"-- -- Top {n} saved in {path2out}")

        # Save
        self.SG.save_supergraph()
        self.SG.deactivate()

        return

    def read_vocabulary(self, vocab_filename):
        """
        Reads a vocabultary from file.

        Parameters
        ----------
        vocab_filename : str
            Path to vocabulary

        Returns
        -------
        vocab_w2id : dict
            Vocabulary dictionary using words as keys {word_i : id_word_i}
        vocab_id2w : dict
            Vocabulary dictionary using word ids as keys {i : word_i}
        """
        vocab_w2id = {}
        vocab_id2w = {}
        with open(vocab_filename, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                vocab_w2id[line.strip()] = i
                vocab_id2w[str(i)] = line.strip()

        return vocab_w2id, vocab_id2w

    def get_keyword_descriptions(self, vocabfile, W, n_keywords=10):
        """
        Returns a description o each row of a weight matrix. Each column of
        the weight matrix is assumed to contain the weights associated to
        a single word specified in a given vocabulary

        Parameters
        ----------
        vocabfile : str
            Path to vocabulary
        W : numpy array (n_words x n)
            Weight matrix
        n_keywords : int, optional (default=10)
            Number of keywords used to describe each row of the weight matrix
        """

        # Read vocabulary
        vocab_w2id, vocab_id2w = self.read_vocabulary(vocabfile)

        # Select keywords
        tpc = range(W.shape[0])
        tpc_descs = []
        for i in tpc:
            # Compute keywords for the i-th row
            words = [vocab_id2w[str(idx2)]
                     for idx2 in np.argsort(W[i])[::-1][0:n_keywords]]
            tpc_descs.append((i, ', '.join(words)))

        return tpc_descs

    def _get_community_labels(self, path2tm, p2c, n_keywords=10):
        """
        Get a list of keywords characterizing the communities

        Parameters
        ----------
        path2tm : str
            Path to the folder containing the corpus informacions
        p2c : dict
            Dictionary project : communities. Each project is identified by its
            reference
        n_keywords : int, optional (default=10)
            Number of keywords used to describe each community

        Returns
        -------
        W : numpy array
            A weight matrix with one row per community and one column per word
            in the vocabulary

        com_keywords : dict
            A dictionary {community : keywords}
        """

        # Configurable parameters:
        ref_col = 'corpusid'

        # Load vocabulary
        corpus = blei.BleiCorpus(
            os.path.join(path2tm, 'PlanEst_AIall-mult.dat'),
            os.path.join(path2tm, 'PlanEst_AI_vocabulary.gensim'))

        # Initialize word_count matrix, containing the number of repetitions of
        # each word in each community.
        ncoms = max(p2c.values()) + 1            # Number of communities
        vocab_size = len(corpus.id2word)         # Vocabulary size
        word_count = np.zeros((ncoms, vocab_size))

        # Map project references to indices
        path2nodenames = os.path.join(path2tm, 'docs_metadata.csv')
        df_allnodes = pd.read_csv(path2nodenames, usecols=[ref_col])

        # This is not much efficient because I should run over the docs in p2c
        # which are only a fraction of the docs in corpus. However, I do not
        # known how to get random access to docs in corpus.
        for doc_id, bow in enumerate(corpus):
            # Get the reference name associated to doc_id
            ref = df_allnodes.loc[doc_id, ref_col]
            if ref in p2c:
                comm_id = int(p2c[ref])
                for word in bow:
                    word_count[comm_id, word[0]] += word[1]

        # Convert word counts to a weight matrix with probabilistic rows
        W = normalize(word_count, axis=1, norm='l1')

        # Downscalign very common terms usually provides better subcorpus
        # descriptions
        if np.min(W) < 1e-12:
            W += 1e-12
        denom = np.reshape((sum(np.log(W)) / ncoms), (vocab_size, 1))
        denom = np.ones((ncoms, 1)).dot(denom.T)
        W = W * (np.log(W) - denom)

        com_keywords = self.get_keyword_descriptions(
            os.path.join(path2tm, 'vocab.txt'), W, n_keywords=n_keywords)
        com_keywords = dict(com_keywords)

        return W, com_keywords




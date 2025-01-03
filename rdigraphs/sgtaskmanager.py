import logging

import pathlib
import os
import shutil
import _pickle as pickle
import numpy as np
import yaml    # Conda install pyyaml

import pandas as pd
import time
import math

# Libraries for Halo visualization
import http.server
import threading
import webbrowser

# Local imports
from rdigraphs.datamanager.datamanager import DataManager
from rdigraphs.supergraph.snode import DataGraph
from rdigraphs.supergraph.supergraph import SuperGraph
from rdigraphs.community_plus.community_plus import CommunityPlus

import gc


class SgTaskManager(object):
    """
    Task Manager for the RDIgraph analyzer.

    The behavior of this class depends on the state of the project, in
    dictionary self.state, with the following entries:

    'isProject'   : If True, project created. Metadata variables loaded
    'configReady' : If True, config file loaded. Datamanager activated.
    """

    def __init__(self, path2project, paths2data={}, path2source=None,
                 metadata_fname='metadata.pkl',
                 config_fname='parameters.yaml', keep_active=False):
        """
        Opens a corpus classification project.

        Parameters
        ----------
        path2project : str
            Path to the project
        paths2data : dict, optional (default={})
            Dictionary of paths
        metadata_fname : str, optional (default='metadata.pkl')
            Name of the metadata file
        config_fname : str, optional (default='parameters.yaml')
            Name of the file containing the configuration variables
        keep_active : bool, optional (default=False)
            If False, graphs are removed from memory (but not from files)
            after the tasks.
        """

        # This is the minimal information required to start with a project
        self.path2project = pathlib.Path(path2project)
        self.path2metadata = self.path2project / metadata_fname
        self.path2config = self.path2project / config_fname
        self.metadata_fname = metadata_fname
        self.path2halo = None    # Path to Halo software

        # This may be required by some methods
        self.path2source = path2source
        if 'topicmodels' in paths2data:
            self.path2tm = paths2data['topicmodels']
        if 'agents' in paths2data:
            self.path2agent = paths2data['agents']
        if 'ACL_models' in paths2data:
            self.path2ACL = paths2data['ACL_models']

        # These are the default file and folder names for the folder
        # structure of the project. It can be modifier by entering other
        # names as arguments of the create or the load method.
        self.f_struct = {
            'import': 'import',               # Folder of input data sources
            'export': 'export',
            'output': 'output',
            'snodes': 'graphs',
            'sedges': 'bigraphs',
            'metagraph': 'metagraph'}

        # State variables that will be laoded from the metadata file when
        # when the project was loaded.
        self.state = {
            'isProject': False,     # True if the project exist.
            'configReady': False,   # True if config file has been processed
            'dbReady': False}       # True if db exists and can be connected

        # Other class variables
        self.keep_active = keep_active
        self.DM = None       # Data manager object
        self.cf = None       # Handler to the config file
        self.db_tables = {}  # List of (empty or not) tables in the database
        self.ready2setup = False  # True after create() or load() are called

        # Other parameters
        self.blocksize = None
        self.useGPU = False

        # Supergraph
        self.SG = None

        return

    def _deactivate(self):
        """
        Deactivates the whole supergraph if the state variable self.keep_active
        is False
        """

        if not self.keep_active:
            self.SG.deactivate()

        return

    def create(self, f_struct=None):
        """
        Creates a RDI graph analysis project.
        To do so, it defines the main folder structure, and creates (or cleans)
        the project folder, specified in self.path2project

        Parameters
        ----------
        f_struct : dict or None, optional (default=None)
            Contains all information related to the structure of project files
            and folders: paths (relative to ppath), file names, suffixes,
            prefixes or extensions that could be used to define other files
            or folders.
            (default names are used when not given)

            If None, default names are given to the whole folder tree
        """

        # This is just to abbreviate
        p2p = self.path2project

        # Check and clean project folder location
        if p2p.is_dir():
            print('Folder {} already exists.'.format(p2p))

            # Remove current backup folder, if it exists
            old_p2p = pathlib.Path(str(p2p) + '_old')
            if old_p2p.is_dir():
                shutil.rmtree(old_p2p)

            # Copy current project folder to the backup folder.
            shutil.move(p2p, old_p2p)
            print(f'Moved to {old_p2p}')

        # Create project folder and subfolders
        self.path2project.mkdir()

        self.update_folders(f_struct)

        # Place a copy of a default configuration file in the project folder.
        # This file should be adapted to the new project settings.
        shutil.copyfile('config/parameters.default.yaml', self.path2config)

        # Update the state of the project.
        self.state['isProject'] = True

        # Save metadata
        self.save_metadata()

        # The project is ready to setup, but the user should edit the
        # configuration file first
        self.ready2setup = True

        # Create empty supergraph
        p = self.path2project / self.f_struct['metagraph']
        p2sn = self.path2project / self.f_struct['snodes']
        p2se = self.path2project / self.f_struct['sedges']
        self.SG = SuperGraph(path=p, path2snodes=p2sn, path2sedges=p2se)

        return

    def update_folders(self, f_struct=None):
        """
        Updates the project folder structure using the file and folder
        names in f_struct.

        Parameters
        ----------
        f_struct: dict or None, optional (default=None)
            Contains all information related to the structure of project files
            and folders: paths (relative to ppath), file names, suffixes,
            prefixes or extensions that could be used to define other files
            or folders.
            (default names are used when not given)

            If None, default names are given to the whole folder tree
        """

        # ######################
        # Project file structure

        # Overwrite default names in self.f_struct dictionary by those
        # specified in f_struct
        if f_struct is not None:
            self.f_struct.update(f_struct)

        # This is just to abbreviate
        p2p = self.path2project

        # Now, we have to do something different with folders and subfolders.
        # for this reason, I define a list specifying which are the main
        # folders. This is not very nice...
        main_folders = ['import', 'export', 'output', 'snodes', 'sedges',
                        'metagraph']
        for d in main_folders:
            path2d = p2p / self.f_struct[d]
            if not path2d.exists():
                path2d.mkdir()

        return

    def save_metadata(self):
        """
        Save metadata into a pickle file
        """

        # Save metadata
        metadata = {'f_struct': self.f_struct, 'state': self.state}
        with open(self.path2metadata, 'wb') as f:
            pickle.dump(metadata, f)

    def load(self, f_struct={}):
        """
        Loads an existing project, by reading the metadata file in the project
        folder.

        It can be used to modify file or folder names, or paths, by specifying
        the new names/paths in the f_struct dictionary.

        Parameters
        ----------
        f_struct: dict or None, optional (default=None)
            Contains all information related to the structure of project files
            and folders: paths (relative to ppath), file names, suffixes,
            prefixes or extensions that could be used to define other files
            or folders.
            (default names are used when not given)

            If None, default names are given to the whole folder tree
        """

        # WARNING FOR THE DEVELOPER: Do not change prints by logging messages
        # because the logger is set after reading the configoration file,
        # once the project folder is stablished.

        # Check and clean project folder location
        if not self.path2project.exists():
            print(f'-- Folder {self.path2project} does not exist. '
                  'You must create the project first')

        # Check metadata file
        elif not self.path2metadata.exists():
            print(
                f'-- ERROR: Metadata file {self.path2metadata} does not'
                '   exist.\n'
                '   This is likely not a project folder. Select another '
                'project or create a new one.')

        else:
            # Load project metadata
            print('-- Loading metadata file...')

            with open(self.path2metadata, 'rb') as f:
                metadata = pickle.load(f)

            # Store state and fils structure
            self.state = metadata['state']

            # The following is used to automatically update any changes in the
            # keys of the self.f_struct dictionary. This will be likely
            # unnecesary once a stable version of the code is reached.
            self.update_folders(metadata['f_struct'])
            # Save updated folder structure in the metadata file.
            self.save_metadata()

            if self.state['configReady']:
                self.ready2setup = True
                self.setup()
                print(
                    f'-- Project {self.path2project} succesfully loaded.')
            else:
                print(
                    f'-- Project {self.path2project} loaded, but '
                    'configuration file could not be activated. \n'
                    'Revise the configuration file and activate it.')

            # Load supergraph
            p = self.path2project / self.f_struct['metagraph']
            p2sn = self.path2project / self.f_struct['snodes']
            p2se = self.path2project / self.f_struct['sedges']
            self.SG = SuperGraph(path=p, path2snodes=p2sn, path2sedges=p2se)

        return

    def set_logs(self):
        """
        Configure logging messages.
        """

        # Log to file and console
        p = self.global_parameters['logformat']
        fpath = self.path2project / p['filename']

        # Configure logging to file
        logging.basicConfig(
            filename=fpath, format=p['file_format'], level=p['file_level'],
            datefmt=p['datefmt'], filemode='w')

        # Clear existing handlers to prevent duplicate logs
        mylogger = logging.getLogger('')
        if mylogger.hasHandlers():
            for handler in mylogger.handlers[:]:
                mylogger.removeHandler(handler)
                handler.close()

        # Define a Handler to write messages to the sys.stderr
        console = logging.StreamHandler()
        # Set level for console
        console.setLevel(p['cons_level'])
        
        # Set the formatter if needed
        console.setFormatter(logging.Formatter(p['cons_format'], datefmt=p['datefmt']))

        # Add the handler to the root logger
        mylogger.addHandler(console)
        mylogger.setLevel(p['file_level'])

        return
    
    def setup(self):
        """
        Set up the classification projetc. To do so:

        (1) Loads the configuration file and initializes the data manager.

        (2) Creates a DB table.
        """

        if self.ready2setup is False:
            # Here, print, and not logging, because the logger has not been
            # set up yet.
            print("---- Error: you cannot setup a project that has not been "
                  "created or loaded")
            return

        with open(self.path2config, 'r') as f:
            self.global_parameters = yaml.safe_load(f)

        # Set up the logging format
        self.set_logs()

        db_params = self.global_parameters['connections']
        self.DM = DataManager(self.path2project, db_params,
                              path2source=self.path2source)

        self.state['dbReady'] = self.DM.dbON

        # Update the project state
        if self.DM.SQL == {} and self.DM.Neo4j is None:
            logging.info("-- No databases connected")
        else:
            logging.info(
                "-- The following databases have been successfully connected:")

            for corpus in self.DM.SQL:
                if self.DM.SQL[corpus].dbON:
                    logging.info(f"-- -- {corpus}")
                    self.db_tables[corpus] = self.DM.SQL[corpus].getTableNames()
                    if self.db_tables[corpus] == []:
                        logging.info("-- -- DB with no tables in the DB")
                    else:
                        all_tables = ', '.join(self.db_tables[corpus])
                        logging.info("      Available tables are: ")
                        logging.info(f"         {all_tables}")

            if self.DM.Neo4j is not None:
                logging.info("-- -- Neo4j")
                self.db_tables['neo4j'] = self.DM.Neo4j.get_db_structure()
                logging.info("      Available components are:")
                node_list = [x for x, y in self.db_tables['neo4j'].items()
                            if y['type'] == 'node']
                edge_list = [x for x, y in self.db_tables['neo4j'].items()
                            if y['type'] == 'relationship']
                logging.info(f"         Nodes: {', '.join(node_list)}")
                logging.info(f"         Edges: {', '.join(edge_list)}")

        self.state['configReady'] = True
        self.blocksize = self.global_parameters['algorithms']['blocksize']
        if 'useGPU' in self.global_parameters['algorithms']:
            self.useGPU = self.global_parameters['algorithms']['useGPU']
        # Path to Halo software for visualization of bipartite graphs
        self.path2halo = self.global_parameters['path2halo']

        # Sace the state of the project.
        self.save_metadata()

        return

    # ##################################
    # Get methods for the menu navigator
    # ##################################
    def get_names_of_SQL_dbs(self):
        """
        Returns the list of available databases
        """

        return list(self.DM.SQL.keys())

    def get_names_of_dataset_tables(self):
        """
        Returns the list of available tables with raw graph data
        """

        return self.DM.get_names_of_dataset_tables()

    def get_sql_table_names(self, graph, db):
        """
        Get tables in the given database

        Parameters
        ----------
        db : str
            Name of the database

        Returns
        -------
        table_names : list of str
            Names of the tables in the database
        """

        if db in self.db_tables:
            table_name = self.DM.SQL[db].getTableNames()
        else:
            table_name = None
            logging.warning('---- Option not available')

        return table_name

    def get_table_atts(self, graph, db, table, *args):
        """
        Get table attributes in the given database

        Parameters
        ----------
        graph : str
            Not used
        db : str
            Name of the database
        table : str
            Name of the table to read attributes

        Returns
        -------
        table_names : list of str
            Names of the tables in the database
        """

        if self.DM.Neo4j is None:
            logging.info("-- Neo4j database is not available")
            return []

        if db in self.db_tables:
            col_names = self.DM.SQL[db].getColumnNames(table)
        else:
            col_names = []
            logging.warning(f'---- Database {db} is not available')

        return col_names

    def get_Neo4J_snodes(self):
        """
        Get snodes in Neo4J db

        Returns
        -------
        snodes : list
            Name of snodes
        """

        if self.DM.Neo4j is None:
            logging.info("-- Neo4j database is not available")
            return []

        gdb_struct = self.DM.Neo4j.get_db_structure()
        snodes = [k for k, v in gdb_struct.items() if v['type'] == 'node']

        return snodes

    def get_Neo4J_sedges(self):
        """
        Get sedges in Neo4J db

        Returns
        -------
        sedges : list
            Name of sedges
        """

        if self.DM.Neo4j is None:
            logging.info("-- Neo4j database is not available")
            return []

        gdb_struct = self.DM.Neo4j.get_db_structure()
        sedges = [k for k, v in gdb_struct.items()
                  if v['type'] == 'relationship']

        return sedges

    def get_attributes(self, path):
        """
        Returns attributes of the graph in path.

        Parameters
        ----------
        path : str
            Path to data

        Returns
        -------
        c : list
            List of attributes
        """

        # Create graph object
        label = pathlib.Path(path).name

        return self.SG.get_attributes(label)
        # return self.get_communities(path)

    def _get_att_subset(self, path, att_class):
        """
        Returns the list of graph attributes that are available for a given
        graph and from a given group.

        Parameters
        ----------
        path : str or pathlib.Path
            Path to data
        param_group : str
            Name of the class of attributes to get

        Returns
        -------
        c : list
            List of attributes available from the selected class
        """

        # Create graph object
        label = pathlib.Path(path).name
        metadata = self.SG.get_metadata(label)

        atts = None
        if att_class in metadata:
            atts = list(metadata[att_class].keys())

        return atts

    def get_communities(self, path):
        """
        Returns community models of the graph in path.

        Parameters
        ----------
        path : str
            Path to data

        Returns
        -------
        atts : list
            List of community models computed for the graph
        """

        return self._get_att_subset(path, 'communities')

    def get_local_features(self, path):
        """
        Returns the local features available for the graph in path.

        Parameters
        ----------
        path : str
            Path to data

        Returns
        -------
        atts : list
            List of local features computed for the graph
        """

        return self._get_att_subset(path, 'local_features')

    def get_source_atts(self, path, *args):
        """
        Returns attributes of the source snode for the bipartite graph in path

        Parameters
        ----------
        path : str or pathlib.Path
            Path to data
        args : tuple, optional
            Possible extra arguments that are ignored

        Returns
        -------
        atts : list
            List of available attributes at the selected snode
        """

        # Create graph object
        e_label = pathlib.Path(path).name

        # Load source snode
        s_label, t_label = self.SG.get_terminals(e_label)
        atts = self.SG.get_attributes(s_label)

        return atts

    def get_target_atts(self, path, *args):
        """
        Returns attributes of the target snode for the bipartite graph in path

        Parameters
        ----------
        path : str or pathlib.Path
            Path to data
        args : tuple, optional
            Possible extra arguments that are ignored

        Returns
        -------
        atts : list
            List of available attributes at the selected snode
        """

        # Create graph object
        e_label = pathlib.Path(path).name

        # Load source snode
        s_label, t_label = self.SG.get_terminals(e_label)
        atts = self.SG.get_attributes(t_label)

        return atts

    def get_graphs_with_features(self, *args):
        """
        Returns a list of the available snodes with saved attributes
        """

        return self.SG.get_snodes_with_features()

    # ############
    # SQL db views
    # ############
    def showSDBdata(self, option):
        """
        Print a general overview of the selected source (SQL) database

        Parameters
        ----------
        option : str
            Name of the db table
        """

        if option in self.db_tables:
            self.DM.SQL[option].showDBview()
        else:
            logging.info('---- Option not available. Likely, the DB could not '
                         'be successfully connected')
        return

    def showGDBdata(self, option, snodes, sedges):
        """
        Print a general overview of the selected database

        Parameters
        ----------
        option : str
            Name of the db table
        snodes : list
            List of snodes
        sedges : list
            List of sedges
        """

        if option == 'Neo4j':
            # Overview the graph database
            self.DM.Neo4j.showDBview()
        elif option in snodes:
            # Overview the selected snode
            attribs = self.DM.Neo4j.properties_of_label(option)
            print("---- Supernode ", option)
            print("          Attributes: {}".format(attribs))
        elif option in sedges:
            # Overview the selected snode
            attribs = self.DM.Neo4j.properties_of_relationship(option)
            print("---- Superedge ", option)
            print("          Attributes: {}".format(attribs))
        else:
            print('---- Option not available')
        return

    # ################
    # Neo4J management
    # ################
    def show_Neo4J(self):
        """
        Print a general overview of the whole database
        """

        if self.DM.Neo4j is None:
            logging.info("-- Neo4j database is not available")
            return

        self.DM.Neo4j.showDBview()

        return

    def show_Neo4J_snode(self, snode):
        """
        Print a general overview of the selected snode

        Parameters
        ----------
        snode : str
            Name of the snode
        """

        if self.DM.Neo4j is None:
            logging.info("-- Neo4j database is not available")
            return

        # Overview the selected snode
        attribs = self.DM.Neo4j.properties_of_label(snode)
        logging.info(f"---- Supernode {snode}")
        logging.info(f"          Attributes: {attribs}")
        return

    def show_Neo4J_sedge(self, sedge):
        """
        Print a general overview of the selected sedge

        Parameters
        ----------
        sedge : str
            Name of the sedge
        """

        if self.DM.Neo4j is None:
            logging.info("-- Neo4j database is not available")
            return

        # Overview the selected snode
        attribs = self.DM.Neo4j.properties_of_relationship(sedge)
        print(f"---- Superedge {sedge}")
        print(f"          Attributes: {attribs}")
        return

    def getGDBstruct(self):
        """
        Get structure of the graph database

        Returns
        -------
        snodes : list
            Name of snodes
        sedges : list
            Name of sedges
        """

        if self.DM.Neo4j is None:
            logging.info("-- Neo4j database is not available")
            return

        gdb_struct = self.DM.Neo4j.get_db_structure()
        snodes = [k for k, v in gdb_struct.items() if v['type'] == 'node']
        sedges = [k for k, v in gdb_struct.items()
                  if v['type'] == 'relationship']

        return snodes, sedges

    def resetGDBdata(self, option, snodes, sedges):
        """
        Reset (drop and create emtpy) tables from the database.

        Parameters
        ----------
        option : str
            Selected node or edge to reset
        snodes : list
            List of available nodes
        sedges :
            List of available sedges
        """

        if self.DM.Neo4j is None:
            logging.info("-- Neo4j database is not available")
            return

        if option == 'Neo4j':
            self.DM.Neo4j.resetDB()
            logging.info("---- Graph database has been reset")
        elif option in snodes:
            self.DM.Neo4j.dropNodes(option)
            logging.info(f"---- Nodes of type {option} have been reset.")
        elif option in sedges:   # (option in sedges)
            self.DM.Neo4j.drop_relationship(option)
            logging.info(f"---- Edges of type {option} have been reset.")

        return

    def reset_Neo4J(self):
        """
        Reset the whole database
        """

        if self.DM.Neo4j is None:
            logging.info("-- Neo4j database is not available")
        else:
            self.DM.Neo4j.resetDB()
            logging.info("---- Graph database have been reset")

        return

    def reset_Neo4J_snode(self, snode):
        """
        Reset (drop and create emtpy) tables from the database.

        Parameters
        ----------
        snode : str
            Selected node or edge to reset
        """

        if self.DM.Neo4j is None:
            logging.info("-- Neo4j database is not available")
        else:
            self.DM.Neo4j.dropNodes(snode)
            logging.info(f"---- Nodes of type {snode} have been reset.")

        return

    def reset_Neo4J_sedge(self, sedge):
        """
        Reset (drop and create emtpy) tables from the database.

        Parameters
        ----------
        sedge : str
            Selected node or edge to reset
        """

        if self.DM.Neo4j is None:
            logging.info("-- Neo4j database is not available")
        else:
            self.DM.Neo4j.drop_relationship(sedge)
            logging.info(f"---- Edges of type {sedge} have been reset.")

        return

    def export_graph_2_neo4J(self, path2graph, label_nodes):
        """
        Export graphs from csv files to neo4J.

        Parameters
        ----------
        path2graph : str or pathlib.Path
            Path to graph
        """

        if self.DM.Neo4j is None:
            logging.info("-- Neo4j database is not available")
            return

        # Load graph object
        graph_name = pathlib.Path(path2graph).name
        dg = DataGraph(path=path2graph, label=graph_name)

        # Call the export method in DM
        col_ref_nodes = dg.REF
        label_edges = graph_name
        self.DM.Neo4j.export_graph(label_nodes, dg.path2nodes, col_ref_nodes,
                                   label_edges, dg.path2edges)

        return

    def export_bigraph_2_neo4J(self, path2graph, label_source, label_target):
        """
        Export bipartite graph from csv files to neo4J.

        Parameters
        ----------
        path2graph : str or pathlib.Path
            Path to graph
        """

        if self.DM.Neo4j is None:
            logging.info("-- Neo4j database is not available")
            return

        # Load bigraph object
        graph_name = pathlib.Path(path2graph).name
        dg = DataGraph(path=path2graph, label=graph_name)

        # Call the export method in DM
        label_nodes = (label_source, label_target)
        col_ref_nodes = dg.REF
        label_edges = graph_name
        self.DM.Neo4j.export_graph(label_nodes, dg.path2nodes, col_ref_nodes,
                                   label_edges, dg.path2edges)

        return

    # ###########
    # Load graphs
    # ###########
    def import_snode_from_table(
            self, table_name, n0=0, sampling_factor=1, params={},
            load_feather=False, save_feather=False):
        """
        Import a graph from a table containing node names, attributes and
        (possibly) embeddings

        Parameters
        ----------
        table_name : str
            Name of the folder containing the table of nodes
        n0 : int or float, optional (default=0).
            Number of nodes. If 0 all nodes are imported. If 0 < n0 < 1, this
            is the fraction of the total. If n0 > 1, number of nodes
        sampling_factor : float, optional (default=1)
            Sampling factor.
        params : dict, optional (default={})
            Dictionary of parameters (specific of the dataset)
        load_feather : bool, optional (default=False)
            If True, data are imported from a feather file, if available
        save_feather : bool, optional (default=False)
            If True, dataframe is saved to a feather file

        Notes
        -----
        The final no. of nodes is the minimum between n0 and the result of
        applyng the sampling factor over the whole dataset
        Both parameters (n0 and sampling_factor) may be needed for large
        datasets: the sampling factor can be used to avoid reading all data
        dataset from the parquet files, while n0 is used as the true target
        number of nodes
        """

        # #########
        # LOAD DATA

        # Take the path to the data from the config file, if possible
        # path = self.global_parameters['source_data'][corpus]
        logging.info(f'-- Loading dataset {table_name}')

        df_nodes, source_metadata = self.DM.import_graph_data_from_tables(
            table_name, sampling_factor, params=params)

        # Load labels of features from the corpus metadata
        T_labels = None
        if 'feature_labels' in source_metadata:
            if 'n_topics' in params:
                T_labels = (
                    source_metadata['feature_labels'][params['n_topics']])
            else:
                T_labels = source_metadata['feature_labels']

        # Take feature matrix from embeddings
        T = np.array(df_nodes['embeddings'].tolist())

        # Remove embeddings from df_nodes
        df_nodes.drop(columns='embeddings', inplace=True)

        # nodes = df_nodes['id'].tolist()

        # Take a random sample
        if n0 <= 0:
            n_gnodes = len(df_nodes)
        elif n0 < 1:
            n_gnodes = int(n0 * len(df_nodes))
        else:
            n_gnodes = int(n0)

        # ########################
        # LOAD DATA INTO NEW SNODE

        # Create datagraph with the full feature matrix
        self.SG.makeSuperNode(label=table_name, nodes=df_nodes, T=T,
                              save_T=True, T_labels=T_labels)
        self.SG.sub_snode(table_name, n_gnodes, ylabel=table_name,
                          sampleT=True, save_T=True)

        # #####################
        # SHOW AND SAVE RESULTS

        logging.info(f'Zero-edge graph loaded with {n_gnodes} nodes')
        # Save graph: nodes and edges
        self.SG.save_supergraph()

        # Reset snode. This is to save memory.
        self._deactivate()

        return

    def import_nodes_and_model(self, path, n0=0):
        """
        Import nodes from a csv and npz files.

        Parameters
        ----------
        path : str or pathlib.Path
            Path to the model
        n0 : int or float, optional (default=0).
            Number of nodes. If 0 all nodes are imported
        """

        # Name of the selected corpus topic model
        graph_name = pathlib.Path(path).name

        # #########
        # LOAD DATA

        # Take the path to the data from the config file, if possible
        # path = self.global_parameters['source_data'][corpus]
        logging.info(f'-- Loading data matrix from {path}')

        # Load data matrix
        data, df_nodes = self.DM.readCoordsFromFile(
            path, sparse=True, ref_col='corpusid')
        T = data['thetas']
        nodes = df_nodes['corpusid'].tolist()

        # Take a random sample
        if n0 <= 0:
            n_gnodes = len(nodes)
        elif n0 < 1:
            n_gnodes = int(n0 * len(nodes))
        else:
            n_gnodes = int(n0)

        # ########################
        # LOAD DATA INTO NEW SNODE

        # Create datagraph with the full feature matrix
        self.SG.makeSuperNode(label=graph_name, nodes=nodes, T=T, save_T=True)
        self.SG.sub_snode(graph_name, n_gnodes, ylabel=graph_name,
                          sampleT=True, save_T=True)

        # #####################
        # SHOW AND SAVE RESULTS

        logging.info(f'Zero-edge graph loaded with {n_gnodes} nodes')
        # Save graph: nodes and edges
        self.SG.save_supergraph()

        # Reset snode. This is to save memory.
        self._deactivate()

        return

    def import_co_citations_graph(self):
        """
        Loads a co-citations-graph (only for ACL) and saves it in a new snode
        """

        # Corpus and model specification
        graph_name = 'cocitations'

        # #######################
        # Load co-citations graph
        t0 = time.time()
        logging.info(
            '-- -- Loading co-citations graph. This may take some time ...')
        df = self.DM.SQL['Pu'].readDBtable(
            'outCitations_graph', limit=None, selectOptions=None,
            filterOptions=None, orderOptions=None)

        # ##########
        # Parameters

        # Take list of sources, targets and weights from the dataframe
        source_nodes = df.ACLid1.tolist()
        target_nodes = df.ACLid2.tolist()
        weights = df.weight.tolist()
        nodes = sorted(list(set(source_nodes + target_nodes)))

        # Create snode, add edges and save
        self.SG.makeSuperNode(label=graph_name, nodes=nodes)
        self.SG.snodes[graph_name].set_edges(source_nodes, target_nodes,
                                             weights)
        self.SG.save_supergraph()
        self._deactivate()

        # Compute submatrix of co-citations
        logging.info(f"-- Co-citations graphp saved in {time.time()-t0} secs")

        return

    def import_SCOPUS_citations_subgraph(self, type_of_graph):
        """
        Loads a citations subgraph from table 'citations' in SCOPUS SQL
        database

        The subgraph contains the nodes with attributes in table 'document'
        form the same databse.

        Parameters
        ----------
        type_of_graph : str {'undirected', 'cite_to', 'cited_by'}
            Type of graph
        """

        t0 = time.time()

        # Corpus and model specification
        if type_of_graph == 'undirected':
            graph_name = 'citationS'
        elif type_of_graph == 'cite_to':
            graph_name = 'citationSto'
        elif type_of_graph == 'cited_by':
            graph_name = 'citationSby'
        else:
            logging.error('-- Unknown type of graph. Options "undirected" ',
                          '"cite_to" and "cited_by" only')
            exit()

        # ############################
        # Read data from SQL databases
        logging.info("-- Importing SCOPUS citations from database...")
        col_ref = 'eid'
        # Attributes and their fillna values. This is to replaces nan values,
        # that cause some problems to graph dataframes.
        fill_values = {'title': ' ', 'doi': ' ', 'pub_year': 0,
                       'citation_count': 0}
        atts = list(fill_values.keys())

        # Import graph data
        if type_of_graph in ['undirected', 'cite_to']:
            # undirected graphs are loaded as cite_to, but they could be
            # loaded as cited_by indistinctly
            nodes, df_atts, source_nodes, target_nodes = \
                self.DM.load_SCOPUS_citations_with_atts(
                    col_ref, atts, 'cite_to')
        else:  # It must be 'cited_by'
            nodes, df_atts, source_nodes, target_nodes = \
                self.DM.load_SCOPUS_citations_with_atts(
                    col_ref, atts, 'cited_by')

        # ############
        # Create snode
        logging.info("-- Building graph from citation data ...")
        self.SG.makeSuperNode(label=graph_name, nodes=nodes,
                              edge_class='directed')
        self.SG.snodes[graph_name].add_attributes('eid', df_atts, fill_values)
        self.SG.snodes[graph_name].set_edges(source_nodes, target_nodes)

        # ##########
        # Save graph
        self.SG.save_supergraph()
        self._deactivate()

        # Compute submatrix of co-citations
        logging.info(f"-- Citations graph saved in {time.time()-t0} secs")

        return

    def import_SCOPUS_citations_graph(self, type_of_graph):
        """
        Loads a citations graph from table citations in SCOPUS SQL database

        The graph is not restricted to the docs with Spanish authors. It
        includes all items in the citations table (about 5.8 M nodes).

        No attributes are included, because the SCOPUS database contains
        attributes from a small subset of papers.

        Parameters
        ----------
        type_of_graph : str {'undirected', 'cite_to', 'cited_by'}
            Type of graph
        """

        t0 = time.time()

        # Corpus and model specification
        graph_name = 'citationL'
        # Corpus and model specification
        if type_of_graph == 'undirected':
            graph_name = 'citationL'
        elif type_of_graph == 'cite_to':
            graph_name = 'citationLto'
        elif type_of_graph == 'cited_by':
            graph_name = 'citationLby'
        else:
            logging.error('-- Unknown type of graph. Options "undirected" ',
                          '"cite_to" and "cited_by" only')
            exit()

        # ############################
        # Read data from SQL databases
        logging.info("-- Importing SCOPUS citations from database...")
        col_ref = 'eid'
        # Attributes and their fillna values. This is to replaces nan values,
        # that cause some problems to graph dataframes.
        fill_values = {'title': ' ', 'doi': ' ', 'pub_year': 0,
                       'citation_count': 0}
        atts = list(fill_values.keys())

        # Import graph data
        if type_of_graph in ['undirected', 'cite_to']:
            nodes, df_atts, source_nodes, target_nodes = \
                self.DM.load_SCOPUS_citations_all(col_ref, atts, 'cite_to')
        else:  # it must be 'cited_by' here
            nodes, df_atts, source_nodes, target_nodes = \
                self.DM.load_SCOPUS_citations_all(col_ref, atts, 'cited_by')

        # ############
        # Create snode
        logging.info("-- Building graph from citation data ...")
        mode = {'cite_to': 'directed', 'cited_by': 'directed',
                'undirected': 'undirected'}
        self.SG.makeSuperNode(label=graph_name, nodes=nodes,
                              edge_class=mode[type_of_graph])
        self.SG.snodes[graph_name].add_attributes('eid', df_atts, fill_values)
        self.SG.snodes[graph_name].set_edges(source_nodes, target_nodes)

        # ##########
        # Save graph
        self.SG.save_supergraph()
        self._deactivate()

        # Compute submatrix of co-citations
        logging.info(f"-- Citations graph saved in {time.time()-t0} secs")

        return

    def generate_minigraph(self):
        """
        Generates a hand-made minigraph for testing purposes.
        """

        # Create snode, add edges and save
        label = 'minigraph'
        self.SG.makeSuperNode(label=label, edge_class='directed')

        # Make nodes
        chars = 'ABCDEFGH'   # IJKLMNOPQRSTUVWXYZ'
        for c in chars:
            self.SG.snodes[label].add_single_node(c)

        # Make links
        self.SG.snodes[label].add_single_edge('A', 'B', 1)
        self.SG.snodes[label].add_single_edge('A', 'C', 1)
        self.SG.snodes[label].add_single_edge('D', 'C', 1)
        self.SG.snodes[label].add_single_edge('F', 'C', 1)
        self.SG.snodes[label].add_single_edge('G', 'C', 1)
        self.SG.snodes[label].add_single_edge('D', 'E', 1)
        self.SG.snodes[label].add_single_edge('E', 'G', 1)

        self.SG.save_supergraph()
        self._deactivate()

        return

    def import_node_atts(self, path, dbname, table, att, att_ref):
        """
        Load attributes from a given table from a SQL database and add them to
        a given snode.

        Parameters
        ----------
        path : str or pathlib.Path
            Path to the graph to add the new attribute
        db : str
            Type of the database storing the data
        table : str
            Name of the table in the given db that contains tue attribute
        att : str
            Name of the attribute
        att_ref : str
            Name of the attribute containing the node identifier
        """

        # Read attributes from db
        # The use of things like f'`{att}`' below is to avoid an error if the
        # att name has an space
        # att4sql = [f'`{att_ref}`', f'`{att}`']
        if isinstance(att, str):
            att4sql = f'`{att_ref}`, `{att}`'
        else:
            att_str = [f"`{a}`" for a in att]
            att4sql = f"`{att_ref}`, " + ", ".join(att_str)

        logging.info("-- Importing node attributes from database")
        att_values = self.DM.SQL[dbname].readDBtable(
            table, limit=None, selectOptions=att4sql, filterOptions=None,
            orderOptions=None)

        # Create graph object
        graph_name = pathlib.Path(path).name
        self.SG.add_snode_attributes(graph_name, att_ref, att_values)

        # Save modified snode
        self.SG.save_supergraph()
        # Deactivate snode
        self._deactivate()
        logging.info(f'-- Attribute {att} loaded into graph {graph_name}')

        return

    def remove_snode_attributes(self, path, att):
        """
        Load attributes from a given table from a SQL database and add them to
        a given snode.

        Parameters
        ----------
        path : str or pathlib.Path
            Path to the graph where the attribute must be removed
        att : str
            Name of the attribute
        """

        # Create graph object
        graph_name = pathlib.Path(path).name

        # In the current version of the SuperGraph class, snode activation must
        # be done before calling to the snode method. Maybe I should consider
        # activation inside the methods
        self.SG.remove_snode_attributes(graph_name, att)

        # Save modified snode
        self.SG.save_supergraph()
        # Deactivate snode
        self._deactivate()
        logging.info(f'-- Attribute {att} removed from graph {graph_name}')

        return

    # ##############################
    # Supergraph reading and edition
    # ##############################
    def show_SuperGraph(self):
        """
        Show current supergraph structure
        """

        self.SG.describe()

        return

    def show_snode(self, path2snode):
        """
        A quick preview of a supernode.

        Parameters
        ----------
        path2snode : str or pathlib.Path
            Path to the supernode
        """

        label = pathlib.Path(path2snode).name
        self.SG.activate_snode(label)
        self.SG.snodes[label].pprint(10)
        self._deactivate()

        return

    def show_sedge(self, path2sedge):
        """
        A quick preview of a superedge.

        Parameters
        ----------
        path2sedge : str or pathlib.Path
            Path to the superedge
        """

        label = pathlib.Path(path2sedge).name
        self.SG.activate_sedge(label)
        self.SG.sedges[label].pprint()
        self._deactivate()

        return

    def reset_snode(self, path):
        """
        Reset snode in path

        Parameters
        ----------
        path : str or pathlib.Path
            Path to snode
        """

        label = pathlib.Path(path).name
        self.SG.drop_snode(label)
        logging.info(f'---- Graph {label} has been removed.')

        return

    def reset_sedge(self, path):
        """
        Reset sedge in path

        Parameters
        ----------
        path : str or pathlib.Path
            Path to sedge
        """

        label = pathlib.Path(path).name
        self.SG.drop_sedge(label)

        logging.info(f'---- Bigraph {label} has been removed.')

        return

    # ###########
    # Graph tools
    # ###########
    def subsample_graph(self, path, mode, n0):
        """
        Subsample graph

        Parameters
        ----------
        path : str
            Path to graph
        mode : str
            If 'newgraph', create a new snode with the subgraph
        n0 : int
            Target number of nodes
        """

        # Create graph object
        graph_name = pathlib.Path(path).name
        if not self.SG.is_active_snode(graph_name):
            self.SG.activate_snode(graph_name)

        if mode == 'newgraph':
            new_graph_name = f'{graph_name}_{n0}'
        else:
            new_graph_name = graph_name

        # It the original graph has a feature matrix, it is sampled too.
        sampleT = self.SG.snodes[graph_name].T is not None
        # If the original feature matrix has been save, it is saved too
        save_T = self.SG.snodes[graph_name].save_T
        self.SG.sub_snode(graph_name, n0, new_graph_name, sampleT=sampleT,
                          save_T=save_T)

        if mode == 'newgraph':
            # Remove the original graph from memory to avoid saving it.
            self.SG.deactivate_snode(graph_name)
        self.SG.save_supergraph()
        logging.info(f'---- Graph {new_graph_name} saved with '
                     f'{self.SG.snodes[new_graph_name].n_nodes} nodes and '
                     f'{self.SG.snodes[new_graph_name].n_edges} edges')
        self._deactivate()

        return

    def filter_edges(self, path, th):
        """
        Subsample graph from threshold

        Parameters
        ----------
        path : str or pathlib.Path
            Path to graph
        th : str
            Threshold. Edges with smaller weight are removed.
        """

        # Create graph object
        graph_name = pathlib.Path(path).name

        self.SG.filter_edges_from_snode(graph_name, th)

        self.SG.save_supergraph()
        self._deactivate()

        return

    def largest_community_subgraph(self, path, comm):
        """
        Subsample graph taking the nodes from the largest community.

        Parameters
        ----------
        path : str or pathlib.Path
            Path to graph
        comm : str
            Name of the community
        """

        # Create graph object
        graph_name = pathlib.Path(path).name
        new_graph_name = f'sub_{graph_name}{comm}'

        att = comm
        value = 0
        self.SG.sub_snode_by_value(graph_name, att, value,
                                   ylabel=new_graph_name)

        # Remove the original graph from memory to avoid saving it.
        self.SG.deactivate_snode(graph_name)
        self.SG.save_supergraph()
        logging.info(f'---- Graph {new_graph_name} saved with '
                     f'{self.SG.snodes[new_graph_name].n_nodes} nodes and '
                     f'{self.SG.snodes[new_graph_name].n_edges} edges')
        self._deactivate()

        return

    def remove_isolated_nodes(self, path):
        """
        Remove isolated nodes

        Parameters
        ----------
        path : str or pathlib.Path
            Path to snode
        """

        # Create graph object
        graph_name = pathlib.Path(path).name

        # Remove isolated nodes
        self.SG.remove_isolated_nodes(graph_name)
        # Save modified snode
        self.SG.save_supergraph()
        # Deactivate snode
        self._deactivate()

        return

    def import_agents(self, path2tables, path2snode):
        """
        Import agents

        Parameters
        ----------
        path2tables : str or pathlib.Path
            Path to tables
        path2snode : str
            Path to snode
        """

        # Source node
        s_label = pathlib.Path(path2snode).name

        # Load csv_files
        path2_p2r = path2tables / 'researcher_project.csv'

        df_p2r = pd.read_csv(path2_p2r)    # , usecols=['corpusid'])
        source_nodes = list(df_p2r['REFERENCIA'])
        target_nodes = list(df_p2r['disambiguated_id'])
        edges = list(zip(source_nodes, target_nodes))

        # Generate new snode and sedge from the selected attribute
        t_label = 'R_' + s_label

        # Check if a snode exists with the same name than the target snode.
        # to avoid name collision. This is to permit several agent graphs,
        # so as to generate different agent graphs from the same data source.
        check_is_snode = True
        while check_is_snode:
            if self.SG.is_snode(t_label):
                t_label = 'R' + t_label
            else:
                check_is_snode = False

        e_label = s_label + '_2_' + t_label
        self.SG.snode_from_edges(s_label, edges, target=t_label,
                                 e_label=e_label)

        # Deactivate s_label, that does not need to be saved
        self.SG.deactivate_snode(s_label)
        # Save active snodes and sedges
        self.SG.save_supergraph()
        # Clean memory:
        self._deactivate()

        return

    def disambiguate_node(self):
        """
        Disambiguate a given node from a given graph based on the topological
        structure of the related graphs and bigraphs in the supergraph

        Parameters
        ----------
        path : str
            Path to snode
        """

        # Modularity threshold
        th = 0.9

        # Ask for node name:
        node_name = input('\n-- Write the node name to disambigate: ')

        bgs, total_score = self.SG.disambiguate_node(node_name)

        print("-- Summary of scores:")
        print(bgs)

        if total_score > th:
            logging.info(f'-- Node {node_name} should be split')

        return

    def label_nodes_from_feature_labels(self, path):
        """
        Reads feature labels from the corpus metadata and assigns labels to
        nodes with feature vector according to their dominating features

        Parameters
        ----------
        path : str
            Path to snode
        """

        # Get snode name
        graph = pathlib.Path(path).name

        self.SG.label_nodes_from_features(graph)

        # Save modified snode
        self.SG.save_supergraph()
        # Deactivate snode
        self._deactivate()

        return

    # ###############
    # Graph inference
    # ###############
    def equivalence_graph(self, path):
        """
        This method manages the equivalence graph, which is a graph that
        connects all nodes with the same feature vector into its equivalence
        class.

        Parameters
        ----------
        path : str or pathlib.Path
            Path to snode
        """

        s_label = pathlib.Path(path).name
        t_label = f'eq_{s_label}'
        e_label = s_label + '_2_' + t_label

        # #########
        # LOAD DATA

        # Take the path to the data from the config file, if possible
        # path = self.global_parameters['source_data'][corpus]
        logging.info(f'-- Loading data matrix from {path}')

        # Load data matrix
        data, df_nodes = self.DM.readCoordsFromFile(path, sparse=True)
        T = data['thetas']
        nodes = df_nodes['corpusid'].tolist()

        # #################
        # MAKE SOURCE SNODE

        # The source snode is a graph (with no edges) containing as many nodes
        # as rows in the data matrix.

        # Create datagraph with the full feature matrix
        self.SG.makeSuperNode(s_label, nodes=nodes, T=T)
        self.SG.save_supergraph()

        # Generate new snode and sedge from the selected attribute
        self.SG.snode_from_eqs(s_label, target=t_label, e_label=e_label)
        self.SG.save_supergraph()

        # Clean memory:
        self._deactivate()

        return

    def infer_eq_simgraph(self, path, sim, n0=0, epn=10):
        """
        This method manages the generation of an equivalence similarity
        (semantic) graph.

        It is similar to a concatenation of self.equivalence_graph() (to
        transform the original feature matrix into the reduced matrix without
        row repetitions) and infer_sim_graph() (to compute the similarity
        graph)

        Parameters
        ----------
        path : str
            Path to the model
        sim : str
            Similarity measure
        n0 : int or float, optional (default=0).
            Number of nodes. If 0 all nodes are imported. If 0 < n0 < 1,
            this is the fraction of the total. If n0 > 1, number of nodes
        epn : int or float, optional (default=0).
            Number of edges per node.
        """

        # Name of the selected corpus topic model
        s_label = pathlib.Path(path).name
        t_label = f'eq_{s_label}'            # Name of the equivalent snode
        e_label = s_label + '_2_' + t_label  # Name of the connecting sedge

        # #########
        # LOAD DATA

        # Take the path to the data from the config file, if possible
        # path = self.global_parameters['source_data'][corpus]
        logging.info(f'-- Loading data matrix from {path}')

        # Load data matrix
        data, df_nodes = self.DM.readCoordsFromFile(
            path, sparse=True, ref_col='corpusid')
        T = data['thetas']
        nodes = df_nodes['corpusid'].tolist()

        # Take a random sample
        if n0 <= 0:
            n_gnodes = len(nodes)
        elif n0 < 1:
            n_gnodes = int(n0 * len(nodes))
        else:
            n_gnodes = int(n0)

        #
        logging.info(f'Topic matrix loaded with {n_gnodes} nodes')

        n_edges = int(epn * n_gnodes)

        # #################
        # MAKE SOURCE SNODE

        # The source snode is a graph (with no edges) containing as many nodes
        # as rows in the data matrix.

        # Create datagraph with the full feature matrix
        self.SG.makeSuperNode(label=s_label, nodes=nodes, T=T)
        self.SG.save_supergraph()

        # Generate new snode and sedge from the selected attribute
        self.SG.snode_from_eqs(s_label, target=t_label, e_label=e_label)
        self.SG.save_supergraph()

        # Clean memory:
        if not self.keep_active:
            self.SG.deactivate_snode(s_label)
            self.SG.deactivate_sedge(t_label)

        # ########################
        # COMPUTE SIMILARITY GRAPH

        # Create datagraph with the full feature matrix
        self.SG.sub_snode(t_label, n_gnodes, ylabel=t_label, sampleT=True)
        self.SG.computeSimGraph(
            t_label, n_edges=n_edges, n_gnodes=n_gnodes, similarity=sim,
            g=1, blocksize=self.blocksize, useGPU=self.useGPU)

        # #####################
        # SHOW AND SAVE RESULTS

        # Log some results
        # md = dg.metadata['graph']
        md = self.SG.snodes[t_label].metadata
        logging.info(f"-- -- Similarity measure: {md['edges']['metric']}")
        logging.info(f"-- -- Number of nodes: {md['nodes']['n_nodes']}")
        logging.info(f"-- -- Number of edges: {md['edges']['n_edges']}")
        logging.info(f"-- -- Average neighbors per node: "
                     f"{md['edges']['neighbors_per_sampled_node']}")
        logging.info(f"-- -- Density of the similarity graph: "
                     f"{100 * md['edges']['density']} %")

        # Save graph: nodes and edges
        self.SG.save_supergraph()
        # Reset snode. This is to save memory.
        self._deactivate()

        return

    def infer_sim_graph(self, path2snode, sim, n0=0, n_epn=10):
        """
        This method manages the generation of similarity (semantic) graphs.

        Parameters
        ----------
        path2snode : str
            Path to the snode
        sim : str
            Similarity measure
        n0 : int or float of None, optional (default=0)
            Number of nodes. If 0, all nodes are taken.
            If 0 < n0 < 1, it is the fraction of the total no. of nodes
        n_epn : int, optional (default=10)
            Average number of edges per node.
        """

        # Name of the graph
        graph_name = pathlib.Path(path2snode).name
        self.SG.activate_snode(graph_name)
        n_nodes = self.SG.snodes[graph_name].n_nodes
        n_edges = int(n_epn * n_nodes)

        # ########################
        # COMPUTE SIMILARITY GRAPH

        self.SG.computeSimGraph(
            graph_name, n_edges=n_edges, similarity=sim, g=1,
            blocksize=self.blocksize, useGPU=self.useGPU, tmp_folder=None,
            save_every=20_000_000, verbose=True)

        # #####################
        # SHOW AND SAVE RESULTS

        # Log some results
        md = self.SG.snodes[graph_name].metadata
        logging.info(f"-- -- Similarity measure: {md['edges']['metric']}")
        logging.info(f"-- -- Number of nodes: {md['nodes']['n_nodes']}")
        logging.info(f"-- -- Number of edges: {md['edges']['n_edges']}")
        logging.info(f"-- -- Average neighbors per node: "
                     f"{md['edges']['neighbors_per_sampled_node']}")
        logging.info(f"-- -- Density of the similarity graph: "
                     f"{100 * md['edges']['density']} %")

        gc.collect()

        # Save graph: nodes and edges
        self.SG.save_supergraph()

        # Reset snode. This is to save memory.
        self._deactivate()

        return

    def import_and_infer_sim_graph(self, path, sim, n0=0, n_epn=10,
                                   label=None):
        """
        This method manages the generation of similarity (semantic) graphs.

        Parameters
        ----------
        path : str or pathlib.Path
            Path to the model
        sim : str
            Similarity measure
        n0 : int or float, optional (default=0)
            Number of nodes. If 0, all nodes are taken.
            If 0 < n0 < 1, it is the fraction of the total no. of nodes
        n_epn : int, optional (default=10)
            Average number of edges per node.
        """

        # Name of the selected corpus topic model
        if label is None:
            graph_name = pathlib.Path(path).name
        else:
            graph_name = label

        # #########
        # LOAD DATA

        # Take the path to the data from the config file, if possible
        # path = self.global_parameters['source_data'][corpus]
        logging.info(f'-- Loading data matrix from {path}')

        # Load data matrix
        data, df_nodes = self.DM.readCoordsFromFile(
            path, sparse=True, ref_col='corpusid')

        T = data['thetas']
        del data

        if df_nodes is None:
            n_max = T.shape[0]
            n_dig = int(math.log10(n_max) + 1)
            nodes = [str(n).zfill(n_dig) for n in range(n_max)]
        else:
            nodes = df_nodes['corpusid'].tolist()

        # Take a random sample
        if n0 <= 0:
            n_gnodes = len(nodes)
        elif n0 < 1:
            n_gnodes = int(n0 * len(nodes))
        else:
            n_gnodes = int(n0)

        #
        logging.info(f'Topic matrix loaded with {n_gnodes} nodes')

        n_edges = int(n_epn * n_gnodes)

        # ########################
        # COMPUTE SIMILARITY GRAPH

        # Create datagraph with the full feature matrix
        self.SG.makeSuperNode(label=graph_name, nodes=nodes, T=T)
        self.SG.sub_snode(graph_name, n_gnodes, ylabel=graph_name,
                          sampleT=True)

        self.SG.computeSimGraph(
            graph_name, n_edges=n_edges, n_gnodes=n_gnodes, similarity=sim,
            g=1, blocksize=self.blocksize, useGPU=self.useGPU, tmp_folder=None,
            save_every=20_000_000, verbose=True)

        # #####################
        # SHOW AND SAVE RESULTS

        # Log some results
        # md = dg.metadata['graph']
        md = self.SG.snodes[graph_name].metadata
        logging.info(f"-- -- Similarity measure: {md['edges']['metric']}")
        logging.info(f"-- -- Number of nodes: {md['nodes']['n_nodes']}")
        logging.info(f"-- -- Number of edges: {md['edges']['n_edges']}")
        logging.info(f"-- -- Average neighbors per node: "
                     f"{md['edges']['neighbors_per_sampled_node']}")
        logging.info(f"-- -- Density of the similarity graph: "
                     f"{100 * md['edges']['density']} %")

        gc.collect()

        # Save graph: nodes and edges
        self.SG.save_supergraph()

        # Reset snode. This is to save memory.
        self._deactivate()

        return

    def infer_sim_bigraph(self, s_label, t_label, sim, epn=10):
        """
        This method manages the generation of similarity (semantic) bipartite
        graphs.

        It assumes that the feature vectors in source and target nodes are
        comparable.

        Parameters
        ----------
        s_label : str
            Name of the source graph (it must contain a feature matrix)
        t_path : str
            Name of the source graph (it must contain a feature matrix that was
            comparable to that of the source graph)
        sim : str
            Similarity measure
        epn : int, optional (default=10)
            Average number of edges per node.
        """

        # ###########
        # LOAD GRAPHS

        self.SG.activate_snode(s_label)
        self.SG.activate_snode(t_label)

        n_source = self.SG.snodes[s_label].n_nodes
        n_target = self.SG.snodes[t_label].n_nodes
        # Target number of edges
        n_edges = int(0.5 * epn * (n_source + n_target))

        # ##########################
        # COMPUTE SIMILARITY BIGRAPH

        # Create datagraphs with the full feature matrices
        e_label = f"{s_label}_2_{t_label}"
        self.SG.computeSimBiGraph(
            s_label, t_label, e_label=e_label, n_edges=n_edges,
            n_gnodesS=n_source, n_gnodesT=n_target, similarity=sim, g=1,
            blocksize=self.blocksize, useGPU=self.useGPU)

        # #####################
        # SHOW AND SAVE RESULTS

        # Log some results
        # md = dg.metadata['graph']
        md = self.SG.sedges[e_label].metadata
        logging.info(f"-- -- Similarity measure: {md['edges']['metric']}")
        logging.info(f"-- -- Number of nodes: {md['nodes']['n_nodes']}")
        logging.info(f"-- -- Number of edges: {md['edges']['n_edges']}")
        logging.info(f"-- -- Average neighbors per node: "
                     f"{md['edges']['neighbors_per_sampled_node']}")
        logging.info(f"-- -- Density of the similarity graph: "
                     f"{100 * md['edges']['density']} %")

        # Save graph: nodes and edges
        self.SG.save_supergraph()

        # Reset snode. This is to save memory.
        self._deactivate()

        return

    def inferBGfromA(self, path, attribute, t_label=None, e_label=None):
        """
        Infer bipartite graph from a categorical attribute

        Parameters
        ----------
        path : str
            Path to snode
        attribute : str
            Name of the snode attribute used to generate the bipartite graph
        t_label : str or None, optional (default=None)
            Name of the target s_node
        e_label : str or None, optional (default=None)
            Name of the bipartite graph
        """

        # ##########
        # LOAD GRAPH

        # Load source snode
        s_label = pathlib.Path(path).name

        # ################
        # MAKE SUPERGRAPH

        # Generate new snode and sedge from the selected attribute
        if t_label is None:
            t_label = attribute + '_' + s_label
        if e_label is None:
            e_label = s_label + '_2_' + t_label

        self.SG.snode_from_atts(s_label, attribute, target=t_label,
                                e_label=e_label, save_T=True)

        self.SG.save_supergraph()
        # Clean memory:
        self._deactivate()

        return

    def transduce(self, path, order):
        """
        Compute a transductive graph from a snode and a sedge

        Parameters
        ----------
        path : str
            Path to the sedge (snode is inferred from the sedge name)
        order : int
            Order parameters of the transduced graph
        """

        # Load sedge
        xylabel = pathlib.Path(path).name

        # Compute transduced graph
        self.SG.transduce(xylabel, n=order, normalize=True)

        # Save modified snode
        self.SG.save_supergraph()
        # Deactivate snode
        self._deactivate()

        return

    def infer_ppr_graph(self, path):
        """
        Compute a transductive graph from a snode and a sedge

        Parameters
        ----------
        path : str or pathlib.Path
            Path to the sedge (snode is inferred from the sedge name)
        """

        # Load sedge
        label = pathlib.Path(path).name

        # Compute transduced graph
        th = 0.9
        self.SG.compute_ppr(label, th=th, inplace=False)

        # Save modified snode
        self.SG.save_supergraph()
        # Deactivate snode
        self._deactivate()

        return

    def inferSimBG(self):
        """
        Not available
        """
        return

    def inferTransit(self, path_xm, path_my):
        """
        Infer transitive graph from two bipartite graphs

        Parameters
        ----------
        path_xm : str or pathlib.Path
            Path to first bipartite graph (sedge)
        path_my : str or pathlib.Path
            Path to second bipartite graph (sedge)
        """

        # Identify sedges XM and MY
        xmlabel = pathlib.Path(path_xm).name
        mylabel = pathlib.Path(path_my).name

        # Activate sedges to get some info from them
        self.SG.activate_sedge(xmlabel)
        self.SG.activate_sedge(mylabel)

        # Get origin and target nodes:
        xmlabel0 = self.SG.sedges[xmlabel].metadata['graph']['source']
        xmlabel1 = self.SG.sedges[xmlabel].metadata['graph']['target']
        mylabel0 = self.SG.sedges[mylabel].metadata['graph']['source']
        mylabel1 = self.SG.sedges[mylabel].metadata['graph']['target']

        # Identify the connecting snode (i.e., the 'm' snode).
        mlabel = set([xmlabel0, xmlabel1]) & set([mylabel0, mylabel1])

        if len(mlabel) == 0:
            logging.warning("The selected sedges have no common snode")
        elif len(mlabel) == 2:
            logging.warning("The selected sedges have the same snodes. "
                            "One and only one node must be the same")
        else:

            # Name of the output sedge
            e_label = ((set([xmlabel0, xmlabel1]) - mlabel).pop() + '_2_'
                       + (set([mylabel0, mylabel1]) - mlabel).pop())

            # Compute transitive graph
            self.SG.transitive_graph(e_label, xmlabel, mylabel)

        # Save
        self.SG.deactivate_sedge(xmlabel)
        self.SG.deactivate_sedge(mylabel)
        self.SG.save_supergraph()
        self._deactivate()

        return

    # ##############
    # Graph analysis
    # ##############
    def local_graph_analysis(self, parameter, path):
        """
        Computes a local parameter for a snode

        Parameters
        ----------
        parameter : str
            Local parameter to compute
        path : str or pathlib.Path
            Path to snode
        """

        # Create graph object
        graph_name = pathlib.Path(path).name

        # Local graph analysis
        self.SG.local_snode_analysis(graph_name, parameter=parameter)

        # Save
        self.SG.save_supergraph()
        self._deactivate()

        return

    def detectCommunities(self, algorithm, path, comm_label=None,
                          seed=None):
        """
        Applies a community detection algorithm to a given snode

        Parameters
        ----------
        algorithm : str
            Community detection algoritms
        path : str
            Path to snode
        comm_label : str or None (default=None)
            Label for the column storing the community indices
        seed : int or None (default=None)
            Seed for randomization
        """

        if comm_label is None:
            comm_label = algorithm

        # Create graph object
        graph_name = pathlib.Path(path).name

        # In the current version of the SuperGraph class, snode activation must
        # be done before calling to the snode method. Maybe I should consider
        # activation inside the method
        self.SG.detectCommunities(
            graph_name, alg=algorithm, ncmax=None, comm_label=comm_label,
            seed=seed)

        # ############
        # SAVE RESULTS

        # Save graph: nodes and edges
        self.SG.save_supergraph()
        # Reset snode. This is to save memory.
        self._deactivate()

        return

    def community_metric(self, path, community, parameter):
        """
        Compute a global metric for a graph partition resulting from a
        community detection algorithm

        Parameters
        ----------
        path : str or pathlib.Path
            Path to snode
        community : str
            Community detection algorithm
        parameter : str
            Metric to compute
        """

        logging.info(f'-- Computing community metric: {parameter}')
        # Create graph object
        graph_name = pathlib.Path(path).name
        dg = DataGraph(label=graph_name, path=path)

        # Local graph analysis
        dg.community_metric(cd_alg=community, parameter=parameter)

        # Save Results
        dg.save_metadata()

        return

    def compare_communities(self, path1, comm1, path2, comm2, metric):
        """
        Compate two graph partitions

        Parameters
        ----------
        path1 : str or pathlib.Path
            Path to 1st snode
        comm1 : str
            Name of the partition from 1st snode
        path2 : str or pathlib.Path
            Path to 2nd snode
        comm2 : str
            Name of the partition from 2nd snode
        metric : str
            Metric used for the comparison
        """

        logging.info(f'-- Computing communities with metric: {metric}')

        # Create graph objects
        graph1_name = pathlib.Path(path1).name
        self.SG.activate_snode(graph1_name)
        graph2_name = pathlib.Path(path2).name
        if path2 != path1:
            self.SG.activate_snode(graph2_name)

        # Get community structures (lists of membership values)
        clabels1 = self.SG.snodes[graph1_name].df_nodes[comm1].tolist()
        clabels2 = self.SG.snodes[graph2_name].df_nodes[comm2].tolist()

        # Call community comparator with the given metric
        CD = CommunityPlus()
        d = CD.compare_communities(clabels1, clabels2, method=metric,
                                   remove_none=False)

        # Print results
        logging.info(f'-- -- The value of {metric.upper()} between '
                     f'community {comm1} from graph {graph1_name} and '
                     f'community {comm2} from graph {graph2_name} is {d}')

        self._deactivate()

        return

    def profile_node(self):

        # Ask for node name:
        node_name = input('\n-- Write the node name to disambigate: ')

        node_profile = self.SG.profile_node(node_name)
        logging.info("-- Node profile")
        logging.info(f"-- Name of the node: {node_name}")
        logging.info(f"-- Report: {node_profile}")

        return

    # ###################
    # Graph visualization
    # ###################
    def graph_layout(self, path2snode, attribute, num_iterations=50):
        """
        Compute the layout of the given graph

        Parameters
        ----------
        path2snode : str or pathlib.Path
            Path to snode
        attribute: str
            Snode attribute used to color the graph
        """

        # Create graph object
        graph_name = pathlib.Path(path2snode).name

        # Parameters
        if self.SG.get_metadata(graph_name)['nodes']['n_nodes'] > 100:
            alg = 'fa2'
            gravity = 40
        else:
            alg = 'fr'
            gravity = 1000

        self.SG.graph_layout(graph_name, attribute, gravity=gravity, alg=alg,
                             num_iterations=num_iterations)

        # ############
        # SAVE RESULTS

        # Save graph: nodes and edges
        self.SG.save_supergraph()
        # Reset active snodes or sedges. This is to save memory.
        self._deactivate()

        return

    def display_graph(self, path2snode, attribute, path=None):
        """
        Display the graph using matplotlib

        Parameters
        ----------
        path2snode : str or pathlib.Path
            Path to snode
        attribute: str
            Snode attribute used to color the graph
        path: pathlib.Path, str or None, optional (default=None)
            Path to save the figure. If None, the figure is saved in a
            default path.

        Returns
        -------
        attrib_2_idx : dict
            Dictionary attributes -> RGB colors. It stores the colors used
            to represent the attribute value for each node.
        """

        # Create graph object
        graph_name = pathlib.Path(path2snode).name

        att_2_idx = self.SG.display_graph(graph_name, attribute, path=path)

        # ############
        # SAVE RESULTS

        # Save graph: nodes and edges
        self.SG.save_supergraph()
        # Reset active snodes or sedges. This is to save memory.
        self._deactivate()

        return att_2_idx

    def display_bigraph(self, path2sedge, s_att1, s_att2, t_att, t_att2=None,
                        template_html="bigraph_template.html",
                        template_js="make_bigraph_template.js"):
        """
        Generates a bigraph visualization based on halo.

        Parameters
        ----------
        path2sedge : str or pathlib.Path
            Path to the bipartite graph. The bigraph (sedge) and both the
            source and target snodes must already exist in the supergraph
            structure.
            The name of the sedge and the source and target snodes is taken
            from the folder name.
        s_att1 : str
            Name of the first attribute of the source snode.
            It should be a string attribute (not tested with others)
        s_att2 : str
            Name of the second attribute of the source node.
            It should be a string attribute, though it could work with
            integers too (not fully tested).
        t_att : str
            Name of the attribute of the target node.
            It should be a string attribute (not tested with others)
        t_att2 : str or None, optional (default=None)
            Name of the second attribute of the target node
            It should be a string attribute (not tested with others)
        template_html : str
            Name of the template html file.
        template_js : str
            Name of the template js file
        """

        def _htmlize(text0):
            """
            Replace tildes in text by standard html forms
            """
            text = text0.replace('á', '&aacute;')
            text = text.replace('é', '&eacute;')
            text = text.replace('í', '&iacute;')
            text = text.replace('ó', '&oacute;')
            text = text.replace('ú', '&uacute;')
            text = text.replace('ñ', '&ntilde;')

            return text

        # ###########################
        # Export graph to Halo format

        # Get names of bipartite graph and terminal graphs
        e_label = pathlib.Path(path2sedge).name
        s_label, t_label = self.SG.get_terminals(e_label)

        # # Export graph to Halo format
        label_map = self.SG.export_2_halo(
            e_label, s_att1, s_att2, t_att, t_att2)
        inv_map = {v: k for k, v in label_map.items()}

        # #########################
        # Copy csv into Halo folder
        path_orig = path2sedge / f'halo_{e_label}.csv'
        path_dest = self.path2halo / 'halo' / 'data' / 'bigraph_halo.csv'
        shutil.copyfile(path_orig, path_dest)

        # ##############
        # Edit HTML file

        # # Read template file
        path2html_template = pathlib.Path('templates') / template_html
        text = open(path2html_template).read()
        title = f'{s_label} - {t_label}'
        text = text.replace('[[TITLE]]', title)
        text = text.replace('[[SOURCE]]', s_label)
        text = text.replace('[[TARGET]]', t_label)
        # This is to avoid errors with tildes in some visualizations
        text = _htmlize(text)

        # Write html file
        path2html = self.path2halo / 'halo' / 'bigraph.html'
        with open(path2html, "w+") as f:
            f.write(text)

        # ##############
        # Edit JS file

        # # Read template file
        path2html_template = pathlib.Path('templates') / template_js
        text = open(path2html_template).read()
        text = text.replace('[[TITLE]]', title)
        for x in ['SOURCE_NM', 'SOURCE_CAT', 'TARGET_CAT', 'TARGET_NM']:
            text = text.replace(f'[[{x}]]', inv_map[x])

        # Write html file
        path2js = self.path2halo / 'halo' / 'make_bigraph.js'
        with open(path2js, "w+") as f:
            f.write(text)

        # #########
        # Display graph

        # Move to halo folder temporarily
        owd = Path.cwd()
        os.chdir(self.path2halo)

        port = 8080
        handler = http.server.SimpleHTTPRequestHandler

        print('Starting server...')
        server_address = ("", port)
        httpd = http.server.HTTPServer(server_address, handler)
        thread = threading.Thread(target=httpd.serve_forever)
        thread.start()
        print("Server has started. Displaying graph...")
        webbrowser.open(f'http://localhost:{port}/halo/bigraph.html', new=0)
        input("-- -- Press Enter to continue...\n\n")

        threading.Thread(target=httpd.shutdown, daemon=True).start()

        # Return to running folder
        os.chdir(owd)

        return

    def show_top_nodes(self, path, feature):
        """
        Shows a reduced list of nodes from a given graph, ranked by the value
        of a single feature

        Parameters
        ----------
            path : str or pathlib.Path
                Path to the graph
            feature : str
                Name of the local feature
        """

        # Size of the top list
        n = 400

        # Create graph obje
        label = pathlib.Path(path).name

        self.SG.activate_snode(label)

        # Sort nodes vy decreasing value of the selected feature
        df = self.SG.snodes[label].df_nodes.sort_values(
            feature, axis=0, ascending=False)

        # Remove rows with NaNs. This is aimed for citation graphs
        if 'eid' in df.columns:
            df = df[~pd.isna(df['eid'])]

        print(f"-- -- Top ranked nodes in {label} by {feature}")
        print(df.head(50))

        # Save the top n columns only
        path2out = (self.path2project / self.f_struct['output']
                    / f'top_{label}_{feature}.xlsx')
        df.iloc[:n].to_excel(path2out, index=False, encoding='utf-8')
        print(f"-- -- Top {n} saved in {path2out}")

        self._deactivate()

        return

    # ############
    # Graph export
    # ############
    def export_2_parquet(self, path):
        """
        Exports graph to parquet files

        Parameters
        ----------
            path : str
                Path to the graph
        """

        graph_name = pathlib.Path(path).name

        path2folder = self.path2project / self.f_struct['export']
        path2nodes = path2folder / f"{graph_name}_nodes.parquet"
        path2edges = path2folder / f"{graph_name}_edges.parquet"

        self.SG.export_2_parquet(graph_name, path2nodes, path2edges)

        self._deactivate()

        return


class SgTaskManagerCMD(SgTaskManager):
    """
    Extends task manager to get data from the user through a command window
    """

    def _request_confirmation(self, msg="     Are you sure?"):
        """
        Requests a confirmation from user

        Parameters
        ----------
        msg : str
            Prompt message

        Returns
        -------
        r : str {'yes', 'no'}
            User response
        """

        # Iterate until an admissible response is got
        r = ''
        while r not in ['yes', 'no']:
            r = input(msg + ' (yes | no): ')

        return r == 'yes'

    # ################
    # Neo4J management
    # ################
    def reset_Neo4J(self):
        """
        Reset the whole database
        """

        print("---- This will reset the entire database. All data will be "
              "lost.")
        if self._request_confirmation():
            super().reset_Neo4J()
        else:
            logging.info("---- Reset cancelled")

        return

    def reset_Neo4J_snode(self, snode):
        """
        Reset (drop and create emtpy) tables from the database.

        Parameters
        ----------
        snode : str
            Selected node or edge to reset
        """

        print("---- WARNING: This will reset the snode from the database.")
        if self._request_confirmation():
            super().reset_Neo4J_snode(snode)
        else:
            logging.info("---- Reset cancelled")

        return

    def reset_Neo4J_sedge(self, sedge):
        """
        Reset (drop and create emtpy) tables from the database.

        Parameters
        ----------
        sedge : str
            Selected node or edge to reset
        """

        print("---- WARNING: This will reset the sedge from the database.")
        if self._request_confirmation():
            super().reset_Neo4J_sedge(sedge)
            self.DM.Neo4j.drop_relationship(sedge)
        else:
            logging.info("---- Reset cancelled")

        return

    # ###########
    # Load graphs
    # ###########
    def import_snode_from_table(self, table_name):
        """
        Import a graph from a table containing node names, attributes and
        (possibly) embeddings

        Parameters
        ----------
        table_name : str
            Name of the folder containing the table of nodes
        """

        # #######################
        # REQUEST NUMBER OF NODES

        # The final no. of nodes is the minimum between the selected number of
        # nodes and the result of applyng the sampling factor over the whole
        # datasets
        # Both parameter are needed for cases where the dataset is too large.
        # The sampling factor can be used to avoid reading the whole dataset
        # from the parquet files, while n0 is used as the true target number
        # of nodes

        # Number of nodes.
        # If 0, all nodes are used
        # If 0 < n_nodes < 1, this is the fraction of the total
        # If n_nodes > 1, number of nodes
        n0_default = 0
        n0 = float(input(f"Select number of nodes [0=all]: ") or n0_default)

        # Sampling factor
        sf_default = 1
        sampling_factor = float(input(
            f"Sampling factor (default 1 = no sampling): ") or sf_default)

        super().import_snode_from_table(
            table_name, n0=n0, sampling_factor=sampling_factor)

        return

    def import_nodes_and_model(self, path):
        """
        This method manages the generation of similarity (semantic)
        graphs.

        Parameters
        ----------
        path : str
            Path to the model
        """

        # #######################
        # REQUEST NUMBER OF NODES

        # Number of nodes.
        # If 0, all nodes are used
        # If 0 < n_nodes < 1, this is the fraction of the total
        # If n_nodes > 1, number of nodes
        n0_default = 0
        n0 = float(input(f"Select number of nodes [0=all]: ") or n0_default)

        super().import_nodes_and_model(path, n0=n0)

        return

    # ###########
    # Graph tools
    # ###########
    def subsample_graph(self, path, mode):
        """
        Subsample graph

        Parameters
        ----------
        path : str or pathlib.Path
            Path to graph
        mode : str
            If 'newgraph', create a new snode with the subgraph
        """

        # Read number of nodes in the graph
        graph_name = pathlib.Path(path).name
        atts = self.SG.get_attributes(graph_name)
        n_nodes = atts['nodes']['n_nodes']

        # Number of target nodes.
        n0 = int(n_nodes / 2)
        n0 = int(input(f"Select number of target nodes [{n0}]: ") or n0)

        super().subsample_graph(path, mode, n0)

        return

    def filter_edges(self, path):
        """
        Subsample graph

        Parameters
        ----------
        path : str
            Path to graph
        """

        # Read minimum weight value
        graph_name = pathlib.Path(path).name
        if not self.SG.is_active_snode(graph_name):
            self.SG.activate_snode(graph_name)
        smin = min(self.SG.snodes[graph_name].df_edges['Weight'])

        # Ask threshold
        th = float(input(f"Select threshold [{smin}]: ") or smin)

        super().filter_edges(path, th)

        return

    # ###############
    # Graph inference
    # ###############
    def infer_eq_simgraph(self, path, sim):
        """
        This method manages the generation of an equivalence similarity
        (semantic) graph.

        It is similar to a concatenatio of self.equivalence_graph() (to
        transform the original feature matrix into the reduced matrix without
        row repetitions) and infer_sim_graph() (to compute the similarity
        graph)

        Parameters
        ----------
        path : str
            Path to the model
        sim : str
            Similarity measure
        """

        # Request number of nodes.
        # If 0, all nodes are used
        # If 0 < n_nodes < 1, this is the fraction of the total
        # If n_nodes > 1, number of nodes
        default = 0
        n0 = float(input(f"Select number of nodes [0=all]: ") or default)

        # Request number of edges
        default = 10
        epn = float(input(
            f"Select average number of edges per node [{default}]: ")
            or default)    # Be careful: this 'or' is order-sensitive...

        # Compute equivalent similarity graph
        super().infer_eq_simgraph(path, sim, n0=n0, epn=epn)

        return

    def infer_sim_graph(self, path2snode, sim, n0=None, n_epn=None):
        """
        This method manages the generation of similarity (semantic) graphs.

        Parameters
        ----------
        path2snode : str
            Path to the snode
        sim : str
            Similarity measure
        n0 : int or float of None, optional (default=None)
            Number of nodes. If None, it is requested to the user.
            If 0 < n_epn < 1, it is the fraction of the total no. of nodes
            If 0, all nodes are taken.
        n_epn : int or None, optional (default=None)
            Average number of edges per node. If None, it is requested to the
            user. If 0, 10 nodes are taken.
        """

        # Request number of edges
        if n_epn is None:
            default = 10
            # Be careful: this 'or' is order-sensitive...
            n_epn = float(input(f"Select average number of edges per node "
                                f"[{default}]: ") or default)

        super().infer_sim_graph(path2snode, sim, n0=n0, n_epn=n_epn)

        return

    def import_and_infer_sim_graph(self, path, sim, n0=None, n_epn=None,
                                   label=None):
        """
        This method manages the generation of similarity (semantic) graphs.

        Parameters
        ----------
        path : str
            Path to the model
        sim : str
            Similarity measure
        n0 : int or float of None, optional (default=None)
            Number of nodes. If None, it is requested to the user.
            If 0 < n_epn < 1, it is the fraction of the total no. of nodes
            If 0, all nodes are taken.
        n_epn : int or None, optional (default=None)
            Average number of edges per node. If None, it is requested to the
            user. If 0, 10 nodes are taken.
        """

        # Request number of nodes.
        # If 0, all nodes are used
        # If 0 < n_nodes < 1, this is the fraction of the total
        # If n_nodes > 1, number of nodes
        if n0 is None:
            n0_default = 0
            # Be careful: this 'or' is order-sensitive...
            n0 = float(input(f"Select number of nodes [0=all]: ")
                       or n0_default)

        # Requeest number of edges
        if n_epn is None:
            n0_default = 10
            # Be careful: this 'or' is order-sensitive...
            n_epn = float(input(f"Select average number of edges per node "
                                f"[{n0_default}]: ") or n0_default)

        super().import_and_infer_sim_graph(
            path, sim, n0=n0, n_epn=n_epn, label=None)

        return

    def infer_sim_bigraph(self, s_label, t_label, sim):
        """
        This method manages the generation of similarity (semantic) bipartite
        graphs.

        It assumes that the feature vectors in source and target nodes are
        comparable.

        Parameters
        ----------
        s_label : str
            Name of the source graph (it must contain a feature matrix)
        t_path : str
            Name of the source graph (it must contain a feature matrix that was
            comparable to that of the source graph)
        sim : str
            Similarity measure
        """

        # Request number of edges per node
        n_def = 10
        # Be careful: this 'or' is order-sensitive...
        epn = float(input(f"Average edges per node [{n_def}]: ") or n_def)

        super().infer_sim_bigraph(s_label, t_label, sim, epn=epn)

        return

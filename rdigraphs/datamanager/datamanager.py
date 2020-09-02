"""
The classes in this file provide functionality to interact with the specific
databases provided for the PTL projects.

"""

from __future__ import print_function    # For python 2 copmatibility
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
import copy
import os
import logging

try:
    import tkinter as tk
    from tkinter import filedialog
except:
    pass

# Local imports
from rdigraphs.datamanager.base_dm_sql import BaseDMsql
from rdigraphs.datamanager.base_dm_neo4j import BaseDMneo4j


class DMsql(BaseDMsql):
    """
    This class is an extension of BaseDMsql to include some additional
    functionality
    """

    def showDBview(self):

        title = "= Database {} ====================".format(self.dbname)

        print("=" * len(title))
        print(title)
        print("=" * len(title))
        print("")

        for table in self.getTableNames():

            print("---- Table: {}".format(table))
            cols, n_rows = self.getTableInfo(table)
            print("---- ---- Attributes: {}".format(', '.join(cols)))
            print("---- ---- No. of rows: {}".format(n_rows))
            print("")

        return


class DMneo4j(BaseDMneo4j):
    """
    This class is an extension of BaseDMneo4j to include some additional
    functionality
    """

    def showDBview(self):

        title = "= Neo4J Database ================================"

        print("=" * len(title))
        print(title)
        print("=" * len(title))
        print("")

        gdb_struct = self.get_db_structure()
        snodes = gdb_struct.labels
        sedges = gdb_struct.relationships

        print("---- Supernodes:")
        for snode in snodes:

            print("          " + snode)
            # cols, n_rows = self.getTableInfo(snode)
            # print("---- ---- Attributes: {}".format(', '.join(cols)))
            # print("---- ---- No. of rows: {}".format(n_rows))
            print("")

        print("---- Superedges:")
        for sedge in sedges:

            print("          " + sedge)
            # cols, n_rows = self.getTableInfo(snode)
            # print("---- ---- Attributes: {}".format(', '.join(cols)))
            # print("---- ---- No. of rows: {}".format(n_rows))
            print("")

        return


class DataManager(object):
    """
    This is the datamanager for the RDI project. It provides functionality to
    manage both the neo4j graph DB and the SQL databased containing the source
    data. To do so, it uses generic managers for Neo4j and SQL.
    """

    def __init__(self, path2project, db_params):
        """
        Initializes several SQL DataManager objects and one Noe4J DataManager:

        Eaxh SQL manager is stored in dictionary self.SQL. Tipically:

            self.SQL['Pr']  : SQL db for projects
            self.SQL['Pu']  : SQL db for publications
            self.SQL['Pa']  : SQL db for patents
            self.Neo4j      : Neo4j db for graphs.

        Parameters
        ----------
        path2project : str
            Path to the project folder
        db_params : dict
            Parameters to stablish db connections.
        """

        # Store paths to the main project folders and files
        self._db_params = db_params
        self._path2project = copy.copy(path2project)

        # This will be a fake SQL connector, used to forze a link in the UML
        # diagram generated with pyreverse
        self.lastSQL = None
        self.SQL = {}
        self.Neo4j = None
        self.dbON = False  # Set to true after DB connects. Likely deprecated

        # Create handlers for all sql DBs
        selections = db_params['SQL']['db_selection']
        for corpus, name in selections.items():
            if name:
                logging.info(f'-- -- Connecting database {name}')
                db = db_params['SQL']['databases'][name]
                if db['category'] == corpus:
                    connector = db['connector']
                    server = db['server']
                    user = db['user']
                    password = db['password']
                    if 'port' in db:
                        port = db['port']
                    else:
                        port = None
                    if 'unix_socket' in db:
                        unix_socket = db['unix_socket']
                    else:
                        unix_socket = None

                    self.lastSQL = DMsql(
                        name, connector, path2db=None, db_server=server,
                        db_user=user, db_password=password, db_port=port,
                        unix_socket=unix_socket, charset='utf8mb4')
                    self.SQL[corpus] = self.lastSQL

                    # dbON = FALSE if any of the requested DBs cannot be
                    # connected
                    self.dbON = self.dbON * self.SQL[corpus].dbON
                else:
                    # The selected database is not of the expected type
                    print(f"-- -- ERROR: a DB of class {name} was expected. " +
                          f"A DB of class {db['category']} was found.")

        # Crate handler for the Neo4j db
        db = db_params['neo4j']
        server = db['server']
        user = db['user']
        password = db['password']

        try:
            self.Neo4j = DMneo4j(server, password, user)
        except:
            logging.warning("-- Neo4J DB connection failed. " +
                            "Graph data is not accessible")
            self.dbON = False

        return

    def readCoordsFromFile(self, fpath=None, fields=['thetas'], sparse=False,
                           path2nodenames=None, ref_col='corpusid'):
        """
        Reads a data matrix from a given path.
        This method assumes a particular data structure of the PTL projects

        Parameters
        ----------
        fpath : str or None, optional (default=None)
            Path to the file that contains the topic model.
            (the data file is assumed to be modelo.npz or modelo_sparse.npz
            inside that folder)
        fields : str or list, optional (default=['thetas'])
            Name of the field or fields containing the doc-topic matrix
        sparse : bool, optional (default=False)
            If True, the doc-topic matrix is sparse, otherwise dense
        path2nodenames : str or None, optional (default=None)
            path to file containing metadata (in particular node names). If
            None, the file is assumed to be in fpath, with name
            docs_metadata.csv
        ref_col : str, optional (default='corpusid')
            Name of the column in the metadata file (given by path2nodenames)
            that contains the doc id.

        Returns
        -------
        data_out : dict
            Output data dictionary
        df_nodes : dataframe
            Dataframe of nodes
        """

        # #############
        # DATA LOCATION

        # If there is no path in the config file, a GUI is used
        if fpath is None:

            # Open a GUI to select the source data folder
            msg_g1 = 'Select the folder that contains the source data'
            print(msg_g1)
            root = tk.Tk()    # Create the window interface
            root.withdraw()   # This is to avoid showing the tk window
            # Show the filedialog window (only)
            root.overrideredirect(True)
            root.geometry('0x0+0+0')
            root.deiconify()
            root.lift()
            root.focus_force()
            # path = filedialog.askopenfilename(initialdir=os.getcwd(),
            #                                   title=msg_g0)
            fpath = filedialog.askdirectory(
                initialdir=os.path.dirname(self.path2project), title=msg_g1)
            root.update()     # CLose the filedialog window
            root.destroy()    # Destroy the tkinter object.

        # If there is only one file with extension npz, take it
        fnpz = [f for f in os.listdir(fpath) if f.split('.')[-1] == 'npz']
        if len(fnpz) == 1:
            path2topics = os.path.join(fpath, fnpz[0])
        # otherwise, take the one with the specified names
        elif sparse:
            path2topics = os.path.join(fpath, 'modelo_sparse.npz')
        else:
            path2topics = os.path.join(fpath, 'modelo.npz')

        if path2nodenames is None:
            path2nodenames = os.path.join(fpath, 'docs_metadata.csv')

        # ##############
        # LOADING TOPICS

        data = np.load(path2topics, mmap_mode=None, allow_pickle=True,
                       fix_imports=True, encoding='ASCII')
        logging.info(f'-- -- Topic model loaded from {path2topics}')

        # Since data is and
        data_out = {}
        for field in fields:
            if field == 'thetas' and field not in data:
                # If here, it means that the topic matrix is sparse.
                # We must build it up in the sparse.csr format.
                data_out['thetas'] = csr_matrix(
                    (data['thetas_data'], data['thetas_indices'],
                     data['thetas_indptr']), shape=data['thetas_shape'])
            else:
                data_out[field] = data[field]

        # #############
        # LOADING NAMES

        if os.path.isfile(path2nodenames):
            df_nodes = pd.read_csv(path2nodenames, usecols=[ref_col])
        else:
            logging.info(f'-- -- File {path2nodenames} with node names does '
                         'not exist. No dataframe of nodes is returned')
            df_nodes = None

        del data

        return data_out, df_nodes

    def _load_SCOPUS_citations_data(self, col_ref, atts, mode='cite_to'):
        """
        Extracts data from table 'citations' in SCOPUS SQL database

        The subgraph contains the nodes with attributes in table 'document'
        form the same databse.

        Parameters
        ----------
        col_ref : str
            Column in sql table that will be used as index
        atts : list of str
            Columns in sql table that will be taken as attributes
        mode : str {'cite_to', 'cited_by'}
            If 'cite_to', source nodes are the citing papers
            If 'cited_by', source nodes are the cited papers

        Returns
        -------
        nodes : list
            Nodes with attributes in table 'document'
        df_atts : pandas dataframe
            Attributes of the nodes
        source_nodes : list
            Source of all edges between nodes in the list of nodes
        target_nodes
            Target of all edges between nodes in the list of nodes.
            Edges are assumed to be in the same order in source_nodes and
            target_nodes
        """

        # ##########################
        # Load nodes with attributes

        logging.info(
            '-- -- Loading nodes. This may take some time ...')
        df_nodes = self.SQL['Pu'].readDBtable(
            'document', limit=None, selectOptions=None, filterOptions=None,
            orderOptions=None)

        nodes_with_atts = df_nodes[col_ref].tolist()

        if col_ref not in atts:
            atts = [col_ref] + atts
        df_atts = df_nodes[atts]

        # ###################
        # Load citation graph

        logging.info(
            '-- -- Loading citations graph. This may take some time ...')
        df = self.SQL['Pu'].readDBtable(
            'citation', limit=None, selectOptions=None, filterOptions=None,
            orderOptions=None)

        # ##############
        # Edge selection

        # Take list of sources and targets
        if mode == 'cite_to':
            source_nodes = df.cite_from.tolist()
            target_nodes = df.cite_to.tolist()
        elif mode == 'cited_by':
            source_nodes = df.cite_to.tolist()
            target_nodes = df.cite_from.tolist()
        else:
            logging.error(
                '-- Unknown citation mode. Options cite_to and cited_by only')
            exit()

        return nodes_with_atts, df_atts, source_nodes, target_nodes

    def load_SCOPUS_citations_with_atts(self, col_ref, atts, mode='cite_to'):
        """
        Extracts data from table 'citations' in SCOPUS SQL database

        The subgraph contains the nodes with attributes in table 'document'
        form the same databse.

        Parameters
        ----------
        col_ref : str
            Column in sql table that will be used as index
        atts : list of str
            Columns in sql table that will be taken as attributes
        mode : str {'cite_to', 'cited_by'}
            If 'cite_to', source nodes are the citing papers
            If 'cited_by', source nodes are the cited papers

        Returns
        -------
        nodes : list
            Nodes with attributes in table 'document'
        df_atts : pandas dataframe
            Attributes of the nodes
        source_nodes : list
            Source of all edges between nodes in the list of nodes
        target_nodes
            Target of all edges between nodes in the list of nodes.
            Edges are assumed to be in the same order in source_nodes and
            target_nodes
        """

        # ##########################
        # Load nodes with attributes

        nodes_with_atts, df_atts, source_nodes, target_nodes = \
            self._load_SCOPUS_citations_data(col_ref, atts, mode)

        # Make list of nodes in edges that are not nodes in df_nodes
        citing_set = set(source_nodes + target_nodes)
        nodes_set = set(nodes_with_atts)
        external_nodes_set = citing_set - nodes_set

        # Mapping nodes to a binary flag indicating membership to df_nodes
        nodes_dict = dict([(x, 1) for x in list(nodes_set)] +
                          [(x, 0) for x in list(external_nodes_set)])

        # Select edges connecting nodes in df_nodes only
        edges = [x for x in zip(source_nodes, target_nodes)
                 if nodes_dict[x[0]] & nodes_dict[x[1]]]
        source_nodes, target_nodes = zip(*edges)

        return nodes_with_atts, df_atts, source_nodes, target_nodes

    def load_SCOPUS_citations_all(self, col_ref, atts, mode='cite_to'):
        """
        Extracts all data from table 'citations' in SCOPUS SQL database

        The graph contains all nodes in the citation graph, no matter if
        they have attributes in table 'document'.

        Parameters
        ----------
        col_ref : str
            Column in sql table that will be used as index
        atts : list of str
            Columns in sql table that will be taken as attributes
        mode : str {'cite_to', 'cited_by'}
            If 'cite_to', source nodes are the citing papers
            If 'cited_by', source nodes are the cited papers

        Returns
        -------
        nodes : list
            Nodes with attributes in table 'document'
        df_atts : pandas dataframe
            Attributes of the nodes
        source_nodes : list
            Source of all edges between nodes in the list of nodes
        target_nodes
            Target of all edges between nodes in the list of nodes.
            Edges are assumed to be in the same order in source_nodes and
            target_nodes
        """

        # Load nodes with attributes
        nodes_with_atts, df_atts, source_nodes, target_nodes = \
            self._load_SCOPUS_citations_data(col_ref, atts, mode)

        # Take complete set of nodes
        nodes = sorted(list(set(source_nodes + target_nodes)))

        return nodes, df_atts, source_nodes, target_nodes

#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Main program for the RDIgraph analyzer

Created on June 18 2018

@author: Jesús Cid Sueiro

# For profiling: kernprof -lv mainRDIgraph.py ...

"""

import pathlib
import argparse

# Local imports
from rdigraphs.menu_navigator.menu_navigator import MenuNavigator
from rdigraphs.valtaskmanager import ValTaskManager

# ########################
# Main body of application
# ########################

# ####################
# Read input arguments

# settings
parser = argparse.ArgumentParser()
parser.add_argument('--p', type=str, default=None,
                    help="path to a new or an existing evaluation project")
parser.add_argument('--source', type=str, default='../source_data',
                    help="path to the source data folder")
args = parser.parse_args()

# Read project_path
project_path = args.p
if args.p is None:
    while project_path is None or project_path == "":
        project_path = input('-- Write the path to the project to load or '
                             'create: ')
project_path = pathlib.Path(project_path)

if project_path.is_dir():
    option = 'load'
else:
    option = 'create'
active_options = None
query_needed = False

# Create SuperGraph Project
# paths2data stores the path to all source data that might be needed by the
# task manager
paths2data = {'topicmodels': path2source / 'models',
              'agents': path2source / 'agents',
              'ACL_models': path2source / 'ACL_models'}
tm = ValTaskManager(project_path, paths2data)

# ########################
# Prepare user interaction
# ########################

# paths2data stores the path to all folders that might be needed by the
# menu navigator. They may differ from those needed by the task manager
paths2data = {'graphs': project_path / 'graphs',
              'bigraphs': project_path / 'bigraphs',
              'topicmodels': path2source / 'models',
              'agents': path2source / 'agents',
              'ACL_models': path2source / 'ACL_models' / 'models' / 'tm',
              'validation': project_path / 'output' / 'validation'}
path2menu = pathlib.Path('val_menu.yaml')

# ##############
# Call navigator
# ##############

menu = MenuNavigator(tm, path2menu, paths2data)
menu.front_page(title="Topic model validator")
menu.navigate(option, active_options)

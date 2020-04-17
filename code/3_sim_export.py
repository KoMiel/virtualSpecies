
# import packages

import json
import numpy as np
import os

from landscape import Landscape
from settlement import Settlement
from export import Export


# import settings

with open('../settings.json') as f:
    settings = json.load(f)

# extract general settings

directoryLandscapes = settings['directory']['landscapes']
directorySettlements = settings['directory']['settlements']
directoryDatasets = settings['directory']['datasets']

nGrids = settings['grid']['n']
nDistributions = settings['nDistributions']

# import scenarios

with open('../scenarios.json') as f:
    scenarios = json.load(f)

# main program

# loop over all scenarios

for scenario in scenarios['scenarios']:

    # generate lists to extract information on variables

    varLabel = list()
    varWeight = list()

    # extract information on variables

    for variable in settings['var']:
        varLabel.append(settings['var'][variable]['label'])
        varWeight.append(settings['var'][variable]['weight'])

    interactionW = scenario['interactionW']

    # read the subpath to the scenario worked on

    subpath = scenario['path']

    # generate a directory for the scenario if it did not already exist

    if not os.path.exists(directoryDatasets + subpath):
        os.makedirs(directoryDatasets + subpath)

    # loop over all previously generated grids

    for grid in range(0, nGrids):

        # read the previously generated landscape object (the grid)

        lscape = Landscape()
        lscape.pkl_import(path=directoryLandscapes + 'landscape' + str(grid) + '.pkl')

        # start with an empty list of distributions

        settlements = []

        # loop over all distributions and import them

        for distribution in range(0, nDistributions):

            settlements.append(Settlement(landscape=lscape))
            settlements[distribution].import_from_pkl(path=directorySettlements + subpath + 'settlement' + str(grid) + '_' + str(distribution) + ".pkl")

        # generate an export object

        exp = Export(landscape=lscape, settlements=settlements, var_labels=varLabel)

        # merge all settlements to one data set and export it to .txt and .pkl

        exp.merge()

        # export to .txt and .pkl

        exp.export_to_txt(directoryDatasets + subpath + 'dataset' + str(grid) + ".txt")
        exp.export_to_pkl(directoryDatasets + subpath + 'dataset' + str(grid) + ".pkl")

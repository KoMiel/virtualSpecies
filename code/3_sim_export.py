

### import packages ###

import json
import numpy as np
import os

from landscape import landscape
from settlement import settlement                              
from export import export  


### import settings ###

with open('../settings.json') as f:
    settings = json.load(f)

# read general settings

directoryLandscapes = settings['directory']['landscapes']
directorySettlements = settings['directory']['settlements']
directoryDatasets = settings['directory']['datasets']

gridNum = settings['grid']['num']
yearsNum = settings['years']


### import scenarios ###

with open('../scenarios.json') as f:
    scenarios = json.load(f)


### main program ###

# loop over all scenarios

for scenario in scenarios['scenarios']:
        
    # generate lists to read information on variables
    
    varLabel = list()
    varWeight = list()
        
    # read information on variables (species response to them)

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
    
    for grid in range(0,gridNum):
            
        # read the previously generated landscape object (the grid)
        
        lscape = landscape()
        lscape.pkl_import(path = directoryLandscapes + 'landscape' + str(grid) + '.pkl')
            
        # start with an empty list of distributions

        settlements = []
        
        # loop over all distributions and import them
        
        for year in range(0,yearsNum):
                           
            settlements.append(settlement(landscape = lscape))
            settlements[year].import_from_pkl(path = directorySettlements + subpath + 'settlement' + str(grid) + '_' + str(year) + ".pkl")
                        
        # generate an export object
        
        exp = export(landscape = lscape, settlements = settlements, varLabels = varLabel)
        
        # merge all settlements to one dataset and export it to .txt and .pkl
        
        exp.merge()
        
        # export to .txt and .pkl

        exp.export_to_txt(directoryDatasets + subpath + 'dataset' + str(grid) + ".txt")
        exp.export_to_pkl(directoryDatasets + subpath + 'dataset' + str(grid) + ".pkl")
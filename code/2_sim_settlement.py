
# import packages

import json
import numpy as np
import os

from landscape import Landscape
from settlement import Settlement

from joblib import Parallel, delayed


# function for parallel computing

def parallel_settlements(grid):
    
    # generate a random seed, use it and store it
    
    random_seed = round(np.random.random()*1000000)
    np.random.seed(random_seed)
    file_seed = open(directorySettlements + subpath + 'randomSeeds.txt','a')
    file_seed.write(str(scenario['interactionW']) + "," + str(grid) + "," + str(random_seed) + '\n')
    file_seed.close()
    
    # read the previously generated landscape object (the grid)
    
    lscape = Landscape()
    lscape.pkl_import(path=directoryLandscapes + 'landscape' + str(grid) + '.pkl')
        
    # start with an empty list of distributions
    
    settlements = []
    
    # loop over multiple distributions
    
    for distribution in range(0, nDistributions):
    
        # print message to keep track of the progress
                       
        print("Generating settlement " + str(distribution) + " on grid " + str(grid))
    
        # sample the random number of individuals

        if nIndividualsMin < nIndividualsMax:
            n_individuals = np.random.randint(low=nIndividualsMin, high=nIndividualsMax)
        else:
            n_individuals = nIndividualsMin

        # new settlement object

        settlements.append(Settlement(interaction_w=interactionW, landscape=lscape))
        
        # set the response of the species to the different variables
        
        for index, label in enumerate(varLabel):
            settlements[distribution].set_weight(label=label, weight=varWeight[index])
               
        # place the individuals in the grid using gibbs sampling
        
        settlements[distribution].gibbs(n_individuals=n_individuals, n_iterations=n_individuals * nSelections)

        # save the distribution

        settlements[distribution].export_to_pkl(path=directorySettlements + subpath + 'settlement' + str(grid)
                                                       + '_' + str(distribution) + ".pkl")


# import settings

with open('../settings.json') as f:
    settings = json.load(f)

# extract general settings

directoryLandscapes = settings['directory']['landscapes']
directorySettlements = settings['directory']['settlements']

nGrid = settings['grid']['n']
nDistributions = settings['nDistributions']

# import scenarios

with open('../scenarios.json') as f:
    scenarios = json.load(f)

# main program

nCores = settings['nCores']

# loop over all scenarios

for scenario in scenarios['scenarios']:

    # generate lists to store information on variables
       
    varLabel = list()
    varWeight = list()
    
    # extract information on variables (species' response to them)
    
    for variable in settings['var']:
        varLabel.append(settings['var'][variable]['label'])
        varWeight.append(settings['var'][variable]['weight'])

    interactionW = scenario['interactionW']
    
    # read the range from which the number of individuals is generated
    
    nIndividualsMin = scenario['nIndividualsMin']
    nIndividualsMax = scenario['nIndividualsMax']
        
    # extract the subpath to the scenario worked on

    subpath = scenario['path']

    # extract the number individuals are allowed to choose new sites

    nSelections = scenario['nSelections']
    
    # generate a directory for the scenario if it did not already exist
    
    if not os.path.exists(directorySettlements + subpath):
        os.makedirs(directorySettlements + subpath)
            
    # loop over all grids and perform parallel settlements
    
    results = Parallel(n_jobs=nCores)(delayed(parallel_settlements)(grid) for grid in range(0, nGrid))

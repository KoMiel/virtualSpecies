

### import packages ###

import json
import numpy as np
import os

from landscape import landscape
from settlement import settlement


### import settings ###

with open('../settings.json') as f:
    settings = json.load(f)

# read general settings

directoryLandscapes = settings['directory']['landscapes']
directorySettlements = settings['directory']['settlements']

gridNum = settings['grid']['num']
yearsNum = settings['years']


### import scenarios ###

with open('../scenarios.json') as f:
    scenarios = json.load(f)


### main program ###

# loop over all scenarios

for scenario in scenarios['scenarios']:

    # generate list to store randomSeeds
    
    randomSeeds = list()

    # generate lists to read information on variables
       
    varLabel = list()
    varWeight = list()
    
    # read information on variables (species response to them)
    
    for variable in settings['var']:
        varLabel.append(settings['var'][variable]['label'])
        varWeight.append(settings['var'][variable]['weight'])

    interactionW = scenario['interactionW']
    
    # read the range from which the number of individuals is generated
    
    birdsNumMin = scenario['birdsNumMin']
    birdsNumMax = scenario['birdsNumMax']
        
    # read the subpath to the scenario worked on

    subpath = scenario['path']
    
    # generate a directory for the scenario if it did not already exist
    
    if not os.path.exists(directorySettlements + subpath):
        os.makedirs(directorySettlements + subpath)
            
    # loop over all previously generated grids
    
    for grid in range(0,gridNum):
        
        # generate a random seed and append it to the list
        
        randomSeed = np.random.randint(1000000)
        np.random.seed(randomSeed)
        randomSeeds.append(randomSeed)

        # read the previously generated landscape object (the grid)
        
        lscape = landscape()
        lscape.pkl_import(path = directoryLandscapes + 'landscape' + str(grid) + '.pkl')
            
        # start with an empty list of distributions
        
        settlements = []
        
        # loop over multiple distributions (called year here, but could be any time delay between observations)
        
        for year in range(0,yearsNum):

            # print message to keep track of the progress
                           
            print("Executing settlement on grid " + str(grid) + ", settlement " + str(year))
        
            # sample the random number of individuals (called birds here, but, again, could be other species, too) and initialize a new distribution
            
            birdsNum = np.random.randint(low = birdsNumMin, high = birdsNumMax)
            settlements.append(settlement(interactionW = interactionW, landscape = lscape))
            
            # set the response of the species to the different variables
            
            for index, label in enumerate(varLabel):
                settlements[year].set_weight(label = label, weight = varWeight[index])
                   
            # place the individuals in the grid using gibbs sampling
            
            settlements[year].gibbs(birdsNum = birdsNum, itNum = birdsNum * 10)
                
            # save the distribution
            
            settlements[year].export_to_pkl(path = directorySettlements + subpath + 'settlement' + str(grid) + '_' + str(year) + ".pkl")        

    # store the random seeds
    f = open(directorySettlements + subpath + 'randomSeeds.txt','w')
    
    for line in randomSeeds:
        f.write(str(line) + '\n')
    f.close()
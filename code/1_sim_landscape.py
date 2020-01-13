

### import packages ###

import json
import numpy as np
import os

from landscape import landscape


### import settings ###

with open('../settings.json') as f:
    settings = json.load(f)

# read general settings

directoryLandscapes = settings['directory']['landscapes']

gridNum = settings['grid']['num']
gridSize = settings['grid']['size']

# read variable settings for all continuos variables

varLabel = list()
varSmooth1 = list()
varSmooth2 = list()

for var in settings['var']:
    varLabel.append(settings['var'][var]['label'])
    varSmooth1.append(settings['var'][var]['smooth1'])
    varSmooth2.append(settings['var'][var]['smooth2'])

# read settings for binary variable (for completely hostile sites)

unsuitableSmooth1 = settings['unsuitable']['smooth1']
unsuitableSmooth2 = settings['unsuitable']['smooth2']
unsuitableThreshold = settings['unsuitable']['threshold']

# vector to store random seeds

randomSeeds = list()


### main program ###

# create new directory to store results if it didn't already exist

if not os.path.exists(directoryLandscapes):
    os.makedirs(directoryLandscapes)

# loop over number of grids that are to be generated

for grid in range(0,gridNum):

    # print message to keep track of the progress

    print("Generating grid " + str(grid))    
    
    # generate a random seed and append it to the list
    
    randomSeed = np.random.randint(1000000)
    np.random.seed(randomSeed)
    randomSeeds.append(randomSeed)
        
    # generate a landscape object with binary variable
    
    lscape = landscape(size = gridSize, smooth1 = unsuitableSmooth1, smooth2 = unsuitableSmooth2, threshold = unsuitableThreshold)
    
    #simulate all continuos variables
    
    for index, label in enumerate(varLabel):  
        lscape.sim_pred(label = label, smooth1 = varSmooth1[index], smooth2 = varSmooth2[index])
    
    # save the object
    lscape.pkl_export(path = directoryLandscapes + 'landscape' + str(grid) + '.pkl')

# store all random seeds in a file

f = open(directoryLandscapes + 'randomSeeds.txt','w')

for line in randomSeeds:
    f.write(str(line) + '\n')
f.close()
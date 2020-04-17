
# import packages

import json
import numpy as np
import os

from landscape import Landscape

# import settings

with open('../settings.json') as f:
    settings = json.load(f)

# extract general settings

directoryLandscapes = settings['directory']['landscapes']

gridN = settings['grid']['n']
gridSize = settings['grid']['size']

# extract variable settings for all continuous variables

varLabel = list()
varSmooth1 = list()
varSmooth2 = list()

for var in settings['var']:
    varLabel.append(settings['var'][var]['label'])
    varSmooth1.append(settings['var'][var]['smooth1'])
    varSmooth2.append(settings['var'][var]['smooth2'])

# read settings for binary variable (used for completely hostile sites)

unsuitableSmooth1 = settings['unsuitable']['smooth1']
unsuitableSmooth2 = settings['unsuitable']['smooth2']
unsuitableThreshold = settings['unsuitable']['threshold']

# main program

# create new directory to store results if it didn't already exist

if not os.path.exists(directoryLandscapes):
    os.makedirs(directoryLandscapes)

# loop over number of grids that are to be generated

for grid in range(0, gridN):

    # print message to keep track of the progress

    print("Generating grid " + str(grid))

    # generate a random seed, use it and store it

    randomSeed = round(np.random.random() * 1000000)
    np.random.seed(randomSeed)
    f = open(directoryLandscapes + 'randomSeeds.txt', 'a')
    f.write(str(grid) + "," + str(randomSeed) + '\n')
    f.close()

    # generate a landscape object with binary variable

    lscape = Landscape(size=gridSize, smooth1=unsuitableSmooth1, smooth2=unsuitableSmooth2, threshold=unsuitableThreshold)

    # simulate all continuous variables

    for index, label in enumerate(varLabel):
        lscape.sim_pred(label=label, smooth1=varSmooth1[index], smooth2=varSmooth2[index])

    # save the object

    lscape.pkl_export(path=directoryLandscapes + 'landscape' + str(grid) + '.pkl')

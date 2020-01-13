

### import packages ###

import joblib as jl
import pandas as pd
import numpy as np

from functionalities import check_position_validity, zeros_array


########################
### landscape class ###
########################

class landscape:
    
    # initialization function
    
    def __init__(self,
                 size = 50,
                 smooth1 = 0,
                 smooth2 = 0,
                 threshold = 0):

        # store the input
        
        self.size = size
        
        # generate an empty pandas series for the results
        
        self.data = pd.Series()

        # generate a binary variable (for completely hostile sites)
        
        self.sim_unsuitable(smooth1 = smooth1, smooth2 = smooth2, threshold = threshold)

    # function to generate a binary variable (for completely hostile sites)

    def sim_unsuitable(self,
                       smooth1,
                       smooth2,
                       threshold):

        # call the normal function to generate a layer (the label is only temporary here)
        
        self.sim_pred(label = "Hidden layer", smooth1 = smooth1, smooth2 = smooth2)
        
        # apply a threshold to generate the final layer
        
        self.data["Unsuitable cells"] = self.data["Hidden layer"].copy()
        self.data["Unsuitable cells"][self.data["Unsuitable cells"] < threshold] = 0
        self.data["Unsuitable cells"][self.data["Unsuitable cells"] >= threshold] = 1
        

    # function to generate a continuous variable, potentially spatially correlated (for smooth1 > 0 and smooth2 > 0)

    def sim_pred(self,
                 label,
                 smooth1,
                 smooth2):
        
        # generate a uniform array of grid size
        
        array = np.random.uniform(low = 0, high = 1, size = (self.size, self.size))        
        
        # multiple smoothing steps
        
        for i in range(0, smooth1):
        
            # temp is the smoothened array. norm is a normalization constant for each grid cell to avoid that grid cells at the borders of the grid will be treated differently
            
            temp = zeros_array(self.size)
            norm = zeros_array(self.size)
            
            # apply a smoothing average window
            
            for x in range(0, self.size): # looping over all grid cells
                for y in range(0, self.size):
                                        
                    for innerX in range(-smooth2, smooth2+1): # applying the window
                        for innerY in range(-smooth2, smooth2+1):
                            
                            if (check_position_validity(self.size, x + innerX, y + innerY) == 0):
        
                                norm[x][y] += 1
                                temp[x][y] += array[x + innerX][y + innerY]
                                
            temp /= norm # divide by the number of cells that contributed to the window
            array = temp.copy()
                
        #adjust to a range of 0 to 1 (small offset to not divide by 0
        
        array -= np.min(array) - 0.01
        array /= (np.max(array) - np.min(array) - 0.01)

        self.data[label] = array.copy()


    # function to save the landscape as a .pkl file

    def pkl_export(self,
                   path):
        
        jl.dump((self.data, self.size), path)


    # function to load the landscape from a .pkl file

    def pkl_import(self,
                   path):
        
        self.data, self.size = jl.load(path)
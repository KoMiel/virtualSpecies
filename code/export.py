

### import packages ###

import joblib as jl
import pandas as pd
import numpy as np


########################
### export class ###
########################

class export:
    
    # initialization function
    
    def __init__(self,
                 landscape,
                 settlements,
                 varLabels):
        
        # store the input
        
        self.landscape = landscape
        self.settlements = settlements
        self.varLabels = varLabels

        # generate an empty pandas series for the results

        self.data = pd.Series()
        
        # generate empty lists for variables (label is 1 for presence of the species and 0 for absence)
        
        self.data["Label"] = []
                
        for var in varLabels:
            self.data[var] = []

        # generate a pandas series for the weights
        
        self.weights = pd.Series()

    
    # function to merge together multiple distributions (different sampling events)
    
    def merge(self):
        
        # loop over all distributions
        
        for index, settlement in enumerate(self.settlements):

            # save all presence locations and information on these positions
            
            for posPres in settlement.data["Bird locations"]:
                
                # presence = 1
                
                self.data["Label"].append(1)
                                          
                # loop over all variables
                
                for idx, var in enumerate(self.varLabels):
                    self.data[var].append(self.landscape.data[var][posPres[0]][posPres[1]])
            
            # save all absence locations and information on these positions
            
            for posAbs in settlement.data["Empty habitat"]:
                
                # absence = 0

                self.data["Label"].append(0)
                                                    
                # loop over all variables

                for idx, var in enumerate(self.varLabels):
                    self.data[var].append(self.landscape.data[var][posAbs[0]][posAbs[1]])
     

    # function for export to .txt (this is the data file that is used for the analysis)
    
    def export_to_txt(self,
                      path):
        
        # generate empty vectors for positions and the sample event the observation belongs to
        
        X = []
        Y = []
        
        year = []
        
        # loop over all presences sampling events
        
        for index, settlement in enumerate(self.settlements):
                                    
            # loop over all presences
            
            for posPres in settlement.data["Bird locations"]:
                
                # store the position and the sampling event
                
                X.append(posPres[0])
                Y.append(posPres[1])
                year.append(index)
                
            # loop over all absences
            
            for posAbs in settlement.data["Empty habitat"]:
                
                # store the position and the sampling event

                X.append(posAbs[0])
                Y.append(posAbs[1])
                year.append(index)
        
        # store presence/absence indicator, position and sampling event in array
        
        array = np.stack((np.asarray(self.data["Label"]), np.asarray(X), np.asarray(Y), np.asarray(year)))
        
        # add the information on the variables by looping over them
        
        for var in self.varLabels:
            array = np.append(array, [np.asarray(self.data[var])], axis = 0)

        # transpose the array for more logical directions
        
        array = np.transpose(array)

        # save the array to file
        
        np.savetxt(path, array)                            

    
    # function to save the object as a .pkl file

    def export_to_pkl(self,
                      path):

        jl.dump((self.data, self.landscape, self.settlements, self.varLabels, self.weights), path)


    # function to load the object from a .pkl file

    def import_from_pkl(self,
                    path):
        
        self.data, self.landscape, self.settlements, self.varLabels, self.weights = jl.load(path)
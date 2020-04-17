
# import packages

import joblib as jl
import pandas as pd
import numpy as np

################
# export class #
################


class Export:
    
    # initialization function
    
    def __init__(self,
                 landscape,
                 settlements,
                 var_labels):
        
        # store the input
        
        self.landscape = landscape
        self.settlements = settlements
        self.var_labels = var_labels

        # generate an empty pandas series for the data

        self.data = pd.Series()
        
        # generate empty lists for variables (label is 1 for presence of the species and 0 for absence)
        
        self.data["Label"] = []
                
        for var in var_labels:
            self.data[var] = []

        # generate a pandas series for the weights
        
        self.weights = pd.Series()
    
    # function to merge multiple distributions (different sampling events) into one
    
    def merge(self):
        
        # loop over all distributions
        
        for index, settlement in enumerate(self.settlements):

            # save all presence locations and information on these positions
            for posPres in settlement.data["Presence locations"]:
                
                # presence = 1
                
                self.data["Label"].append(1)
                                          
                # loop over all variables
                
                for idx, var in enumerate(self.var_labels):
                    self.data[var].append(self.landscape.data[var][posPres[0]][posPres[1]])
            
            # save all absence locations and information on these positions
            
            for posAbs in settlement.data["Empty habitat"]:
                
                # absence = 0

                self.data["Label"].append(0)
                                                    
                # loop over all variables

                for idx, var in enumerate(self.var_labels):
                    self.data[var].append(self.landscape.data[var][posAbs[0]][posAbs[1]])

    # function for export to .txt (this is the data file that is used for the analysis)
    
    def export_to_txt(self,
                      path):
        
        # generate empty vectors for positions and the sample event the observation belongs to
        
        x = []
        y = []
        distribution = []
        
        # loop over all presences sampling events
        
        for index, settlement in enumerate(self.settlements):
                                    
            # loop over all presences
            
            for posPres in settlement.data["Presence locations"]:
                
                # store the position and the sampling event
                
                x.append(posPres[0])
                y.append(posPres[1])
                distribution.append(index)
                
            # loop over all absences
            
            for posAbs in settlement.data["Empty habitat"]:
                
                # store the position and the sampling event

                x.append(posAbs[0])
                y.append(posAbs[1])
                distribution.append(index)
        
        # store presence/absence indicator, position and sampling event in array
        
        array = np.stack((np.asarray(self.data["Label"]), np.asarray(x), np.asarray(y), np.asarray(distribution)))
        
        # add the information on the variables by looping over them
        
        self.var_labels = sorted(self.var_labels)

        for var in self.var_labels:
            array = np.append(array, [np.asarray(self.data[var])], axis=0)

        # transpose the array for more logical directions
        
        array = np.transpose(array)

        # save the array to file
        
        np.savetxt(path, array)                            

    # function to save the object as a .pkl file

    def export_to_pkl(self,
                      path):

        jl.dump((self.data, self.landscape, self.settlements, self.var_labels, self.weights), path)

    # function to load the object from a .pkl file

    def import_from_pkl(self,
                    path):
        
        self.data, self.landscape, self.settlements, self.var_labels, self.weights = jl.load(path)

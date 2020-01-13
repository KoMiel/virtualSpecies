

### import packages ###

import joblib as jl
import pandas as pd
import numpy as np

from functionalities import check_position_validity, distance_function, zeros_array


########################
### settlement class ###
########################

class settlement:
    
    # initialization function
    
    def __init__(self,
                 landscape,
                 interactionW = 0):

        # import the landscape
        
        self.landscape = landscape
                
        # generate pandas series for results (data) and imported weights
        
        self.data = pd.Series()
        self.weights = pd.Series()

        # generate an empty data structure for the locations of the individuals

        self.data["Bird locations"] = []

        # generate an empty array for conspecific interaction and set the corresponding weight

        array = zeros_array(self.landscape.size)
        self.data["Conspecific interaction"] = array.copy()
        
        self.weights["Conspecific interaction"] = interactionW
        
        
    # function to set variable slopes (determines the response of individuals to different habitat factors)
    
    def set_weight(self,
                   label,
                   weight):
        
        self.weights[label] = weight
        self.calc_probs()


    # function that executes a sequential settlement
    
    def sequential(self,
                   birdsNum):

        # place the required number of individuals, updating the settlement probabilities after each one
        
        for i in range(0, birdsNum):
            self.calc_probs()
            self.settle_bird()

        # generate a list of all empty positions
        
        self.empty_habitat()
      

    # function to update the conspecific interaction array if a new individual is placed in the landscape
    
    def add_interaction(self,
                        posX,
                        posY):
        
        # loop over all positions that are potentially relevant
        
        for x in range(- self.landscape.size, self.landscape.size + 1):
            for y in range(- self.landscape.size, self.landscape.size + 1):
                
                # check whether the position is actually on the grid
                
                if (check_position_validity(size = self.landscape.size, x = x + posX, y = y + posY) == 0):
                    
                    # add the conspecific interaction contribution
                    
                    self.data["Conspecific interaction"][x + posX][y + posY] += distance_function(x = x, y = y)


    # function to update conspecific interaction if an individual is removed from the landscape (exactly the opposite of the add-function, comments see above)
    
    def remove_interaction(self,
                           posX,
                           posY):

        for x in range(- self.landscape.size, self.landscape.size + 1):
            for y in range(- self.landscape.size, self.landscape.size + 1):
                
                if (check_position_validity(size = self.landscape.size, x = x + posX, y = y + posY) == 0):
                    self.data["Conspecific interaction"][x + posX][y + posY] -= distance_function(x = x, y = y)


    # function to choose a settlement location for a single individual

    def settle_bird(self):
        
        # calculate the sum of all settlement probabilities in y-direction to choose a spot at random below
        
        probSum = np.sum(self.data["Settlement probability"], axis = 0)

        # choose one spot randomly (according to probabilities, consecutively select y and x-coordinates)
        
        posY = np.random.choice(a = self.landscape.size, p = probSum/sum(probSum))
        posX = np.random.choice(a = len(self.data["Settlement probability"][:,posY]), p = self.data["Settlement probability"][:,posY]/sum(self.data["Settlement probability"][:,posY]))
        
        # add individual to the grid, save the location
        
        self.data["Bird locations"].append([posX, posY])
        
        # update the conspecific interaction
        
        self.add_interaction(posX = posX, posY = posY)


    # function to calculate the settlement probabilities from both conspecific interaction and landscape characteristics
                
    def calc_probs(self):

        # generate an array to store the probabilities

        array = np.ones((self.landscape.size, self.landscape.size))

        # all positions with an individual can't be chosen again. So set their probabilities to 0
        
        if len(self.data["Bird locations"]) > 0:
            array[np.array(self.data["Bird locations"])[:,0], np.array(self.data["Bird locations"])[:,1]] = 0
            
        array[self.landscape.data["Unsuitable cells"] == 0] = 0
        
        # calculate the landscape quality
        
        for k in self.weights.index[:]: #by looping over all relevant factors and summing their influence

            if (k == "Conspecific interaction"): #this is the specific case of conspecific interaction
                array *= np.exp(self.data[k] * self.weights[k])
            else: #all others are habitat factors
                array *= np.exp(self.landscape.data[k] * self.weights[k])                            

        # convert the landscape quality to probabilities and store
        
        array /= array.sum()
        self.data["Settlement probability"] = array.copy()
            

    # function that applies gibbs sampling to generate a stable distribution
    
    def gibbs(self,
              birdsNum,
              itNum):

        # do the sequential settlement routine first
        
        self.sequential(birdsNum = birdsNum)
        
        # loop over gibbs sampling steps
        
        for it in range(itNum):
            
            # randomly pick an individual and remove it from the grid
            
            num = np.random.randint(low = 0, high = birdsNum)
            pos = self.data["Bird locations"][num]
            del self.data["Bird locations"][num]
            
            # remove its influence on other individuals
            
            self.remove_interaction(posX = pos[0], posY = pos[1])
            
            # recalculate the settlement probabilities and choose a pick spot where to place the individual
            
            self.calc_probs()
            self.settle_bird()        

        # get a list of all locations without an individual
        
        self.empty_habitat()


    # function to compute a list of all empty spots that could potentially have been chosen (necessary as absences for the GLMM)

    def empty_habitat(self):

        # generate an empty list
        
        lst = []

        #loop over all locations, if it is empty but potentially could have been filled add it to list (i.e., the completely hostile spots are left out of the list)
        
        for x in range(self.landscape.size):
            for y in range(self.landscape.size):
                    
                if (not [x, y] in self.data["Bird locations"] and self.landscape.data["Unsuitable cells"][x][y] > 0):
                    lst.append([x, y])
        
        # generate an array from the list
        
        array = np.array(lst)
                    
        # store the array
        
        self.data["Empty habitat"] = array.copy()
                    
    # function to save the distribution as a .pkl file

    def export_to_pkl(self,
                      path):

        # save the object

        jl.dump((self.data, self.landscape, self.weights), path)


    # function to load the distribution from a .pkl file

    def import_from_pkl(self,
                        path):

        self.data, self.landscape, self.weights = jl.load(path)

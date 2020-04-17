
# import packages

import joblib as jl
import pandas as pd
import numpy as np

from functionalities import check_position_validity, distance_function, zeros_array

####################
# settlement class #
####################


class Settlement:
    
    # initialization function
    
    def __init__(self,
                 landscape,
                 interaction_w=0):

        # import the landscape
        
        self.landscape = landscape
                
        # generate pandas series for results (data) and imported weights
        
        self.data = pd.Series()
        self.weights = pd.Series()

        # generate an empty data structure for the locations of the individuals

        self.data["Presence locations"] = []

        # generate an empty array for conspecific interaction and set the corresponding weight

        array = zeros_array(self.landscape.size)
        self.data["Conspecific interaction"] = array.copy()
        self.weights["Conspecific interaction"] = interaction_w
        
    # function to set variable slopes (determines the response of individuals to different habitat factors)
    
    def set_weight(self,
                   label,
                   weight):
        
        self.weights[label] = weight

    # function that performs a sequential settlement
    
    def sequential(self,
                   n_individuals):

        # place the required number of individuals, updating the settlement probabilities after each one
        
        for i in range(0, n_individuals):
            self.update_probabilities()
            self.place_individual()

        # generate a list of all empty positions
        
        self.empty_habitat()

    # function to update the conspecific interaction array if a new individual is placed in the landscape
    
    def add_interaction(self,
                        pos_x,
                        pos_y):
        
        # loop over all positions that are potentially relevant
        
        for x in range(- self.landscape.size, self.landscape.size + 1):
            for y in range(- self.landscape.size, self.landscape.size + 1):
                
                # check whether the position is actually on the grid
                
                if check_position_validity(size=self.landscape.size, x=x + pos_x, y=y + pos_y) == 0:
                    
                    # add the conspecific interaction contribution
                    
                    self.data["Conspecific interaction"][x + pos_x][y + pos_y] += distance_function(x=x, y=y)

    # function to update conspecific interaction if an individual is removed from the landscape
    # (exactly the opposite of the add-function, comments see above)
    
    def remove_interaction(self,
                           pos_x,
                           pos_y):

        for x in range(- self.landscape.size, self.landscape.size + 1):
            for y in range(- self.landscape.size, self.landscape.size + 1):
                
                if check_position_validity(size = self.landscape.size, x=x + pos_x, y=y + pos_y) == 0:

                    self.data["Conspecific interaction"][x + pos_x][y + pos_y] -= distance_function(x=x, y=y)

    # function to select a settlement location for a single individual (with random component)

    def place_individual(self):
        
        # calculate the sum of all settlement probabilities in y-direction to choose a spot at random below
        
        probability_sum = np.sum(self.data["Settlement probability"], axis=0)

        # choose one spot randomly (according to probabilities, consecutively select y and x-coordinates)
        
        pos_y = np.random.choice(a=self.landscape.size, p=probability_sum/sum(probability_sum))
        pos_x = np.random.choice(a=len(self.data["Settlement probability"][:, pos_y]),
                                 p=self.data["Settlement probability"][:, pos_y]/sum(self.data["Settlement probability"]
                                                                                     [:, pos_y]))
        
        # add individual to the grid, save the location
        
        self.data["Presence locations"].append([pos_x, pos_y])

        # update the conspecific interaction
        
        self.add_interaction(pos_x=pos_x, pos_y=pos_y)

    # function to calculate the settlement probabilities taking into account
    # both conspecific interaction and landscape characteristics

    def update_probabilities(self):

        # generate an array to store the probabilities

        array = np.ones((self.landscape.size, self.landscape.size))

        # all positions with an individual can't be chosen again. So we set their probabilities to 0 from the start
        
        if len(self.data["Presence locations"]) > 0:
            array[np.array(self.data["Presence locations"])[:, 0], np.array(self.data["Presence locations"])[:, 1]] = 0

        # also exclude unsuitable cells

        array[self.landscape.data["Unsuitable cells"] == 0] = 0
        
        # calculate the landscape quality, by looping over all relevant factors and summing their influence
        
        for k in self.weights.index[:]:

            # conspecific interaction

            if k == "Conspecific interaction":
                array *= np.exp(self.data[k] * self.weights[k])

            # all habitat factors
            else:
                array *= np.exp(self.landscape.data[k] * self.weights[k])

        # convert the landscape quality to probabilities and store

        array /= array.sum()
        self.data["Settlement probability"] = array.copy()

    # function that performs gibbs sampling to generate a stable distribution

    def gibbs(self,
              n_individuals,
              n_iterations):

        # perform a sequential settlement routine first
        
        self.sequential(n_individuals=n_individuals)
        
        # loop over gibbs sampling steps
        
        for it in range(n_iterations):
            
            # randomly pick an individual and remove it from the grid
            
            individual = np.random.randint(low=0, high=n_individuals)
            pos = self.data["Presence locations"][individual]
            del self.data["Presence locations"][individual]
            
            # remove its influence on other individuals
            
            self.remove_interaction(pos_x=pos[0], pos_y=pos[1])
            
            # recalculate the settlement probabilities and choose a spot where to place the individual
            
            self.update_probabilities()
            self.place_individual()

        # get a list of all locations without an individual
        
        self.empty_habitat()

    # function to compute a list of all empty spots that could potentially have been chosen
    # (input for logistic regression)

    def empty_habitat(self):

        # generate an empty list
        
        lst = []

        # loop over all locations, if it is empty but potentially could have been filled add it to list
        # (i.e., the completely hostile spots are left out of the list)
        
        for x in range(self.landscape.size):
            for y in range(self.landscape.size):
                    
                if not [x, y] in self.data["Presence locations"] and self.landscape.data["Unsuitable cells"][x][y] > 0:
                    lst.append([x, y])
        
        # generate an array from the list
        
        array = np.array(lst)
                    
        # store the array
        
        self.data["Empty habitat"] = array.copy()
                    
    # function to save a distribution as a .pkl file

    def export_to_pkl(self,
                      path):

        # save the object

        jl.dump((self.data, self.landscape, self.weights), path)

    # function to load a distribution from a .pkl file

    def import_from_pkl(self,
                        path):

        self.data, self.landscape, self.weights = jl.load(path)

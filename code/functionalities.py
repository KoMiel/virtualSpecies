

### import packages ###

import math
import matplotlib.pyplot as plt
import numpy as np


############################
### function definitions ###
############################


# function to check whether (x,y) is in grid of shape [0 to size, 0 to size], returns 1 if it is not

def check_position_validity(size,
                            x,
                            y):
    
    if (x >= size):
        return 1
    
    if (x < 0):
        return 1
    
    if (y >= size):
        return 1
    
    if (y < 0):
        return 1
    
    return 0


# function to calculate a value that exponentially decreases with the distance (used for conspecific interaction)

def distance_function(x,
                      y):

    # check whether positions are the same
    
    if ((x*x + y*y) == 0):
        return 0
    
    # else calculate the function
    
    return np.exp(-np.sqrt((x*x + y*y)))


# function to generate an array full of zeros

def zeros_array(size):
    
    return np.zeros((size, size))
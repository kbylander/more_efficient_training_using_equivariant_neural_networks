import numpy as np
import matplotlib.pyplot as plt

"""
Returns max accuracy and given epoch and accuracy and epoch of convergence for input list
"""

def converged_at(data):
#Finds the first epoch with at least 95% of the highest accuracy in the measure, or if nonexisting: set variables to 0
    max_acc=np.amax(data)
    index_max_acc=np.argmax(data)
    converged_at=[(index,value) for index,value in enumerate(data) if (value >= 0.95*max_acc and index != index_max_acc)]
    
    if len(converged_at) >0:
        conv_ind,conv_acc=converged_at[0]
    else:
        conv_ind=0
        conv_acc=0
    return (conv_ind,conv_acc,index_max_acc,max_acc)

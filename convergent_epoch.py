import numpy as np
import matplotlib.pyplot as plt

"""
Returns max accuracy and given epoch and accuracy and epoch of convergence for input list
"""

def converged_at(data):
#Finds the first epoch with at least 95% of the highest accuracy in the measure
    max_acc=np.amax(data)
    index_max_acc=np.argmax(data)
    print(data)
    converged_at=[(index,value) for index,value in enumerate(data) if (value >= 0.95*max_acc and index != index_max_acc)]
    conv_ind,conv_acc=converged_at[0]
    return (conv_ind,conv_acc,index_max_acc,max_acc)

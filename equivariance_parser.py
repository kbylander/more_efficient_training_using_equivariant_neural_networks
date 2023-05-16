import torch
import numpy as np
import pandas as pd 
import ast

"""
Method that parse the equivariance-test and outputs a csv file of the per class standard deviation of D4- augmentation of an image 
The output is in the corresponding folder

Need to remove linebreaks in the model outputs for each class

Add your corresponding path
"""

#select input-filemame
in_filename='/mimer/NOBACKUP/groups/naiss2023-22-69/exjobb/results_correct_data/g_custom/C8_LR7e-05_BS64_WD001_ep300_dir1/log'

#creates output filename
out_filename=in_filename[:-3] +'equivariance.csv'
print('saved at:\n ',out_filename)

#Calculates the per-class mean percentage error for model output for every augmentation

#read and remove unnecessary data
data=open(in_filename,'r')
data=data.readlines()[-11:-3]
parsed_data=[]

print()

print('Model output for every D4-augmentation: ')
for index,value in enumerate(data):
    ind=value.find('tensor')
    tmp=ast.literal_eval(value[ind+8:-20])
    parsed_data.append(tmp)
    print(tmp)

print()

parsed_data=np.asarray(parsed_data)

#to test C4 equivariance instead of D4
#parsed_data=parsed_data[:4,:]

mean=np.mean(parsed_data,axis=0)
data=[0 for i in range(14)]

for i,v in enumerate(parsed_data.T):
    for val in v:
        data[i]=+abs(mean[i]-val)
    data[i]=100/(8*mean[i])*data[i]
        

print('The resulting data: \n',data)
data.append(np.mean(data,axis=0))
#append the mean class output deviation
np.savetxt(out_filename, data,delimiter=",",fmt='%.2g')
print('Finished script')

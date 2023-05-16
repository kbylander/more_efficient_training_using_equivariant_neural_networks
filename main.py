#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys

import numpy as np
np.set_printoptions(threshold=sys.maxsize)

import torch
import torch.nn as nn
import matplotlib.pyplot as pyplot
import pickle

from torch.utils.data import Dataset
from torchvision.transforms import RandomRotation
from torchvision.transforms import Pad
from torchvision.transforms import ToTensor
from torchvision.transforms import Compose

import torchvision.transforms.functional as TF
import random

from dataset import TEM_dataset

from PIL import Image

#set seed, may still use non-deterministic functions
torch.manual_seed(0)

import time
current_GMT = time.time()
print("Time: ", current_GMT)


device = 'cuda' if torch.cuda.is_available() else 'cpu'

if torch.cuda.is_available():
    print ("cuda to the rescue")
    print('Running on',torch.cuda.device_count(),torch.cuda.get_device_name())
    rank=list(range(4*torch.cuda.device_count()))
else:
    print ("cuda not available")

NAME_CLASSES = ['Adenovirus', 'Astrovirus', 'CCHF', 'Cowpox', 'Ebola', 'Influenza', 'Lassa', 'Marburg', 'Nipah', 'Norovirus', 'Orf', 'Papilloma', 'Rift Valley', 'Rotavirus']

totensor = ToTensor()

LEARNING_RATE = 7e-5
BATCH_SIZE = 64
WEIGHT_DECAY = 0.01
nr_epochs = 300
angular_resolution = 4
no_channels = 32

from models import CUSTOM, G_CUSTOM, G_VGG16, VGG16 and versions without batch normalisation
#Select CUSTOM or G_CUSTOM model 
model=G_CUSTOM(14,angular_resolution,no_channels).to(device)
#model.load_state_dict(torch.load('/mimer/NOBACKUP/groups/naiss2023-22-69/exjobb/results/g_custom/non_augmented_C6_LR7e-05_BS64_WD001_ep500_dir2/best_state_dict_ep89'))

loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

#print settings and paths
print('learning rate:', LEARNING_RATE)
print('batch_size:', BATCH_SIZE)
print('weight decay:', WEIGHT_DECAY)
print('Number of epochs:', nr_epochs)
print('Number of starting channels:',no_channels)
print('torch.manual_seed(0)')


ABS_PATH='/mimer/NOBACKUP/groups/naiss2023-22-69/exjobb/results_seed0/g_custom/non_augmented_D' + str(angular_resolution) + '_LR'+ str(LEARNING_RATE) + '_BS' + str(BATCH_SIZE) + '_WD' + str(WEIGHT_DECAY).replace('.','') +'_ep' + str(nr_epochs)+'_dir2'

filename_cf_test_best= ABS_PATH + '/cf_test_best.png'
filename_cf_train_best = ABS_PATH + '/cf_train_best.png'
filename_cf_test_final_epoch= ABS_PATH + '/cf_test_final_epoch.png'
filename_cf_train_final_epoch = ABS_PATH + '/cf_train_finalepoch.png'
filename_acc_train_test = ABS_PATH + '/accuracy.png'
filename_best_state_dict = ABS_PATH + '/best_state_dict'
filename_final_state_dict = ABS_PATH + '/final_state_dict'
filename_acc_csv= ABS_PATH + '/accuracy.csv'

print('files found at:')
print(filename_cf_test_best)
print(filename_cf_train_best) 
print(filename_cf_test_final_epoch)
print(filename_cf_train_final_epoch)
print(filename_acc_train_test)

print('model is:')

print(model)

pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Number of parameters: {pytorch_total_params}')


def test_model(model: torch.nn.Module, x: Image):
    # evaluate the `model` on 4 rotated versions of the input image `x`
    model.eval()
    print('entered test_model')    
    print()
    print('#'*30)
    header = 'angle |  ' + '  '.join(["{:6d}".format(d) for d in range(14)])
    print(header)
    with torch.no_grad():
        for r in range(8):
            if r <= 3:
                angle = r * (90)
                x_transformed = totensor(x.rotate(angle, Image.BILINEAR)).reshape(1, 1, 256, 256)
            else:
                #flip-rotate the image
                angle = (r-4) * (90)
                x_transformed = totensor(x.rotate(angle,Image.BILINEAR).transpose(Image.FLIP_LEFT_RIGHT)).reshape(1,1,256,256)
            x_transformed = x_transformed.to(device)

            y = model(x_transformed)
            #y = y.to('cpu').numpy().squeeze()            
            if r <=3:
                print("{:} \t: {}".format(int(angle), y))
            elif r <=5:
                print("{}{} \t: {}".format('flip',int(angle), y))
            else:
                print("{}{} : {}".format('flip',int(angle), y))

    print('#'*100)
    print()

    
#test equivariance before training
tem_test = TEM_dataset(mode='test')
x, y = next(iter(tem_test))
test_model(model, x)

del tem_test

train_transform = Compose([totensor])

print('loading training set')
tem_train = TEM_dataset(mode='train', transform=train_transform)
train_loader = torch.utils.data.DataLoader(tem_train, batch_size=BATCH_SIZE, shuffle=True)
print('finished loading training set')

test_transform = Compose([totensor])
print('loading test set')
tem_test = TEM_dataset(mode='test', transform=test_transform)
test_loader = torch.utils.data.DataLoader(tem_test, batch_size=BATCH_SIZE, shuffle=True)
print('finished loading test set')

#for parameter in model.parameters():
#    print(parameter)

print('start time:')
start_time = time.time()

xb = []
yb = []
times = np.linspace(1 , nr_epochs, nr_epochs)

predicted_training=[]
target_training=[]
predicted_test=[]
target_test=[]
predicted_training_best=[]
target_training_best=[]
predicted_test_best=[]
target_test_best=[]

i_best=0
epoch_best=0
for epoch in range(nr_epochs):
    model.train()
    current_GMT = time.time()
    times[int(epoch)]=current_GMT
    print("Time: ", current_GMT)
    for i, (x, t) in enumerate(train_loader):

        optimizer.zero_grad()

        x = x.to(device)
        t = t.to(device)
        y = model(x)

        loss = loss_function(y, t)

        loss.backward()

        optimizer.step()
    
    if epoch % 1 == 0:
        total = 0
        correct = 0
        with torch.no_grad():
            model.eval()
            for i, (x, t) in enumerate(test_loader):

                x = x.to(device)
                t = t.to(device)
                y = model(x)

                _, prediction = torch.max(y.data, 1)
                total += t.shape[0]
                correct += (prediction == t).sum().item()
                if epoch+1 == nr_epochs:
                    [predicted_test.append(int(i)) for i in prediction]
                    [target_test.append(int(i)) for i in t]
        print(f"epoch {epoch+1}:")
        print(f"test accuracy: {correct/total*100.}")
        yb.append(correct/total*100)
        if((correct/total*100)>i_best):
            torch.save(model.state_dict(), filename_best_state_dict)
            i_best = correct/total*100
            epoch_best=epoch
            target_test_best=target_test
            predicted_test_best = predicted_test
        total = 0
        correct = 0
        with torch.no_grad():
            model.eval()
            for i, (x, t) in enumerate(train_loader):

                x = x.to(device)
                t = t.to(device)
                y = model(x)

                _, prediction = torch.max(y.data, 1)
                total += t.shape[0]
                correct += (prediction == t).sum().item()
                if epoch+1 == nr_epochs:
                    [predicted_training.append(int(i)) for i in prediction]
                    [target_training.append(int(i)) for i in t]
                if epoch == epoch_best:
                    predicted_training_best = predicted_training
                    target_training_best = target_training
        print(f"train accuracy: {correct/total*100.}")
        xb.append(correct/total*100)

torch.save(model.state_dict(),filename_final_state_dict)

end_time = time.time()
print("Time: ", end_time)

print(f"Highest accuracy of test set: {i_best} at epoch: {epoch_best}")
print(f'model converged at epoch: {round(epoch_best*0.95,1)}')

import seaborn as sn
from sklearn.metrics import confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)

#creates and saves a confusion matrix at given filename path
def plot_confusion_matrix(true,pred,filename,name,class_names=NAME_CLASSES):    
    fig,ax=plt.subplots()
    cf_matrix=confusion_matrix(true,pred)
    df_cm=pd.DataFrame((cf_matrix/np.sum(cf_matrix,axis=1)[:,None]*100).round(decimals=2),index = class_names, columns = class_names)    
    plt.figure(figsize=(30,25),dpi=300)
    sn.heatmap(df_cm,annot=True,cmap='Greens',cbar=False,annot_kws={"size": 15})
    plt.tick_params(axis='both', which='both', labelsize=25)
    ax=plt.gca()
    ax.set_xticklabels(class_names,rotation=45,ha='right',rotation_mode='anchor')
    ax.set_yticklabels(class_names,rotation=45,ha='right',rotation_mode='anchor')
    plt.title(name,fontsize=40)
    plt.xlabel("Predicted class",fontsize=30, labelpad=2)
    plt.ylabel("True class",fontsize=30)
    plt.savefig(filename)

plot_confusion_matrix(target_training,predicted_training,filename_cf_train_final_epoch,name='Confusion matrix of train data at the final epoch')
plot_confusion_matrix(target_test,predicted_test,filename_cf_test_final_epoch,name='Confusion matrix of the test data at the final epoch')
plot_confusion_matrix(target_training_best,predicted_training_best,filename_cf_train_best,name='Confusion matrix of train data at the highest test accuracy')
plot_confusion_matrix(target_test_best,predicted_test_best,filename_cf_test_best,name='Confusion matrix of the test data at the highest test accuracy')

#Plots accuracy plot for both cases
import matplotlib.pyplot as plt
import seaborn as sn
import numpy as np
from convergent_epoch import converged_at

conv_ind,conv_acc,max_ind,max_acc=converged_at(yb)
conv_acc=int(round(conv_acc))
max_acc=int(round(max_acc))
conv_ind=int(conv_ind)
max_ind=int(max_ind)
colors={'Convergence':'C4','Max':'C5'}
print(conv_ind,max_ind)
import pandas as pd
#data=pd.DataFrame({'Test':data[0],'Train':data[1]})
data=pd.DataFrame({'test acc':yb,'train acc':xb,'test index':list(range(yb)),'train index':list(range(len(xb)),'max acc':max_acc,'max ind':max_ind,'conv acc':conv_acc,'conv ind':conv_ind})
#markers=pd.DataFrame({'acc':[max_acc,conv_acc],'ind':[max_ind,conv_ind],'color':[3,4]})
with sn.axes_style('darkgrid'):
    g=sn.despine()
    g=sn.scatterplot(data=data,x='max ind',y='max acc',color='C3',markers='o',label='Highest',zorder=4,alpha=0.6)
    g=sn.scatterplot(data=data,x='conv ind',y='conv acc',color='C2',markers='o',label='Converge',zorder=3,alpha=0.6)
    g=sn.lineplot(data=data,x='test index',y='test acc',color='C0',label='Test',alpha=0.8,zorder=2)
    g=sn.lineplot(data=data,x='train index',y='train acc',color='C1',label='Train',alpha=0.8,zorder=1)

    g.legend(loc='lower right')
    g.set_xlabel('Epoch')
    g.set_ylabel('Accuracy')

    # #g=sn.lineplot(x=range(len(xb)),y={'train':yb})
    #g.legend(labels=['Test','Train','Converge','Max'])
    g.set_yticks(list(range(0,101,10)))
    g.set(ylim=(0, 105))

plt.savefig(filename_acc_train_test)

import csv
#save test and training accuracy, first line: test, second line: training
with open(filename_acc_csv,'w') as f:
    file=csv.writer(f)
    file.writerow(yb)
    file.writerow(xb)

#Test equivariance again on trained model
final_tem_test = TEM_dataset(mode='test')
x, y = next(iter(final_tem_test))
test_model(model, x)

print(f'total time for script was: {end_time-start_time} seconds (or {(end_time-start_time)/(60**2)} hours) ')

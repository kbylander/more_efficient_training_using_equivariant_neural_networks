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


from PIL import Image

import time
current_GMT = time.time()
print("Time: ", current_GMT)
NAME_CLASSES = ['Adenovirus', 'Astrovirus', 'CCHF', 'Cowpox', 'Ebola', 'Influenza', 'Lassa', 'Marburg', 'Nipah', 'Norovirus', 'Orf', 'Papilloma', 'Rift Valley', 'Rotavirus']


device = 'cuda' if torch.cuda.is_available() else 'cpu'

if torch.cuda.is_available():
    print ("cuda to the rescue")
    print('Running on',torch.cuda.device_count(),torch.cuda.get_device_name())
else:
    print ("cuda not available")


class MyRotationTransform:
    """Rotate by one of the given angles."""

    def __init__(self, angles):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return TF.rotate(x, angle)

totensor = ToTensor()

class TEM_dataset(Dataset):
    
    def __init__(self, mode, transform=None):
        assert mode in ['train','validation', 'test','augmented_train']
            
        if mode == "train":
            file='/mimer/NOBACKUP/groups/naiss2023-22-69/data/TEM-virus/Cells_train_rot90flipaug_93samplesperclass_14classes.txt'
        if mode == 'augmented_train':
            file='/mimer/NOBACKUP/groups/naiss2023-22-69/data/TEM-virus/Cells_train_noaug_93samplesperclass_14classes.txt'
        elif mode == 'validation':
            file='/mimer/NOBACKUP/groups/naiss2023-22-69/data/TEM-virus/Cells_validation_14classes.txt'
        else:
            file='/mimer/NOBACKUP/groups/naiss2023-22-69/data/TEM-virus/Cells_test_14classes.txt'

        
        self.transform = transform

        data = np.loadtxt(file, delimiter=' ')
        
        if mode == "train":
            self.images = data[:, :-1].reshape(-1, 256, 256).astype(np.float32)
            self.labels = data[:, -1].astype(np.int64)
            self.num_samples = len(self.labels)
        else:
            self.images = data[:, :-1].reshape(-1, 256, 256).astype(np.float32)
            self.labels = data[:, -1].astype(np.int64)
            self.num_samples = len(self.labels)    
        
    def __getitem__(self, index):
        image, label = self.images[index], self.labels[index]
        image = Image.fromarray(image)
        if self.transform is not None:
            image = self.transform(image)
        return image, label
    
    def __len__(self):
        return len(self.labels)

totensor = ToTensor()

LEARNING_RATE=7e-5
BATCH_SIZE=64
WEIGHT_DECAY=0.01
nr_epochs=300

from models import CUSTOM, G_CUSTOM,G_VGG16
#from model import gtest
#Select CUSTOM or G_CUSTOM model 
model=CUSTOM(14).to(device)
loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

#print settings and paths
print('learning rate:', LEARNING_RATE)
print('batch_size:', BATCH_SIZE)
print('weight decay:', WEIGHT_DECAY)
print('Number of epochs:', nr_epochs)
ABS_PATH='/mimer/NOBACKUP/groups/naiss2023-22-69/exjobb/results/custom/LR'+ str(LEARNING_RATE) + '_BS' + str(BATCH_SIZE) + '_WD' + str(WEIGHT_DECAY).replace('.','') +'_ep' + str(nr_epochs)
#ABS_PATH='/cephyr/users/karlby/Alvis/results/test_cnn'
filename_cf_test_best= ABS_PATH + '/cf_test_best.png'
filename_cf_train_best = ABS_PATH + '/cf_train_best.png'
filename_cf_test_final_epoch= ABS_PATH + '/cf_test_final_epoch.png'
filename_cf_train_final_epoch = ABS_PATH + '/cf_train_finalepoch.png'
filename_acc_train_test = ABS_PATH + '/accuracy.png'
print('files found at:')
print(filename_cf_test_best)
print(filename_cf_train_best) 
print(filename_cf_test_final_epoch)
print(filename_cf_train_final_epoch)
print(filename_acc_train_test)

print('model is:')

print(model)

def test_model(model: torch.nn.Module, x: Image):
    # evaluate the `model` on 4 rotated versions of the input image `x`
    model.eval()
    print('entered test_model')    
    print()
    print('##########################################################################################')
    header = 'angle |  ' + '  '.join(["{:6d}".format(d) for d in range(14)])
    print(header)
    with torch.no_grad():
        for r in range(4):
            x_transformed = totensor(x.rotate(r*90., Image.BILINEAR)).reshape(1, 1, 256, 256)
            x_transformed = x_transformed.to(device)

            y = model(x_transformed)
            #y = y.to('cpu').numpy().squeeze()
            
            angle = r * 90
            print("{:5d} : {}".format(angle, y))
    print('##########################################################################################')
    print()

    
# build the test set    
tem_test = TEM_dataset(mode='test')

# retrieve the first image from the test set
x, y = next(iter(tem_test))

# evaluate the model
test_model(model, x)

rotation_transform = MyRotationTransform(angles=[0, 90, 180, 270])

train_transform = Compose([
    #pad,
    #rotation_transform,
    totensor,
])

print('loading training set')
tem_train = TEM_dataset(mode='train', transform=train_transform)
train_loader = torch.utils.data.DataLoader(tem_train, batch_size=BATCH_SIZE, shuffle=True)
print('finished loading training set')


test_transform = Compose([
    #pad,
    totensor,
])
print('loading test set')
tem_test = TEM_dataset(mode='test', transform=test_transform)
test_loader = torch.utils.data.DataLoader(tem_test, batch_size=BATCH_SIZE, shuffle=True)
print('finished loading test set')

#for parameter in model.parameters():
#    print(parameter)

pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(pytorch_total_params)

print('start time:')
start_time = time.time()

xb = []
yb = []
times = np.linspace(1 , nr_epochs, nr_epochs)
predicted_training=[]
target_training=[]
predicted_test=[]
target_test=[]

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
        print(f"epoch {epoch+1} | test accuracy: ")
        print(f" {correct/total*100.}")
        yb.append(correct/total*100)
        if((correct/total*100)>i_best):
            i_best = correct/total*100
            epoch_best=epoch
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

        print(f"epoch {epoch+1} | train accuracy:")
        print(f" {correct/total*100.}")
        xb.append(correct/total*100)


end_time = time.time()
print("Time: ", end_time)

print(f"Highest accuracy of test set: {i_best} at epoch: {epoch_best}")
print(f'model converged at epoch: {round(epoch_best*0.95,1)}')

import seaborn as sn
from sklearn.metrics import confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt

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
    ax.yaxis.set_major_locator(MultipleLocator(20))
    ax.yaxis.set_major_formatter('{x:.0f}')
    ax.yaxis.set_minor_locator(MultipleLocator(2))
    plt.title(name)
    plt.xlabel("Predicted class",fontsize=30, labelpad=2)
    plt.ylabel("True class",fontsize=30)
    plt.savefig(filename)

plot_confusion_matrix(target_training,predicted_training,filename_cf_train_final_epoch,name='Confusion matrix of train data at the final epoch')
plot_confusion_matrix(target_test,predicted_test,filename_cf_test_final_epoch,name='Confusion matrix of the test data at the final epoch')
plot_confusion_matrix(target_training_best,predicted_training_best,filename_cf_train_best,name='Confusion matrix of train data at the highest test accuracy')
plot_confusion_matrix(target_test_best,predicted_test_best,filename_cf_test_best,name='Confusion matrix of the test data at the highest test accuracy')

#Plots accuracy plot for both cases
fig=plt.figure()
plt.plot(list(range(1,len(yb)+1)),yb,label='test',zorder=2,c='C0')
plt.plot(list(range(1,len(xb)+1)),xb,label='train',zorder=1,c='C1')
plt.scatter(epoch_best,i_best,s=10,zorder=3,c='C3')
plt.title('Train & Test accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig(filename_acc_train_test)

#Test equivariance again on trained model
final_tem_test = TEM_dataset(mode='test')
x, y = next(iter(final_tem_test))
test_model(model, x)

print(f'total time for script was: {end_time-start_time}')

time_diff=[]
for ind,val in enumerate(times):
    if ind == 0:
        time_diff.append(val-start_time)
    else:
        time_diff.append(val-times[ind-1])

print(f'average loop time was: {times.mean()}')

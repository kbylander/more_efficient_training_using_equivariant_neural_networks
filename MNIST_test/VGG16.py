#inspiration from https://blog.paperspace.com/vgg-from-scratch-pytorch/

import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms
import matplotlib.pyplot as plt
from torch.utils.data.sampler import SubsetRandomSampler

device = torch.device("cpu")

#load and transform data
def load_dataset(test=False,validation_size=0):
    #transforms images(rezises and crops, transform to tensor and then normalize based on mean and standard deviation)
    trf=transforms.Compose([transforms.Resize(32),transforms.CenterCrop(32),transforms.ToTensor(),transforms.Normalize(mean=0.1285,std=0.2864)])
    if test:
        data_name=datasets.MNIST(root='./data',download=True,train=False,transform = trf)
        return torch.utils.data.DataLoader(data_name, batch_size=len(data_name), shuffle=True)
    
    data_name=datasets.MNIST(root='./data',download=True,train=True,transform = trf)
    
    #create list of ints the length of dataset
    indices = list(range(len(data_name)))

    #Calculate the number of instances in the validation set
    split_ints=int(np.floor(validation_size*len(data_name)))
    
    #random shuffles and samples the indices to train/validation samples and then returns the respective data
    np.random.shuffle(indices)
    train_samples, valid_samples= SubsetRandomSampler(indices[split_ints:]), SubsetRandomSampler(indices[:split_ints])
    return torch.utils.data.DataLoader(data_name, batch_size=60000,sampler=train_samples), torch.utils.data.DataLoader(data_name, batch_size=60000,sampler=valid_samples)


test_data=load_dataset(test=True)
train_data,valid_data=load_dataset(validation_size=0.2)
#calculate mean and std for dataset
def calc_mean_std(dataset):
    images, labels=next(iter(dataset))
    return images.mean([0,2,3]),images.std([0,2,3])

#printed the values, which were inserted into row 20 (transforms.Normalize)
#print(calc_mean_std(data_loader))


def plot_image(dataset):
    images, labels = next(iter(dataset))
    print(len(images))
    images = images.numpy()
    images = images.transpose(0,2,3,1)
    plt.imshow(images[100])
    return plt.show()

#plots a random image from the dataset
#plot_image(test_data)

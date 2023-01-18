#inspiration from https://blog.paperspace.com/vgg-from-scratch-pytorch/

import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms
from torchvision.models import vgg16, VGG16_Weights
import matplotlib.pyplot as plt
from torch.utils.data.sampler import SubsetRandomSampler

device = torch.device("cpu")

#load and transform data for training, validation and test set.  
def load_dataset(test=False,validation_size=0):
    #transforms images(rezises and crops, transform to tensor and then normalize based on mean and standard deviation)
    trf=transforms.Compose([transforms.Grayscale(3),transforms.Resize(32),transforms.CenterCrop(32),transforms.ToTensor(),transforms.Normalize(mean=0.1285,std=0.2864)])
    if test:
        data_name=datasets.MNIST(root='./data',download=True,train=False,transform = trf)
        return torch.utils.data.DataLoader(data_name, batch_size=50, shuffle=True)
    
    data_name=datasets.MNIST(root='./data',download=True,train=True,transform = trf)

    len_data=1000
    #create list of ints the length of dataset
    indices = list(range(len_data))

    #Calculate the number of instances in the validation set
    split_ints=int(np.floor(validation_size*len_data))
    
    #random shuffles and samples the indices to train/validation samples and then returns the respective data
    np.random.shuffle(indices)
    train_samples, valid_samples= SubsetRandomSampler(indices[split_ints:]), SubsetRandomSampler(indices[:split_ints])
    return torch.utils.data.DataLoader(data_name, batch_size=1,sampler=train_samples), torch.utils.data.DataLoader(data_name, batch_size=1,sampler=valid_samples)


test_data=load_dataset(test=True)
train_data,valid_data=load_dataset(validation_size=0.2)
print(len(train_data))
print(len(valid_data))
print(len(test_data))

#calculate mean and std for dataset
def calc_mean_std(dataset):
    images, labels=next(iter(dataset))
    return images.mean([0,2,3]),images.std([0,2,3])

#printed the values, which were inserted into row 20 (transforms.Normalize)
#print(calc_mean_std(data_loader))

#plots a random image from the (test/training/validation data)
def plot_image(dataset):
    images, labels = next(iter(dataset))
    images = images.numpy()
    images = images.transpose(0,2,3,1)
    plt.imshow(images[100])
    return plt.show()

#plot_image(test_data)

no_classes=10
no_epochs = 5
batch_size = 1000
learning_rate = 0.001

#model with number of classes = 10 and hyperparameters:
model = vgg16(10,weights=VGG16_Weights.DEFAULT).to(device)
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)

# Train the model
size_training_set = len(train_data)

for epoch in range(0,no_epochs):
    for i,(images,labels) in enumerate(train_data):
        images=images.to(device)
        images=images.to(device)

        outputs=model(images)
        loss=loss_function(outputs,labels)

        #optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch} of {no_epochs} with loss:{loss.item()}')
    
    with torch.no_grad():
        correct = 0 
        total = 0
        for images, labels in valid_data:
            images=images.to(device)
            labels=labels.to(device)
            outputs = model(images)

            #returns the most likely class predicted
            _, predicted = torch.max(outputs.data,1)

            #calculates accuracy
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            del images, labels, outputs
            print(f'validation accuracy of network {correct/total}')


with torch.no_grad():
    correct = 0 
    total = 0
    for images, labels in test_data:
        images=images.to(device)
        labels=labels.to(device)
        outputs = model(images)

        #returns the most likely class predicted
        _, predicted = torch.max(outputs.data,1)

        #calculates accuracy
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        del images, labels, outputs
        print(f'test accuracy of network {correct/total}')


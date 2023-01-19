"""
Hello world classifier:
Implementing VGG16 on MNIST dataset.

To be run from the main folder
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms,models,datasets
from torchvision.models import vgg16, VGG16_Weights
import matplotlib.pyplot as plt
from torchsummary import summary                     


device = torch.device("cpu")

#transforms images(rezises and crops, transform to tensor)
test_data=datasets.MNIST(root='./MNIST_test/data',download=True,train=False,transform = transforms.Compose([transforms.Grayscale(3),transforms.CenterCrop(32),transforms.Resize(32),transforms.ToTensor()]))
train_data=datasets.MNIST(root='./MNIST_test/data',train=True,transform = transforms.Compose([transforms.Grayscale(3),transforms.CenterCrop(32),transforms.Resize(32),transforms.ToTensor()]))

#split training set into training and validation dataset
len_data=len(train_data)
validation_size=0.2
indices = list(range(len_data))

#Calculate the number of instances in the validation set
split_ints=int(np.floor(validation_size*len_data))

#random shuffles and samples the indices to train/validation samples and then returns the respective data
np.random.shuffle(indices)

train_samples, valid_samples=  torch.utils.data.SubsetRandomSampler(indices[split_ints:]),  torch.utils.data.SubsetRandomSampler(indices[:split_ints])
print(len(train_samples),len(valid_samples))

figure = plt.figure(figsize=(10, 8))
cols, rows = 5, 5
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(train_data), size=(1,)).item()
    img, label = train_data[sample_idx]
    img=img[0:1,:,:]
    figure.add_subplot(rows, cols, i)
    plt.title(label,pad=0)
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")
plt.show()


load_data = {'test' : torch.utils.data.DataLoader(test_data, batch_size=100, shuffle=True),
 'train' : torch.utils.data.DataLoader(train_data,batch_size=100,shuffle=False,sampler=train_samples),
 'validation' : torch.utils.data.DataLoader(train_data,batch_size=100,shuffle=False,sampler=valid_samples)
}

model = vgg16(10,pretrained=True,weights=VGG16_Weights.DEFAULT)
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=0.0001)
losses_list=[]
val_acc=[]
test_acc=[]

def train(num_epochs, model, data,losses_list=losses_list,acc_list=val_acc):
    print("started training")
    no_steps = len(data["train"])
    for epoch in range(num_epochs):
        model.train()
        losses=[]
        for i, (images,labels) in enumerate(data["train"]):
            output= model(images)
            loss = loss_function(output,labels)
            optimizer.zero_grad()
            loss.backward()
            losses.append(loss.item())
            optimizer.step()
            if (i+1) == no_steps:
                losses_list.append(sum(losses)/len(losses))
                print (f'Finished epoch {epoch/num_epochs} with average loss: {losses_list[-1]}')

        test(load_data['validation'],model,text='validation',no_batches=30,acc_list=acc_list)


def test(data,model,acc_list,no_batches=False,text="test"):
    print("started", text)
    #setting drouput and normalization layers to evaluation mode
    model.eval()
    acc=0
    num=0    
    with torch.no_grad():
        for i,(images, labels) in enumerate(data):
            test_output = model(images)
            pred_y = torch.argmax(test_output,dim=1)
            acc += (pred_y == labels).sum().item()
            num += float(labels.size(0))
            if no_batches and i+1>=no_batches: break
    acc_list.append(acc/num)
    print(f'average {text} accuracy: {acc_list[-1]}')

train(3,model,load_data)
test(load_data['test'],model,test_acc)

figure = plt.figure(figsize=(3, 1))
plt.plot(losses_list)
plt.title("Loss")
plt.plot(val_acc)
plt.title('Accuracy during validation')
plt.plot(test_acc)
plt.title('Accuracy during testing')
plt.show()
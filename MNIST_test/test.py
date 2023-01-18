import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms,models,datasets
from torch.autograd import Variable
from torchvision.models import vgg16, VGG16_Weights
import matplotlib.pyplot as plt
from torch.utils.data.sampler import SubsetRandomSampler
from torchsummary import summary                     


device = torch.device("cpu")

#transforms images(rezises and crops, transform to tensor)
test_data=datasets.MNIST(root='./data',download=True,train=False,transform = transforms.Compose([transforms.Grayscale(3),transforms.CenterCrop(32),transforms.Resize(32),transforms.ToTensor()]))
train_data=datasets.MNIST(root='./data',train=True,transform = transforms.Compose([transforms.Grayscale(3),transforms.CenterCrop(32),transforms.Resize(32),transforms.ToTensor()]))

"""
figure = plt.figure(figsize=(10, 8))
cols, rows = 5, 5
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(train_data), size=(1,)).item()
    img, label = train_data[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(label)
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")
plt.show()
"""

load_data = {'test' : torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=True),
 'train' : torch.utils.data.DataLoader(train_data,batch_size=64,shuffle=True)
}

model = vgg16(10,pretrained=True)
model.classifier[-1] = torch.nn.Linear(4096,10)

#summary(model,input_size=(3,32,32))

loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=0.01)

no_epochs = 1

def train(num_epochs, model, data):
    model.train()

    no_steps = len(data["train"])
    for epoch in range(num_epochs):
        for i, (images,labels) in enumerate(data["train"]):
            output= model(images)
            loss = loss_function(output,labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i+1) % 10 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                       .format(epoch + 1, num_epochs, i + 1, no_steps, loss.item()))


def test(data,model):
    #setting drouput and normalization layers to evaluation mode
    model.eval()
    acc=0
    with torch.no_grad():
        correct = 0
        total = 0
        for i,(images, labels) in enumerate(data["test"]):
            test_output = model(images)
            pred_y = torch.argmax(test_output,dim=1)
            print("pred:",pred_y)
            print("lab",labels)
            acc = (pred_y == labels).sum().item() / float(labels.size(0))
            print('test accuracy: %.2f' % acc)
    print('final test accuracy: %.2f' % acc)
        


train(no_epochs,model,load_data)
test(load_data,model)

from torch.utils.data import Dataset
from torchvision.transforms import Grayscale,ToTensor, Compose
from sklearn.metrics import confusion_matrix
import numpy as np
from PIL import Image
from torch import nn
import torch
from torch import cuda

device='cuda' if torch.cuda.is_available() else 'cpu'
# Whether to train on a gpu
train_on_gpu = cuda.is_available()

#if available, set model to cuda,count gpu:s and print
if train_on_gpu:
    #model = model.to('cuda')
    gpu_count = cuda.device_count()
    print(f'{gpu_count} gpus detected.')

number_classes = 10
name_classes=['zero','one','two','three','four','five','six','seven','eight','nine']


class VGG16(torch.nn.Module):
    
    def __init__(self, n_classes=number_classes):
        
        super(VGG16, self).__init__()
        

        #conv1
        in_type=1
        out_type=4
        self.conv1=nn.Sequential(
            nn.Conv2d(in_type,out_type,kernel_size=3,stride=1, padding=1),
            nn.BatchNorm2d(out_type),
            nn.ReLU(inplace=True)
        )

        #conv2
        in_type=4
        self.conv2=nn.Sequential(
            nn.Conv2d(in_type,out_type,kernel_size=3,stride=1, padding=1),
            nn.BatchNorm2d(out_type),
            nn.ReLU(inplace=True)
        )

        #maxpool 1
        self.maxpool1=nn.Sequential(
            nn.MaxPool2d(stride=2,kernel_size=2, padding=0)
        )

        #block 2
        #defining out put type for block
        in_type=out_type
        out_type *= 2
        #conv3
        self.conv3=nn.Sequential(
            nn.Conv2d(in_type,out_type,kernel_size=3,stride=1, padding=1),
            nn.BatchNorm2d(out_type),
            nn.ReLU(inplace=True)
        )
        in_type=out_type

        #conv4
        self.conv4=nn.Sequential(
            nn.Conv2d(in_type,out_type,kernel_size=3,stride=1, padding=1),
            nn.BatchNorm2d(out_type),
            nn.ReLU(inplace=True)
        )

        #maxpool 2
        self.maxpool2=nn.Sequential(
            nn.MaxPool2d(stride=2,kernel_size=2, padding=0)
        )

        #block3
        #defining out put type for block
        in_type=out_type
        out_type *= 2

        #conv 5
        self.conv5=nn.Sequential(
            nn.Conv2d(in_type,out_type,kernel_size=3,stride=1, padding=1),
            nn.BatchNorm2d(out_type),
            nn.ReLU(inplace=True)
        )

        in_type = out_type
        #conv 6
        self.conv6=nn.Sequential(
            nn.Conv2d(in_type,out_type,kernel_size=3,stride=1, padding=1),
            nn.BatchNorm2d(out_type),
            nn.ReLU(inplace=True)
        )

        #conv 7
        self.conv7=nn.Sequential(
            nn.Conv2d(in_type,out_type,kernel_size=3,stride=1, padding=1),
            nn.BatchNorm2d(out_type),
            nn.ReLU(inplace=True)
        )

        #maxpool 3
        self.maxpool3=nn.Sequential(
            nn.MaxPool2d(stride=2,kernel_size=3, padding=1)
        )

        #block 4

        in_type=out_type
        out_type *= 2
        #conv8
        self.conv8=nn.Sequential(
            nn.Conv2d(in_type,out_type,kernel_size=3,stride=1, padding=1),
            nn.BatchNorm2d(out_type),
            nn.ReLU(inplace=True)
        )
        in_type=out_type

        #conv9
        self.conv9=nn.Sequential(
            nn.Conv2d(in_type,out_type,kernel_size=3,stride=1, padding=1),
            nn.BatchNorm2d(out_type),
            nn.ReLU(inplace=True)
        )

        #conv10
        self.conv10=nn.Sequential(
            nn.Conv2d(in_type,out_type,kernel_size=3,stride=1, padding=1),
            nn.BatchNorm2d(out_type),
            nn.ReLU(inplace=True)
        )

        #maxpool 4
        self.maxpool4=nn.Sequential(
            nn.MaxPool2d(stride=2,kernel_size=2, padding=0)
        )

        #block 5
        in_type=out_type
        out_type *= 2

        #conv11
        self.conv11=nn.Sequential(
            nn.Conv2d(in_type,out_type,kernel_size=3,stride=1, padding=1),
            nn.BatchNorm2d(out_type),
            nn.ReLU(inplace=True)
        )
        in_type=out_type

        #conv12
        self.conv12=nn.Sequential(
            nn.Conv2d(in_type,out_type,kernel_size=3,stride=1, padding=1),
            nn.BatchNorm2d(out_type),
            nn.ReLU(inplace=True)
        )

        #conv13
        self.conv13=nn.Sequential(
            nn.Conv2d(in_type,out_type,kernel_size=3,stride=1, padding=1),
            nn.BatchNorm2d(out_type),
            nn.ReLU(inplace=True)
        )

        #maxpool 5
        self.maxpool5=nn.Sequential(
            nn.MaxPool2d(stride=2,kernel_size=2, padding=0)
            )

        #fully connected layer 1
        self.fully_net=nn.Sequential(
            nn.Linear(64,4096),
            nn.ReLU(),
            nn.Linear(4096,4096),
            nn.ReLU(),
            nn.Linear(4096,n_classes),
            nn.Softmax()
        )

    def forward(self, x: torch.Tensor):
        x=self.conv1(x)
        x=self.conv2(x)
        x=self.maxpool1(x)

        x=self.conv3(x)
        x=self.conv4(x)
        x=self.maxpool2(x)

        x=self.conv5(x)
        x=self.conv6(x)
        x=self.conv7(x)

        x=self.maxpool3(x)

        x=self.conv8(x)
        x=self.conv9(x)
        x=self.conv10(x)
        x=self.maxpool4(x)

        x=self.conv11(x)
        x=self.conv12(x)
        x=self.conv13(x)
        x=self.maxpool5(x)

        # classify with the final fully connected layers)
        x = self.fully_net(x.reshape(x.shape[0], -1,))
        return x

class MnistRotDataset(Dataset):
    def __init__(self,mode,transform=None) -> None:
        print("fetching data")
        assert mode in ['train','test']
        if mode == 'train':
            file='./mnist_rotation_new/mnist_all_rotation_normalized_float_train_valid.amat'
        else:
            file='./mnist_rotation_new/mnist_all_rotation_normalized_float_test.amat'

        self.transform=transform
        data = np.loadtxt(file,delimiter=' ')

        self.images=data[:, :-1].reshape(-1,28,28).astype(np.float32)
        self.labels=data[:,-1].astype(np.int64)
        self.num_labels=len(self.labels)

    def __getitem__(self, index):
        image,label=self.images[index],self.labels[index]
        image=Image.fromarray(image)
        if self.transform is not None:
            image=self.transform(image)
        return image,label
    
    def __len__(self):
        return len(self.labels)

totensor=ToTensor()

#build model
model = VGG16().to(device)
predicted=[]
target=[]
def test_model(model: torch.nn.Module, x: Image):
    print("entered testing")
    # evaluate the `model` on 4 rotated versions of the input image `x`
    model.eval()
    
    wrmup = model(torch.randn(1, 1, 28, 28).to(device))
    del wrmup
    
    #x = resize1(pad(x))
    
    print()
    print('##########################################################################################')
    header = 'angle |  ' + '  '.join(["{:5d}".format(d) for d in range(10)])
    print(header)
    with torch.no_grad():
        for r in range(4):
            #x_transformed = totensor(resize2(x.rotate(r*90., Image.BILINEAR))).reshape(1, 1, 29, 29)
            x_transformed = totensor(x.rotate(r*90., Image.Resampling.BILINEAR)).reshape(1, 1, 28, 28)

            x_transformed = x_transformed.to(device)

            y = model(x_transformed)
            y = y.to('cpu').numpy().squeeze()
            
            angle = r * 90
            print("{:4d} : {}".format(angle, y))
    print('##########################################################################################')
    print()

def eval_test(test,accuracy=None,predicted=None,target=None):
    total = 0
    correct = 0
    with torch.no_grad():
        model.eval()
        for i, (x, t) in enumerate(test):

            x = x.to(device)
            t = t.to(device)
            
            y = model(x)

            _, prediction = torch.max(y.data, 1)
            total += t.shape[0]
            correct += (prediction == t).sum().item()
            if type(predicted) is list and type(target) is list:
                [predicted.append(i) for i in prediction]
                [target.append(i) for i in t]
    if type(accuracy)==list:
        accuracy.append(correct/total*100.)
    print(f"epoch {epoch} | test accuracy: {accuracy[-1]}")


# build the test set    
raw_mnist_test = MnistRotDataset(mode='test')

# retrieve the first image from the test set
x, y = next(iter(raw_mnist_test))

# evaluate the model without training
test_model(model, x)

train_transform = Compose([totensor])

mnist_train= MnistRotDataset(mode='train',transform=train_transform)
train_loader = torch.utils.data.DataLoader(mnist_train,batch_size=64)

test_transform= Compose([totensor])
mnist_test = MnistRotDataset(mode='test', transform=test_transform)
test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=64)

loss_function=torch.nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters(),lr=5e-5,weight_decay=1e-5)

print("entered training")
accuracy=[]
for epoch in range(1):
    model.train()
    for i, (x, t) in enumerate(train_loader):
        
        optimizer.zero_grad()

        x = x.to(device)
        t = t.to(device)

        y = model(x)

        loss = loss_function(y, t)

        loss.backward()

        optimizer.step()
    
    if True or epoch % 10 == 0:
        eval_test(train_loader,accuracy)

test_acc=[]
#run test set when training is complete
eval_test(test_loader,test_acc,predicted,target)

# retrieve the first image from the test set
x, y = next(iter(raw_mnist_test))

# evaluate the model
test_model(model, x)
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

cf_matrix=confusion_matrix(target,predicted)
df_cm=pd.DataFrame(cf_matrix/np.sum(cf_matrix,axis=1),index = name_classes, columns = name_classes)
plt.figure(figsize=(15,10))
sn.heatmap(df_cm,annot=True)
plt.savefig('cf_matrix_names.png')

plt.figure()
plt.plot(accuracy)
plt.xlabel("epoch nr")
plt.ylabel("accuracy")
plt.savefig("p4_2.png")
from torch.utils.data import Dataset
from torchvision.transforms import RandomRotation,Pad,Resize,ToTensor, Compose
import numpy as np
from PIL import Image
from e2cnn import gspaces
from e2cnn import nn
import torch

device='cuda' if torch.cuda.is_available() else 'cpu'

class C4SteerableCNN(torch.nn.Module):
    
    def __init__(self, n_classes=10):
        
        super(C4SteerableCNN, self).__init__()
        
        # the model is equivariant under rotations by 90 degrees, modelled by C4
        self.r2_act = gspaces.Rot2dOnR2(N=4)
        
        # the input image is a scalar field, corresponding to the trivial representation
        in_type = nn.FieldType(self.r2_act, [self.r2_act.trivial_repr])
        
        # we store the input type for wrapping the images into a geometric tensor during the forward pass
        self.input_type = in_type



        #block1
        #defining out put type for block

        out_type = nn.FieldType(self.r2_act,16*[self.r2_act.regular_repr])

        #conv1
        self.conv1=nn.SequentialModule(
            nn.R2Conv(in_type,out_type,kernel_size=3,stride=1, padding=1),
            nn.InnerBatchNorm(out_type),
            nn.ReLU(out_type,inplace=True)

        )

        #conv2
        in_type=self.conv1.out_type
        out_type = nn.FieldType(self.r2_act,16*[self.r2_act.regular_repr])
        self.conv2=nn.SequentialModule(
            nn.R2Conv(in_type,out_type,kernel_size=3,stride=1, padding=1),
            nn.InnerBatchNorm(out_type),
            nn.ReLU(out_type,inplace=True)
        )

        #maxpool 1
        self.maxpool1=nn.SequentialModule(
            nn.PointwiseMaxPoolAntialiased(out_type,stride=2,kernel_size=2, padding=0)
        )

        #block 2
        #defining out put type for block
        in_type=self.conv2.out_type
        out_type = nn.FieldType(self.r2_act,32*[self.r2_act.regular_repr])

        #conv3
        self.conv3=nn.SequentialModule(
            nn.R2Conv(in_type,out_type,kernel_size=3,stride=1, padding=1),
            nn.InnerBatchNorm(out_type),
            nn.ReLU(out_type,inplace=True)
        )

        #conv4
        in_type=self.conv3.out_type
        out_type = nn.FieldType(self.r2_act,32*[self.r2_act.regular_repr])
        self.conv4=nn.SequentialModule(
            nn.R2Conv(in_type,out_type,kernel_size=3,stride=1, padding=1),
            nn.InnerBatchNorm(out_type),
            nn.ReLU(out_type,inplace=True)
        )

        #maxpool 2
        self.maxpool2=nn.SequentialModule(
            nn.PointwiseMaxPoolAntialiased(out_type,stride=2,kernel_size=2, padding=0)
        )
        #block3
        #conv 5
        in_type=self.conv4.out_type
        out_type = nn.FieldType(self.r2_act,64*[self.r2_act.regular_repr])
        self.conv5=nn.SequentialModule(
            nn.R2Conv(in_type,out_type,kernel_size=3,stride=1, padding=1),
            nn.InnerBatchNorm(out_type),
            nn.ReLU(out_type,inplace=True)
        )

        #conv 6
        in_type=self.conv5.out_type
        out_type = nn.FieldType(self.r2_act,64*[self.r2_act.regular_repr])
        self.conv6=nn.SequentialModule(
            nn.R2Conv(in_type,out_type,kernel_size=3,stride=1, padding=1),
            nn.InnerBatchNorm(out_type),
            nn.ReLU(out_type,inplace=True)
        )

        #conv 7
        in_type=self.conv6.out_type
        out_type = nn.FieldType(self.r2_act,64*[self.r2_act.regular_repr])
        self.conv7=nn.SequentialModule(
            nn.R2Conv(in_type,out_type,kernel_size=3,stride=1, padding=1),
            nn.InnerBatchNorm(out_type),
            nn.ReLU(out_type,inplace=True)
        )

        #maxpool 3
        self.maxpool3=nn.SequentialModule(
            nn.PointwiseMaxPoolAntialiased(out_type,stride=2,kernel_size=2, padding=0)
        )

        #block 4
        #conv8
        in_type=self.conv7.out_type
        out_type = nn.FieldType(self.r2_act,128*[self.r2_act.regular_repr])
        self.conv8=nn.SequentialModule(
            nn.R2Conv(in_type,out_type,kernel_size=3,stride=1, padding=1),
            nn.InnerBatchNorm(out_type),
            nn.ReLU(out_type,inplace=True)
        )

        #conv9
        in_type=self.conv8.out_type
        out_type = nn.FieldType(self.r2_act,128*[self.r2_act.regular_repr])
        self.conv9=nn.SequentialModule(
            nn.R2Conv(in_type,out_type,kernel_size=3,stride=1, padding=1),
            nn.InnerBatchNorm(out_type),
            nn.ReLU(out_type,inplace=True)
        )

        #conv10
        in_type=self.conv9.out_type
        out_type = nn.FieldType(self.r2_act,128*[self.r2_act.regular_repr])
        self.conv10=nn.SequentialModule(
            nn.R2Conv(in_type,out_type,kernel_size=3,stride=1, padding=1),
            nn.InnerBatchNorm(out_type),
            nn.ReLU(out_type,inplace=True)
        )

        #maxpool 4
        self.maxpool4=nn.SequentialModule(
            nn.PointwiseMaxPoolAntialiased(out_type,stride=2,kernel_size=2, padding=0)
        )

        #block 5
        #conv11
        in_type=self.conv10.out_type
        out_type = nn.FieldType(self.r2_act,256*[self.r2_act.regular_repr])
        self.conv11=nn.SequentialModule(
            nn.R2Conv(in_type,out_type,kernel_size=3,stride=1, padding=1),
            nn.InnerBatchNorm(out_type),
            nn.ReLU(out_type,inplace=True)
        )

        #conv12
        in_type=self.conv11.out_type
        out_type = nn.FieldType(self.r2_act,256*[self.r2_act.regular_repr])
        self.conv12=nn.SequentialModule(
            nn.R2Conv(in_type,out_type,kernel_size=3,stride=1, padding=1),
            nn.InnerBatchNorm(out_type),
            nn.ReLU(out_type,inplace=True)
        )

        #conv13
        in_type=self.conv12.out_type
        out_type = nn.FieldType(self.r2_act,256*[self.r2_act.regular_repr])
        self.conv13=nn.SequentialModule(
            nn.R2Conv(in_type,out_type,kernel_size=3,stride=1, padding=1),
            nn.InnerBatchNorm(out_type),
            nn.ReLU(out_type,inplace=True)
        )

        #maxpool 5
        self.maxpool5=nn.SequentialModule(
            nn.PointwiseMaxPoolAntialiased(out_type,stride=2,kernel_size=2, padding=1)
        )

        #group pooling
        in_type=self.conv13.out_type
        self.gpool=nn.GroupPooling(in_type)
        fc_in=self.gpool.out_type.size

        #fully connected layer 1
        self.fully_net=torch.nn.Sequential(
            torch.nn.Linear(1024,4096),
            torch.nn.ELU(inplace=True),
            #torch.nn.Dropout(p=0.5),
            torch.nn.Linear(4096,4096),
            torch.nn.ELU(inplace=True),
            #torch.nn.Dropout(p=0.5),
            torch.nn.Linear(4096,n_classes),
            #torch.nn.Softmax()
        )

    def forward(self, input: torch.Tensor):
        # wrap the input tensor in a GeometricTensor
        # (associate it with the input type)
        x = nn.GeometricTensor(input, self.input_type)
        
        # apply each equivariant block
        
        # Each layer has an input and an output type
        # A layer takes a GeometricTensor in input.
        # This tensor needs to be associated with the same representation of the layer's input type
        #
        # The Layer outputs a new GeometricTensor, associated with the layer's output type.
        # As a result, consecutive layers need to have matching input/output types

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
        
        x=self.gpool(x)

        x = x.tensor

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

#pad images to size 29
# to allow odd size filters, with stride 2 on a feature map. 
pad=Pad((0,0,1,1),fill=0)

#reduce artifacts (because of rotations) (im a little confused by this)

#resize1 = Resize(87)
#resize2 = Resize(29)

totensor=ToTensor()

#build model
model = C4SteerableCNN().to(device)

def test_model(model: torch.nn.Module, x: Image):
    print("entered testing")
    # evaluate the `model` on 4 rotated versions of the input image `x`
    model.eval()
    
    wrmup = model(torch.randn(1, 1, 29, 29).to(device))
    del wrmup
    
    #x = resize1(pad(x))
    x=pad(x)
    print()
    print('##########################################################################################')
    header = 'angle |  ' + '  '.join(["{:6d}".format(d) for d in range(10)])
    print(header)
    with torch.no_grad():
        for r in range(4):
            #x_transformed = totensor(resize2(x.rotate(r*90., Image.Resampling.BILINEAR))).reshape(1, 1, 29, 29)
            x_transformed = totensor(x.rotate(r*90., Image.Resampling.BILINEAR)).reshape(1, 1, 29, 29)

            x_transformed = x_transformed.to(device)

            y = model(x_transformed)
            y = y.to('cpu').numpy().squeeze()
            
            angle = r * 90
            print("{:5d} : {}".format(angle, y))
    print('##########################################################################################')
    print()

# build the test set    
raw_mnist_test = MnistRotDataset(mode='test')

# retrieve the first image from the test set
x, y = next(iter(raw_mnist_test))

# evaluate the model without training
test_model(model, x)

train_transform = Compose([pad,
#resize1,
RandomRotation(180,expand=False),
#resize2,
totensor])

mnist_train= MnistRotDataset(mode='train',transform=train_transform)
train_loader = torch.utils.data.DataLoader(mnist_train,batch_size=64)

test_transform= Compose([pad,totensor])
mnist_test = MnistRotDataset(mode='test', transform=test_transform)
test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=64)

loss_function=torch.nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters(),lr=1e-5, weight_decay=256.0)

print("entered training")
accuracy=[]

for epoch in range(5):
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
        accuracy.append(correct/total*100.)
        print(f"epoch {epoch} | test accuracy: {accuracy[-1]}")

# retrieve the first image from the test set
x, y = next(iter(raw_mnist_test))

# evaluate the model
test_model(model, x)

import matplotlib.pyplot as plt

plt.plot(accuracy)
plt.xlabel("epoch nr")
plt.ylabel("accuracy")
plt.savefig("p4.png")
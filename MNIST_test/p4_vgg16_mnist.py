from torch.utils.data import Dataset
from torchvision.transforms import RandomRotation,Pad,Resize,ToTensor, Compose
import numpy as np
from PIL import Image
from e2cnn import gspaces
from e2cnn import nn
import torch

device='cuda' if torch.cuda.is_available() else 'cpu'

#set group, we set 90 degree rotations here on the R^2 plane. 
r2_act = gspaces.Rot2dOnR2(N=4)

class C4SteerableCNN(torch.nn.Module):
    
    def __init__(self, n_classes=10):
        
        super(C4SteerableCNN, self).__init__()
        
        # the model is equivariant under rotations by 90 degrees, modelled by C4
        self.r2_act = gspaces.Rot2dOnR2(N=4)
        
        # the input image is a scalar field, corresponding to the trivial representation
        in_type = nn.FieldType(self.r2_act, [self.r2_act.trivial_repr])
        
        # we store the input type for wrapping the images into a geometric tensor during the forward pass
        self.input_type = in_type
        
        # convolution 1
        # first specify the output type of the convolutional layer
        # we choose 24 feature fields, each transforming under the regular representation of C4
        out_type = nn.FieldType(self.r2_act, 24*[self.r2_act.regular_repr])
        self.block1 = nn.SequentialModule(
            nn.MaskModule(in_type, 29, margin=1),
            nn.R2Conv(in_type, out_type, kernel_size=7, padding=1, bias=False),
            nn.InnerBatchNorm(out_type),
            nn.ReLU(out_type, inplace=True)
        )
        
        # convolution 2
        # the old output type is the input type to the next layer
        in_type = self.block1.out_type
        # the output type of the second convolution layer are 48 regular feature fields of C8
        out_type = nn.FieldType(self.r2_act, 48*[self.r2_act.regular_repr])
        self.block2 = nn.SequentialModule(
            nn.R2Conv(in_type, out_type, kernel_size=5, padding=2, bias=False),
            nn.InnerBatchNorm(out_type),
            nn.ReLU(out_type, inplace=True)
        )
        self.pool1 = nn.SequentialModule(
            nn.PointwiseAvgPoolAntialiased(out_type, sigma=0.66, stride=2)
        )
        
        # convolution 3
        # the old output type is the input type to the next layer
        in_type = self.block2.out_type
        # the output type of the third convolution layer are 48 regular feature fields of C8
        out_type = nn.FieldType(self.r2_act, 48*[self.r2_act.regular_repr])
        self.block3 = nn.SequentialModule(
            nn.R2Conv(in_type, out_type, kernel_size=5, padding=2, bias=False),
            nn.InnerBatchNorm(out_type),
            nn.ReLU(out_type, inplace=True)
        )
        
        # convolution 4
        # the old output type is the input type to the next layer
        in_type = self.block3.out_type
        # the output type of the fourth convolution layer are 96 regular feature fields of C8
        out_type = nn.FieldType(self.r2_act, 96*[self.r2_act.regular_repr])
        self.block4 = nn.SequentialModule(
            nn.R2Conv(in_type, out_type, kernel_size=5, padding=2, bias=False),
            nn.InnerBatchNorm(out_type),
            nn.ReLU(out_type, inplace=True)
        )
        self.pool2 = nn.SequentialModule(
            nn.PointwiseAvgPoolAntialiased(out_type, sigma=0.66, stride=2)
        )
        
        # convolution 5
        # the old output type is the input type to the next layer
        in_type = self.block4.out_type
        # the output type of the fifth convolution layer are 96 regular feature fields of C8
        out_type = nn.FieldType(self.r2_act, 96*[self.r2_act.regular_repr])
        self.block5 = nn.SequentialModule(
            nn.R2Conv(in_type, out_type, kernel_size=5, padding=2, bias=False),
            nn.InnerBatchNorm(out_type),
            nn.ReLU(out_type, inplace=True)
        )
        
        # convolution 6
        # the old output type is the input type to the next layer
        in_type = self.block5.out_type
        # the output type of the sixth convolution layer are 64 regular feature fields of C8
        out_type = nn.FieldType(self.r2_act, 64*[self.r2_act.regular_repr])
        self.block6 = nn.SequentialModule(
            nn.R2Conv(in_type, out_type, kernel_size=5, padding=1, bias=False),
            nn.InnerBatchNorm(out_type),
            nn.ReLU(out_type, inplace=True)
        )
        self.pool3 = nn.PointwiseAvgPoolAntialiased(out_type, sigma=0.66, stride=1, padding=0)
        
        self.gpool = nn.GroupPooling(out_type)
        
        # number of output channels
        c = self.gpool.out_type.size
        
        # Fully Connected
        self.fully_net = torch.nn.Sequential(
            torch.nn.Linear(c, 64),
            torch.nn.BatchNorm1d(64),
            torch.nn.ELU(inplace=True),
            torch.nn.Linear(64, n_classes),
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
        x = self.block1(x)
        x = self.block2(x)
        x = self.pool1(x)
        
        x = self.block3(x)
        x = self.block4(x)
        x = self.pool2(x)
        
        x = self.block5(x)
        x = self.block6(x)
        
        # pool over the spatial dimensions
        x = self.pool3(x)
        
        # pool over the group
        x = self.gpool(x)

        # unwrap the output GeometricTensor
        # (take the Pytorch tensor and discard the associated representation)
        x = x.tensor
        
        # classify with the final fully connected layers)
        x = self.fully_net(x.reshape(x.shape[0], -1))
        
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

resize1 = Resize(87)
resize2 = Resize(29)

totensor=ToTensor()

#build model
model = C4SteerableCNN().to(device)

def test_model(model: torch.nn.Module, x: Image):
    print("entered testing")
    # evaluate the `model` on 4 rotated versions of the input image `x`
    model.eval()
    
    wrmup = model(torch.randn(1, 1, 29, 29).to(device))
    del wrmup
    
    x = resize1(pad(x))
    
    print()
    print('##########################################################################################')
    header = 'angle |  ' + '  '.join(["{:6d}".format(d) for d in range(10)])
    print(header)
    with torch.no_grad():
        for r in range(4):
            x_transformed = totensor(resize2(x.rotate(r*90., Image.BILINEAR))).reshape(1, 1, 29, 29)
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
resize1,
RandomRotation(180,resample=Image.Resampling.BILINEAR,expand=False),
resize2,
totensor])

mnist_train= MnistRotDataset(mode='train',transform=train_transform)
train_loader = torch.utils.data.DataLoader(mnist_train,batch_size=64)

test_transform= Compose([pad,totensor])
mnist_test = MnistRotDataset(mode='test', transform=test_transform)
test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=64)

loss_function=torch.nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters(),lr=5e-5,weight_decay=1e-5)

print("entered training")

for epoch in range(2):
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
        print(f"epoch {epoch} | test accuracy: {correct/total*100.}")

# retrieve the first image from the test set
x, y = next(iter(raw_mnist_test))

# evaluate the model
test_model(model, x)
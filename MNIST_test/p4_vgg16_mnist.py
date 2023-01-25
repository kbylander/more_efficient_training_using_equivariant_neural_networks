from torch.utils.data import Dataset
from torchvision.transforms import RandomRotation,Pad,Resize,ToTensor, Compose
import numpy as np
from PIL import Image
from e2cnn import gspaces
from e2cnn import nn
import torch

devide='cuda' if torch.cuda.is_available() else 'cpu'

#set group, we set 90 degree rotations here on the R^2 plane. 
r2_act = gspaces.Rot2dOnR2(N=4)

class c4Steerable(torch.nn.Module):
    def __init__(self,n_classes=10) -> None:
        super(c4Steerable,self).__init__()

        #define rotation equivariance
        self.r2_act=gspaces.Rot2dOnR2(N=4)

        #define input type
        in_type = nn.FieldType(self.r2_act,[self.r2_act.trivial_repr])
        self.input_type = nn.FieldType(self.r2_act,in_type)

        #convolution 1, performing convolution with kernel size 7, and batchnormalization. 
        out_type=nn.FieldType(self.r2_act,24*[self.r2_act.regular_repr])
        self.block1=nn.SequentialModule(
            nn.MaskModule(in_type,29,margin=1),
            nn.R2Conv(in_type,out_type, kernel_size=7,padding=1,bias=False),
            nn.InnerBatchNorm(out_type),
            nn.ReLU(out_type,inplace=True)
        )

        #convolution 2, performing conv with kernel size 5 and padding set to 2, no bias. Batch norm and ReLU. 
        #Output size should be 48.
        
        #input type and output type
        in_type=self.block1.out_type
        out_type=nn.FieldType(self.r2_act,48*[self.r2_act.regular_repr])
        self.block2=nn.SequentialModule(
            nn.MaskModule(in_type,29,margin=1),
            nn.R2Conv(in_type,out_type,kernel_size=5,padding=2),
            nn.InnerBatchNorm(out_type),
            nn.ReLU(out_type,inplace=True)
        )
        #performs gaussian blurring on each filter independently, with sigma set to 0.66
        self.pool1 = nn.SequentialModule(
            nn.PointwiseAvgPoolAntialiased(out_type,sigma=0.66,stride=2)
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
        self.pool3 = nn.PointwiseAvgPoolAntialiased(out_type,sigma=0.66,stride=1,padding=0)
        self.gpool = nn.GroupPooling(out_type)
        c = self.gpool.out_type.size

        #Fully connected layer
        self.fully_net= torch.nn.Sequential(
            torch.nn.Linear(c,64),
            torch.nn.BatchNorm1d(64),
            torch.nn.ELU(inplace=True),
            torch.nn.Linear(64,n_classes)
        )
    def forward(self, input: torch.Tensor):
        x = nn.GeometricTensor(input,self.input_type)
        
        #apply each equivariant block
        x=self.block1(x)
        x=self.block2(x)
        x=self.pool1(x)

        x=self.block3(x)
        x=self.block4(x)
        x=self.pool2(x)

        x=self.block5(x)
        x=self.block6(x)
        x=self.pool3(x)

        x=self.gpool(x)
        x=x.Tensor
        print("see change: ",x,x.reshape(x.shape[0],-1))
        x=self.fully_net(x.reshape(x.shape[0],-1))
        return x

class MnistRotDataset(Dataset):
    def __init__(self,mode,transform=None) -> None:
        assert mode in ['train','test']
        if mode == 'train':
            file='mnist_rotation_new/mnist_all_rotation_normalizer_float_train_valid.amat'
        else:
            file='mnist_rotation_new/mnist_all_rotation_normalized_float_test.amat'

        self.transform=transform
        data = np.loadtxt(file,delimiter=' ')

        self.images=data[:, :-1].reshape(-1,28,28).astype(np.float32)
        self.labels=data[:,-1].astype(np.int64)
        self.num_labels=len(self.labels)
from e2cnn import gspaces
from e2cnn import nn as e2cnn
import torch
from torch import nn

number_classes=10

class C4_VGG16(torch.nn.Module):
    
    def __init__(self, n_classes=number_classes):

        
        super(C4_VGG16, self).__init__()
        
        # the model is equivariant under rotations by 90 degrees, modelled by C4
        self.r2_act = gspaces.Rot2dOnR2(N=4)
        
        # the input image is a scalar field, corresponding to the trivial representation
        in_type = e2cnn.FieldType(self.r2_act, [self.r2_act.trivial_repr])
        
        # we store the input type for wrapping the images into a geometric tensor during the forward pass
        self.input_type = in_type

        #conv relu dropout p=0.5

        #block1
        #defining out put type for block
        out_type = e2cnn.FieldType(self.r2_act,4*[self.r2_act.regular_repr])

        #conv1
        self.conv1=e2cnn.SequentialModule(
            e2cnn.R2Conv(in_type,out_type,kernel_size=3,stride=1, padding=1),
            e2cnn.InnerBatchNorm(out_type),
            e2cnn.ReLU(out_type,inplace=True),
        )

        #conv2
        in_type=self.conv1.out_type
        self.conv2=e2cnn.SequentialModule(
            e2cnn.R2Conv(in_type,out_type,kernel_size=3,stride=1, padding=1),
            e2cnn.InnerBatchNorm(out_type),
            e2cnn.ReLU(out_type,inplace=True),
        )

        #maxpool 1
        self.maxpool1=e2cnn.SequentialModule(
            e2cnn.PointwiseMaxPool(out_type,stride=2,kernel_size=2, padding=0)
        )

        #block 2
        #defining out put type for block
        in_type=self.conv2.out_type
        out_type = e2cnn.FieldType(self.r2_act,8*[self.r2_act.regular_repr])

        #conv3
        self.conv3=e2cnn.SequentialModule(
            e2cnn.R2Conv(in_type,out_type,kernel_size=3,stride=1, padding=1),
            e2cnn.InnerBatchNorm(out_type),
            e2cnn.ReLU(out_type,inplace=True),
        )

        #conv4
        in_type=self.conv3.out_type
        self.conv4=e2cnn.SequentialModule(
            e2cnn.R2Conv(in_type,out_type,kernel_size=3,stride=1, padding=1),
            e2cnn.InnerBatchNorm(out_type),
            e2cnn.ReLU(out_type,inplace=True),
        )

        #maxpool 2
        self.maxpool2=e2cnn.SequentialModule(
            e2cnn.PointwiseMaxPool(out_type,stride=2,kernel_size=2, padding=0)
        )

        #block3
        #defining out put type for block
        in_type=self.conv4.out_type
        out_type = e2cnn.FieldType(self.r2_act,16*[self.r2_act.regular_repr])


        #conv 5
        self.conv5=e2cnn.SequentialModule(
            e2cnn.R2Conv(in_type,out_type,kernel_size=3,stride=1, padding=1),
            e2cnn.InnerBatchNorm(out_type),
            e2cnn.ReLU(out_type,inplace=True),
        )

        #conv 6
        in_type=self.conv5.out_type
        self.conv6=e2cnn.SequentialModule(
            e2cnn.R2Conv(in_type,out_type,kernel_size=3,stride=1, padding=1),
            e2cnn.InnerBatchNorm(out_type),
            e2cnn.ReLU(out_type,inplace=True),
        )

        #conv 7
        in_type=self.conv6.out_type
        self.conv7=e2cnn.SequentialModule(
            e2cnn.R2Conv(in_type,out_type,kernel_size=3,stride=1, padding=1),
            e2cnn.InnerBatchNorm(out_type),
            e2cnn.ReLU(out_type,inplace=True),
        )

        #maxpool 3
        self.maxpool3=e2cnn.SequentialModule(
            e2cnn.PointwiseMaxPool(out_type,stride=2,kernel_size=3, padding=1)
        )

        #block 4

        #conv8
        in_type=self.conv7.out_type
        out_type = e2cnn.FieldType(self.r2_act,32*[self.r2_act.regular_repr])
        self.conv8=e2cnn.SequentialModule(
            e2cnn.R2Conv(in_type,out_type,kernel_size=3,stride=1, padding=1),
            e2cnn.InnerBatchNorm(out_type),
            e2cnn.ReLU(out_type,inplace=True),
        )

        #conv9
        in_type=self.conv8.out_type
        self.conv9=e2cnn.SequentialModule(
            e2cnn.R2Conv(in_type,out_type,kernel_size=3,stride=1, padding=1),
            e2cnn.InnerBatchNorm(out_type),
            e2cnn.ReLU(out_type,inplace=True),
        )

        #conv10
        in_type=self.conv9.out_type
        self.conv10=e2cnn.SequentialModule(
            e2cnn.R2Conv(in_type,out_type,kernel_size=3,stride=1, padding=1),
            e2cnn.InnerBatchNorm(out_type),
            e2cnn.ReLU(out_type,inplace=True),
        )

        #maxpool 4
        self.maxpool4=e2cnn.SequentialModule(
            e2cnn.PointwiseMaxPool(out_type,stride=2,kernel_size=2, padding=0)
        )

        #block 5
        #conv11
        in_type=self.conv10.out_type
        out_type = e2cnn.FieldType(self.r2_act,32*[self.r2_act.regular_repr])
        self.conv11=e2cnn.SequentialModule(
            e2cnn.R2Conv(in_type,out_type,kernel_size=3,stride=1, padding=1),
            e2cnn.InnerBatchNorm(out_type),
            e2cnn.ReLU(out_type,inplace=True),
        )

        #conv12
        in_type=self.conv11.out_type
        self.conv12=e2cnn.SequentialModule(
            e2cnn.R2Conv(in_type,out_type,kernel_size=3,stride=1, padding=1),
            e2cnn.InnerBatchNorm(out_type),
            e2cnn.ReLU(out_type,inplace=True),
        )

        #conv13
        in_type=self.conv12.out_type
        self.conv13=e2cnn.SequentialModule(
            e2cnn.R2Conv(in_type,out_type,kernel_size=3,stride=1, padding=1),
            e2cnn.InnerBatchNorm(out_type),
            e2cnn.ReLU(out_type,inplace=True),
        )

        #maxpool 5
        self.maxpool5=e2cnn.SequentialModule(
            e2cnn.PointwiseMaxPool(out_type,stride=2,kernel_size=2, padding=0)
            )

        #group pooling
        #out_type=self.conv13.out_type
        self.gpool=e2cnn.GroupPooling(out_type)
        c=self.gpool.out_type.size

        #fully connected layer 1
        self.fully_net=torch.nn.Sequential(
            nn.Linear(c,4096),
            nn.ReLU(4096),
            nn.Linear(4096,4096),
            nn.ReLU(4096),
            nn.Linear(4096,n_classes),
            nn.Softmax()
        )

    def forward(self, input: torch.Tensor):
        # wrap the input tensor in a GeometricTensor
        # (associate it with the input type)
        x = e2cnn.GeometricTensor(input, self.input_type)

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

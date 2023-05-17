import numpy as np
from PIL import Image
from e2cnn import gspaces
from e2cnn import nn as e2nn
import e2cnn
import torch
from torch import nn
from torch import cuda

class G_CUSTOM(nn.Module):

    def __init__(self, n_classes,angular_resolution,mult=32):
        super(G_CUSTOM, self).__init__()
        
        # the model is equivariant under rotations by 90 degrees, modelled by C4
        #self.r2_act = gspaces.Rot2dOnR2(N=angular_resolution)
        self.r2_act = gspaces.FlipRot2dOnR2(N=angular_resolution)

        # the input image is a scalar field, corresponding to the trivial representation
        in_type = e2nn.FieldType(self.r2_act,[self.r2_act.trivial_repr])

        # we store the input type for wrapping the images into a geometric tensor during the forward pass
        self.input_type = in_type

        #block1
        #defining out put type for block
        out_type = e2nn.FieldType(self.r2_act,1*[self.r2_act.regular_repr])

        #for not breaking equivariance when concatenating the input before the first maxpooling layer. 
        self.pre_conv=e2nn.SequentialModule(
            e2nn.R2Conv(in_type,out_type,kernel_size=1,stride=1,padding=0,bias=False),
        )

        #conv1
        out_type = e2nn.FieldType(self.r2_act,mult*[self.r2_act.regular_repr])
        in_type=self.pre_conv.out_type
        
        self.conv1=e2nn.SequentialModule(
            e2nn.R2Conv(in_type,out_type,kernel_size=3,stride=1, padding=1),
            e2nn.InnerBatchNorm(out_type),
            e2nn.ReLU(out_type,inplace=True),
        )

        #conv2
        in_type=self.conv1.out_type
        out_type=e2nn.FieldType(self.r2_act,(mult-1)*[self.r2_act.regular_repr])
        self.conv2=e2nn.SequentialModule(
            e2nn.R2Conv(in_type,out_type,kernel_size=3,stride=1, padding=1),
            e2nn.InnerBatchNorm(out_type),
            e2nn.ReLU(out_type,inplace=True),
        )
        in_type=out_type

        #maxpool 1
        in_type=e2nn.FieldType(self.r2_act,mult*[self.r2_act.regular_repr])
        self.maxpool1=e2nn.SequentialModule(
            e2nn.PointwiseMaxPool(in_type,stride=2,kernel_size=2, padding=0),
            e2nn.PointwiseDropout(in_type,p=0.2)
        )

        #conv3
        #input output type for block 2
        out_type=e2nn.FieldType(self.r2_act,mult*[self.r2_act.regular_repr])
        self.conv3=e2nn.SequentialModule(
            e2nn.R2Conv(in_type,out_type,kernel_size=3,stride=1, padding=1),
            e2nn.InnerBatchNorm(out_type),
            e2nn.ReLU(out_type,inplace=True),
        )

        #conv4
        in_type = out_type
        self.conv4=e2nn.SequentialModule(
            e2nn.R2Conv(in_type,out_type,kernel_size=3,stride=1, padding=1),
            e2nn.InnerBatchNorm(out_type),
            e2nn.ReLU(out_type,inplace=True),
        )

        #maxpool 2
        #concatenating input from block 1 and 2
        in_type=e2nn.FieldType(self.r2_act,2*mult*[self.r2_act.regular_repr])
        self.maxpool2=e2nn.SequentialModule(
            e2nn.PointwiseMaxPool(in_type,stride=2,kernel_size=2, padding=0),
            e2nn.PointwiseDropout(in_type,p=0.25)
        )

        #block 3
        out_type=in_type
        self.conv5=e2nn.SequentialModule(
            e2nn.R2Conv(in_type,out_type,kernel_size=3,stride=1,padding=1),
            e2nn.InnerBatchNorm(out_type),
            e2nn.ReLU(out_type,inplace=True)
        )

        self.conv6=e2nn.SequentialModule(
            e2nn.R2Conv(in_type,out_type,kernel_size=3,stride=1,padding=1),
            e2nn.InnerBatchNorm(out_type),
            e2nn.ReLU(out_type,inplace=True)
        )

        in_type=e2nn.FieldType(self.r2_act,4*mult*[self.r2_act.regular_repr])
        self.maxpool3=e2nn.SequentialModule(
            e2nn.PointwiseMaxPool(in_type,stride=4,kernel_size=4, padding=0),
            e2nn.PointwiseDropout(in_type,p=0.30)
        )
        
        #block 4
        out_type = e2nn.FieldType(self.r2_act,2*mult*[self.r2_act.regular_repr])
        self.conv7=e2nn.SequentialModule(
            e2nn.R2Conv(in_type,out_type,kernel_size=3,stride=1,padding=1),
            e2nn.InnerBatchNorm(out_type),
            e2nn.ReLU(out_type,inplace=True)
        )
        
        in_type=out_type
        self.conv8=e2nn.SequentialModule(
            e2nn.R2Conv(in_type,out_type,kernel_size=3,stride=1,padding=1),
            e2nn.InnerBatchNorm(out_type),
            e2nn.ReLU(out_type,inplace=True)
        )
        
        in_type=e2nn.FieldType(self.r2_act,6*mult*[self.r2_act.regular_repr])
        self.maxpool4=e2nn.SequentialModule(
            e2nn.PointwiseMaxPool(in_type, kernel_size=4, stride=4, padding=0),
            e2nn.PointwiseDropout(in_type,p=0.35)
        )

        out_type=e2nn.FieldType(self.r2_act,4*mult*[self.r2_act.regular_repr])
        self.final_conv=e2nn.SequentialModule(
            e2nn.R2Conv(in_type,out_type,kernel_size=4,stride=4,padding=0),
            e2nn.ReLU(out_type,inplace=True)
            )
        
        #group pooling
        in_type = out_type
        self.gpool=e2nn.GroupPooling(in_type)
        c=self.gpool.out_type.size

        #fully connected layer
        self.fully_net=nn.Sequential(
            nn.Linear(c,256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            #this dropout was inplace=True for my experiments. 
            nn.Dropout(p=0.35,inplace=False),
            nn.Linear(256,256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256,n_classes),
            nn.Softmax()
        )

    def forward(self, input: torch.Tensor):
        # wrap the input tensor in a GeometricTensor
        # (associate it with the input type)

        x = e2nn.GeometricTensor(input, self.input_type)
        x=self.pre_conv(x)
        
        #saving 4 channel input in the regular representation
        x_ini=x
        x=self.conv1(x)
        x=self.conv2(x)
        x=e2nn.tensor_directsum([x,x_ini])
        x=self.maxpool1(x)
        x_block1 = x

        x=self.conv3(x)
        x=self.conv4(x)
        x=e2nn.tensor_directsum([x,x_block1])     
        x=self.maxpool2(x)
        x_block2 = x

        x=self.conv5(x)
        x=self.conv6(x)
        x=e2nn.tensor_directsum([x,x_block2])
        x=self.maxpool3(x)

        x_block3 = x
        x=self.conv7(x)
        x=self.conv8(x)
        x=e2nn.tensor_directsum([x,x_block3])


        x=self.maxpool4(x)
        x=self.final_conv(x)
        x=self.gpool(x)
        x = x.tensor

        # classify with the final fully connected layers)
        x = self.fully_net(x.reshape(x.shape[0], -1,))
        return x

class G_CUSTOM_no_batchNorm(nn.Module):

    def __init__(self, n_classes,angular_resolution,mult=32):
        super(G_CUSTOM_no_batchNorm, self).__init__()
        
        # the model is equivariant under rotations by 90 degrees, modelled by C4
        #self.r2_act = gspaces.Rot2dOnR2(N=angular_resolution)
        self.r2_act = gspaces.FlipRot2dOnR2(N=angular_resolution)

        # the input image is a scalar field, corresponding to the trivial representation
        in_type = e2nn.FieldType(self.r2_act,[self.r2_act.trivial_repr])

        # we store the input type for wrapping the images into a geometric tensor during the forward pass
        self.input_type = in_type

        #block1
        #defining out put type for block
        out_type = e2nn.FieldType(self.r2_act,1*[self.r2_act.regular_repr])

        #for not breaking equivariance when concatenating the input before the first maxpooling layer. 
        self.pre_conv=e2nn.SequentialModule(
            e2nn.R2Conv(in_type,out_type,kernel_size=1,stride=1,padding=0,bias=False),
        )

        #conv1
        out_type = e2nn.FieldType(self.r2_act,mult*[self.r2_act.regular_repr])
        in_type=self.pre_conv.out_type
        
        self.conv1=e2nn.SequentialModule(
            e2nn.R2Conv(in_type,out_type,kernel_size=3,stride=1, padding=1),
            #e2nn.InnerBatchNorm(out_type),
            e2nn.ReLU(out_type,inplace=True),
        )

        #conv2
        in_type=self.conv1.out_type
        out_type=e2nn.FieldType(self.r2_act,(mult-1)*[self.r2_act.regular_repr])
        self.conv2=e2nn.SequentialModule(
            e2nn.R2Conv(in_type,out_type,kernel_size=3,stride=1, padding=1),
            #e2nn.InnerBatchNorm(out_type),
            e2nn.ReLU(out_type,inplace=True),
        )
        in_type=out_type

        #maxpool 1
        in_type=e2nn.FieldType(self.r2_act,mult*[self.r2_act.regular_repr])
        self.maxpool1=e2nn.SequentialModule(
            e2nn.PointwiseMaxPool(in_type,stride=2,kernel_size=2, padding=0),
            e2nn.PointwiseDropout(in_type,p=0.2)
        )

        #conv3
        #input output type for block 2
        out_type=e2nn.FieldType(self.r2_act,mult*[self.r2_act.regular_repr])
        self.conv3=e2nn.SequentialModule(
            e2nn.R2Conv(in_type,out_type,kernel_size=3,stride=1, padding=1),
            #e2nn.InnerBatchNorm(out_type),
            e2nn.ReLU(out_type,inplace=True),
        )

        #conv4
        in_type = out_type
        self.conv4=e2nn.SequentialModule(
            e2nn.R2Conv(in_type,out_type,kernel_size=3,stride=1, padding=1),
            #e2nn.InnerBatchNorm(out_type),
            e2nn.ReLU(out_type,inplace=True),
        )

        #maxpool 2
        #concatenating input from block 1 and 2
        in_type=e2nn.FieldType(self.r2_act,2*mult*[self.r2_act.regular_repr])
        self.maxpool2=e2nn.SequentialModule(
            e2nn.PointwiseMaxPool(in_type,stride=2,kernel_size=2, padding=0),
            e2nn.PointwiseDropout(in_type,p=0.25)
        )

        #block 3
        out_type=in_type
        self.conv5=e2nn.SequentialModule(
            e2nn.R2Conv(in_type,out_type,kernel_size=3,stride=1,padding=1),
            #e2nn.InnerBatchNorm(out_type),
            e2nn.ReLU(out_type,inplace=True)
        )

        self.conv6=e2nn.SequentialModule(
            e2nn.R2Conv(in_type,out_type,kernel_size=3,stride=1,padding=1),
            #e2nn.InnerBatchNorm(out_type),
            e2nn.ReLU(out_type,inplace=True)
        )

        in_type=e2nn.FieldType(self.r2_act,4*mult*[self.r2_act.regular_repr])
        self.maxpool3=e2nn.SequentialModule(
            e2nn.PointwiseMaxPool(in_type,stride=4,kernel_size=4, padding=0),
            e2nn.PointwiseDropout(in_type,p=0.30)
        )
        
        #block 4
        out_type = e2nn.FieldType(self.r2_act,2*mult*[self.r2_act.regular_repr])
        self.conv7=e2nn.SequentialModule(
            e2nn.R2Conv(in_type,out_type,kernel_size=3,stride=1,padding=1),
            #e2nn.InnerBatchNorm(out_type),
            e2nn.ReLU(out_type,inplace=True)
        )
        
        in_type=out_type
        self.conv8=e2nn.SequentialModule(
            e2nn.R2Conv(in_type,out_type,kernel_size=3,stride=1,padding=1),
            #e2nn.InnerBatchNorm(out_type),
            e2nn.ReLU(out_type,inplace=True)
        )
        
        in_type=e2nn.FieldType(self.r2_act,6*mult*[self.r2_act.regular_repr])
        self.maxpool4=e2nn.SequentialModule(
            e2nn.PointwiseMaxPool(in_type, kernel_size=4, stride=4, padding=0),
            e2nn.PointwiseDropout(in_type,p=0.35)
        )

        out_type=e2nn.FieldType(self.r2_act,4*mult*[self.r2_act.regular_repr])
        self.final_conv=e2nn.SequentialModule(
            e2nn.R2Conv(in_type,out_type,kernel_size=4,stride=4,padding=0),
            e2nn.ReLU(out_type,inplace=True)
            )
        
        #group pooling
        in_type = out_type
        self.gpool=e2nn.GroupPooling(in_type)
        c=self.gpool.out_type.size

        #fully connected layer
        self.fully_net=nn.Sequential(
            nn.Linear(c,256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            #this dropout was inplace=True for my experiments. 
            nn.Dropout(p=0.35,inplace=False),
            nn.Linear(256,256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256,n_classes),
            nn.Softmax()
        )

    def forward(self, input: torch.Tensor):
        # wrap the input tensor in a GeometricTensor
        # (associate it with the input type)

        x = e2nn.GeometricTensor(input, self.input_type)
        x=self.pre_conv(x)
        
        #saving 4 channel input in the regular representation
        x_ini=x
        x=self.conv1(x)
        x=self.conv2(x)
        x=e2nn.tensor_directsum([x,x_ini])
        x=self.maxpool1(x)
        x_block1 = x

        x=self.conv3(x)
        x=self.conv4(x)
        x=e2nn.tensor_directsum([x,x_block1])   
        x=self.maxpool2(x)
        x_block2 = x

        x=self.conv5(x)
        x=self.conv6(x)
        x=e2nn.tensor_directsum([x,x_block2])
        x=self.maxpool3(x)

        x_block3 = x
        x=self.conv7(x)
        x=self.conv8(x)
        x=e2nn.tensor_directsum([x,x_block3])


        x=self.maxpool4(x)
        x=self.final_conv(x)
        x=self.gpool(x)
        x = x.tensor

        # classify with the final fully connected layers)
        x = self.fully_net(x.reshape(x.shape[0], -1,))
        return x


class CUSTOM(nn.Module):

    """

    Model heavily based on the custom architecture provided in the paper:
    Matuszewski DJ, Sintorn IM. 2021. TEM virus images: Benchmark dataset and
        deep learning classification. Computer Methods and Programs in Biomedicine 209:
        106318.

    """
    #for this model, angular resolution is not relevant
    def __init__(self, n_classes,angular_resolution=None,mult=64):
        
        super(CUSTOM, self).__init__()
        
        #block 1
        #conv1
        in_type=1
        out_type=mult

        #The GCNN used a pre-convolution layer, for  a fair comparison; same is applied here. 
        self.pre_conv=nn.Sequential(
            nn.Conv2d(1,1,kernel_size=1,stride=1,padding=0,bias=False),
        )

        self.conv1=nn.Sequential(
            nn.Conv2d(in_type,out_type,kernel_size=3,stride=1, padding=1),
            nn.BatchNorm2d(out_type),
            nn.ReLU(inplace=True),
        )

        #conv2
        in_type=out_type
        self.conv2=nn.Sequential(
            nn.Conv2d(in_type,out_type-1,kernel_size=3,stride=1, padding=1),
            nn.BatchNorm2d(out_type-1),
            nn.ReLU(inplace=True),
        )

        #maxpool 1
        self.maxpool1=nn.Sequential(
            nn.MaxPool2d(stride=2,kernel_size=2, padding=0),
            nn.Dropout(p=0.2)
        )

        #block 2
        #conv3
        self.conv3=nn.Sequential(
            nn.Conv2d(in_type,out_type,kernel_size=3,stride=1, padding=1),     
            nn.BatchNorm2d(out_type),
            nn.ReLU(inplace=True),
        )

        #conv4
        self.conv4=nn.Sequential(
            nn.Conv2d(in_type,out_type,kernel_size=3,stride=1, padding=1),
            nn.BatchNorm2d(out_type),
            nn.ReLU(inplace=True),
        )

        #maxpool 2
        self.maxpool2=nn.Sequential(
            nn.MaxPool2d(stride=2,kernel_size=2, padding=0),
            nn.Dropout(p=0.25)
        )

        #block 3
        #conv5
        in_type *= 2
        out_type *= 2
        self.conv5=nn.Sequential(
            nn.Conv2d(in_type,out_type,kernel_size=3,stride=1, padding=1),
            nn.BatchNorm2d(out_type),
            nn.ReLU(inplace=True),
        )

        #conv6
        self.conv6=nn.Sequential(
            nn.Conv2d(in_type,out_type,kernel_size=3,stride=1, padding=1),
            nn.BatchNorm2d(out_type),
            nn.ReLU(inplace=True),
        )

        #maxpool 3
        self.maxpool3=nn.Sequential(
            nn.MaxPool2d(stride=4,kernel_size=4, padding=0),
            nn.Dropout(p=0.30)
        )

        #block 4
        #conv 7
        in_type *= 2
        self.conv7=nn.Sequential(
            nn.Conv2d(in_type,out_type,kernel_size=3,stride=1, padding=1),
            nn.BatchNorm2d(out_type),
            nn.ReLU(inplace=True),
        )

        #conv4
        in_type = out_type
        self.conv8=nn.Sequential(
            nn.Conv2d(in_type,out_type,kernel_size=3,stride=1, padding=1),
            nn.BatchNorm2d(out_type),
            nn.ReLU(inplace=True),
        )

        #maxpool 4
        self.maxpool4=nn.Sequential(
            nn.MaxPool2d(stride=4,kernel_size=4, padding=0),
            nn.Dropout(p=0.35)
        )

        in_type *=3
        self.finalconv=nn.Sequential(
            nn.Conv2d(in_type,out_type,padding=0,stride=4,kernel_size=4),
            nn.ReLU(inplace=True)
        )

        c=out_type
        #fully connected layer
        self.fully_net=nn.Sequential(
            nn.Linear(c,256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            #this dropout was inplace=True for my experiments. 
            nn.Dropout(p=0.35,inplace=False),
            nn.Linear(256,256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256,n_classes),
            nn.Softmax()
        )

    def forward(self, x: torch.Tensor):

        x=self.pre_conv(x)

        x_ini=x
        x=self.conv1(x)
        x=self.conv2(x)
        x=torch.cat((x,x_ini),axis=1)
        x=self.maxpool1(x)

        x_block1=x

        x=self.conv3(x)
        x=self.conv4(x)
        x=torch.cat((x,x_block1),axis=1)
        x=self.maxpool2(x)

        x_block2=x

        x=self.conv5(x)
        x=self.conv6(x)
        x=torch.cat((x,x_block2),axis=1)
        x=self.maxpool3(x)
        
        x_block3=x

        x=self.conv7(x)
        x=self.conv8(x)
        x=torch.cat((x,x_block3),axis=1)
        x=self.maxpool4(x)

        x=self.finalconv(x)

        # classify with the final fully connected layers)
        x = self.fully_net(x.reshape(x.shape[0], -1,))
        return x

class CUSTOM_no_batchNorm(nn.Module):

    """

    Model heavily based on the custom architecture provided in the paper:
    Matuszewski DJ, Sintorn IM. 2021. TEM virus images: Benchmark dataset and
        deep learning classification. Computer Methods and Programs in Biomedicine 209:
        106318.

    """
    #Angular resolution is irrelevant for this model. 
    def __init__(self, n_classes,angular_resolution=None,mult=64):
        
        super(CUSTOM_no_batchNorm, self).__init__()
        
        #block 1
        #conv1
        in_type=1
        out_type=mult

        #The GCNN used a pre-convolution layer, for  a fair comparison; same is applied here. 
        self.pre_conv=nn.Sequential(
            nn.Conv2d(1,1,kernel_size=1,stride=1,padding=0,bias=False),
        )

        self.conv1=nn.Sequential(
            nn.Conv2d(in_type,out_type,kernel_size=3,stride=1, padding=1),
            #nn.BatchNorm2d(out_type),
            nn.ReLU(inplace=True),
        )

        #conv2
        in_type=out_type
        self.conv2=nn.Sequential(
            nn.Conv2d(in_type,out_type-1,kernel_size=3,stride=1, padding=1),
            #nn.BatchNorm2d(out_type-1),
            nn.ReLU(inplace=True),
        )

        #maxpool 1
        self.maxpool1=nn.Sequential(
            nn.MaxPool2d(stride=2,kernel_size=2, padding=0),
            nn.Dropout(p=0.2)
        )

        #block 2
        #conv3
        self.conv3=nn.Sequential(
            nn.Conv2d(in_type,out_type,kernel_size=3,stride=1, padding=1),     
            #nn.BatchNorm2d(out_type),
            nn.ReLU(inplace=True),
        )

        #conv4
        self.conv4=nn.Sequential(
            nn.Conv2d(in_type,out_type,kernel_size=3,stride=1, padding=1),
            #nn.BatchNorm2d(out_type),
            nn.ReLU(inplace=True),
        )

        #maxpool 2
        self.maxpool2=nn.Sequential(
            nn.MaxPool2d(stride=2,kernel_size=2, padding=0),
            nn.Dropout(p=0.25)
        )

        #block 3
        #conv5
        in_type *= 2
        out_type *= 2
        self.conv5=nn.Sequential(
            nn.Conv2d(in_type,out_type,kernel_size=3,stride=1, padding=1),
            #nn.BatchNorm2d(out_type),
            nn.ReLU(inplace=True),
        )

        #conv6
        self.conv6=nn.Sequential(
            nn.Conv2d(in_type,out_type,kernel_size=3,stride=1, padding=1),
            #nn.BatchNorm2d(out_type),
            nn.ReLU(inplace=True),
        )

        #maxpool 3
        self.maxpool3=nn.Sequential(
            nn.MaxPool2d(stride=4,kernel_size=4, padding=0),
            nn.Dropout(p=0.30)
        )

        #block 4
        #conv 7
        in_type *= 2
        self.conv7=nn.Sequential(
            nn.Conv2d(in_type,out_type,kernel_size=3,stride=1, padding=1),
            #nn.BatchNorm2d(out_type),
            nn.ReLU(inplace=True),
        )

        #conv4
        in_type = out_type
        self.conv8=nn.Sequential(
            nn.Conv2d(in_type,out_type,kernel_size=3,stride=1, padding=1),
            #nn.BatchNorm2d(out_type),
            nn.ReLU(inplace=True),
        )

        #maxpool 4
        self.maxpool4=nn.Sequential(
            nn.MaxPool2d(stride=4,kernel_size=4, padding=0),
            nn.Dropout(p=0.35)
        )

        in_type *=3
        self.finalconv=nn.Sequential(
            nn.Conv2d(in_type,out_type,padding=0,stride=4,kernel_size=4),
            nn.ReLU(inplace=True)
        )


        c=out_type
        #fully connected layer
        self.fully_net=nn.Sequential(
            nn.Linear(c,256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            #this dropout was inplace=True for my experiments. 
            nn.Dropout(p=0.35,inplace=False),
            nn.Linear(256,256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256,n_classes),
            nn.Softmax()
        )

    def forward(self, x: torch.Tensor):

        x=self.pre_conv(x)

        x_ini=x
        x=self.conv1(x)
        x=self.conv2(x)
        x=torch.cat((x,x_ini),axis=1)
        x=self.maxpool1(x)

        x_block1=x

        x=self.conv3(x)
        x=self.conv4(x)
        x=torch.cat((x,x_block1),axis=1)
        x=self.maxpool2(x)

        x_block2=x

        x=self.conv5(x)
        x=self.conv6(x)
        x=torch.cat((x,x_block2),axis=1)
        x=self.maxpool3(x)
        
        x_block3=x

        x=self.conv7(x)
        x=self.conv8(x)
        x=torch.cat((x,x_block3),axis=1)
        x=self.maxpool4(x)

        x=self.finalconv(x)

        # classify with the final fully connected layers)
        x = self.fully_net(x.reshape(x.shape[0], -1,))
        return x

class VGG16(nn.Module):
    
    def __init__(self, n_classes,angular_resolution=None,mult=64):
        
        super(VGG16, self).__init__()
        

        #conv1
        in_type=1
        out_type=mult
        self.conv1=nn.Sequential(
            nn.Conv2d(in_type,out_type,kernel_size=3,stride=1, padding=1),
            nn.BatchNorm2d(out_type),
            nn.ReLU(inplace=True)
        )

        #conv2
        in_type=out_type
        self.conv2=nn.Sequential(
            nn.Conv2d(in_type,out_type,kernel_size=3,stride=1, padding=1),
            nn.BatchNorm2d(out_type),
            nn.ReLU(inplace=True)
        )

        #maxpool 1
        self.maxpool1=nn.Sequential(
            nn.MaxPool2d(stride=2,kernel_size=2, padding=0),
            nn.Dropout(p=0.2)
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
            nn.MaxPool2d(stride=2,kernel_size=2, padding=0),
            nn.Dropout(p=0.25)
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
            nn.MaxPool2d(stride=2,kernel_size=2, padding=0),
            nn.Dropout(p=0.30)
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
            nn.MaxPool2d(stride=4,kernel_size=4, padding=0),
            nn.Dropout(p=0.35)
        )

        #block 5
        in_type=out_type

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
            nn.Conv2d(in_type,out_type,kernel_size=4,stride=2, padding=0),
            nn.BatchNorm2d(out_type),
            nn.ReLU(inplace=True)
        )

        #maxpool 5
        self.maxpool5=nn.Sequential(
            nn.MaxPool2d(stride=1,kernel_size=3, padding=0),
            nn.Dropout(p=0.40)
            )

        #fully connected layer 1
        self.fully_net=nn.Sequential(
            nn.Linear(out_type,4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(4096),
            nn.Dropout(p=0.5),
            nn.Linear(4096,4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(4096),
            nn.Dropout(p=0.5),
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

class VGG16_no_batchnorm(nn.Module):
    
    def __init__(self, n_classes,angular_resolution=None,mult=64):
        
        super(VGG16_no_batchnorm, self).__init__()
        

        #conv1
        in_type=1
        out_type=mult
        self.conv1=nn.Sequential(
            nn.Conv2d(in_type,out_type,kernel_size=3,stride=1, padding=1),
            #nn.BatchNorm2d(out_type),
            nn.ReLU(inplace=True)
        )

        #conv2
        in_type=out_type
        self.conv2=nn.Sequential(
            nn.Conv2d(in_type,out_type,kernel_size=3,stride=1, padding=1),
            #nn.BatchNorm2d(out_type),
            nn.ReLU(inplace=True)
        )

        #maxpool 1
        self.maxpool1=nn.Sequential(
            nn.MaxPool2d(stride=2,kernel_size=2, padding=0),
            nn.Dropout(p=0.2)

        )

        #block 2
        #defining out put type for block
        in_type=out_type
        out_type *= 2
        #conv3
        self.conv3=nn.Sequential(
            nn.Conv2d(in_type,out_type,kernel_size=3,stride=1, padding=1),
            #nn.BatchNorm2d(out_type),
            nn.ReLU(inplace=True)
        )
        in_type=out_type

        #conv4
        self.conv4=nn.Sequential(
            nn.Conv2d(in_type,out_type,kernel_size=3,stride=1, padding=1),
            #nn.BatchNorm2d(out_type),
            nn.ReLU(inplace=True)
        )

        #maxpool 2
        self.maxpool2=nn.Sequential(
            nn.MaxPool2d(stride=2,kernel_size=2, padding=0),
            nn.Dropout(p=0.25)
        )

        #block3
        #defining out put type for block
        in_type=out_type
        out_type *= 2

        #conv 5
        self.conv5=nn.Sequential(
            nn.Conv2d(in_type,out_type,kernel_size=3,stride=1, padding=1),
            #nn.BatchNorm2d(out_type),
            nn.ReLU(inplace=True)
        )

        in_type = out_type
        #conv 6
        self.conv6=nn.Sequential(
            nn.Conv2d(in_type,out_type,kernel_size=3,stride=1, padding=1),
            #nn.BatchNorm2d(out_type),
            nn.ReLU(inplace=True)
        )

        #conv 7
        self.conv7=nn.Sequential(
            nn.Conv2d(in_type,out_type,kernel_size=3,stride=1, padding=1),
            #nn.BatchNorm2d(out_type),
            nn.ReLU(inplace=True)
        )

        #maxpool 3
        self.maxpool3=nn.Sequential(
            nn.MaxPool2d(stride=2,kernel_size=2, padding=0),
            nn.Dropout(p=0.30)
        )

        #block 4

        in_type=out_type
        out_type *= 2
        #conv8
        self.conv8=nn.Sequential(
            nn.Conv2d(in_type,out_type,kernel_size=3,stride=1, padding=1),
            #nn.BatchNorm2d(out_type),
            nn.ReLU(inplace=True)
        )
        in_type=out_type

        #conv9
        self.conv9=nn.Sequential(
            nn.Conv2d(in_type,out_type,kernel_size=3,stride=1, padding=1),
            #nn.BatchNorm2d(out_type),
            nn.ReLU(inplace=True)
        )

        #conv10
        self.conv10=nn.Sequential(
            nn.Conv2d(in_type,out_type,kernel_size=3,stride=1, padding=1),
            #nn.BatchNorm2d(out_type),
            nn.ReLU(inplace=True)
        )

        #maxpool 4
        self.maxpool4=nn.Sequential(
            nn.MaxPool2d(stride=4,kernel_size=4, padding=0),
            nn.Dropout(p=0.35)

        )

        #block 5
        in_type=out_type

        #conv11
        self.conv11=nn.Sequential(
            nn.Conv2d(in_type,out_type,kernel_size=3,stride=1, padding=1),
            #nn.BatchNorm2d(out_type),
            nn.ReLU(inplace=True)
        )
        in_type=out_type

        #conv12
        self.conv12=nn.Sequential(
            nn.Conv2d(in_type,out_type,kernel_size=3,stride=1, padding=1),
            #nn.BatchNorm2d(out_type),
            nn.ReLU(inplace=True)
        )

        #conv13
        self.conv13=nn.Sequential(
            nn.Conv2d(in_type,out_type,kernel_size=4,stride=2, padding=0),
            #nn.BatchNorm2d(out_type),
            nn.ReLU(inplace=True)
        )

        #maxpool 5
        self.maxpool5=nn.Sequential(
            nn.MaxPool2d(stride=1,kernel_size=3, padding=0),
            nn.Dropout(p=0.40)

            )

        #fully connected layer 1
        self.fully_net=nn.Sequential(
            nn.Linear(out_type,4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(4096),
            nn.Dropout(p=0.5),
            nn.Linear(4096,4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(4096),
            nn.Dropout(p=0.5),
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


class G_VGG16(nn.Module):

    def __init__(self, n_classes,angular_resolution=4,mult=32):
        
        super(G_VGG16, self).__init__()
        
        # the model is equivariant under rotations by 90 degrees, modelled by C4
        self.r2_act = gspaces.FlipRot2dOnR2(N=angular_resolution)
        #self.r2_act=gspaces.Rot2dOnR2(N=8)
        
        # the input image is a scalar field, corresponding to the trivial representation
        in_type = e2nn.FieldType(self.r2_act, [self.r2_act.trivial_repr])
        # we store the input type for wrapping the images into a geometric tensor during the forward pass
        self.input_type = in_type

        #block1
        #defining out put type for block
        out_type = e2nn.FieldType(self.r2_act,mult*[self.r2_act.regular_repr])

        #conv1
        self.conv1=e2nn.SequentialModule(
            e2nn.R2Conv(in_type,out_type,kernel_size=3,stride=1, padding=1),
            e2nn.InnerBatchNorm(out_type),
            e2nn.ReLU(out_type,inplace=True),
        )

        #conv2
        in_type=self.conv1.out_type
        self.conv2=e2nn.SequentialModule(
            e2nn.R2Conv(in_type,out_type,kernel_size=3,stride=1, padding=1),
            e2nn.InnerBatchNorm(out_type),
            e2nn.ReLU(out_type,inplace=True),
        )
        in_type=self.conv2.out_type
        #maxpool 1
        self.maxpool1=e2nn.SequentialModule(
            e2nn.PointwiseMaxPool(out_type,stride=2,kernel_size=2, padding=0),
            e2nn.PointwiseDropout(in_type,p=0.20)
        )

        #block 2
        #defining out put type for block
        out_type = e2nn.FieldType(self.r2_act,mult*2*[self.r2_act.regular_repr])

        #conv3
        self.conv3=e2nn.SequentialModule(
            e2nn.R2Conv(in_type,out_type,kernel_size=3,stride=1, padding=1),
            e2nn.InnerBatchNorm(out_type),
            e2nn.ReLU(out_type,inplace=True),
        )

        #conv4
        in_type=self.conv3.out_type
        self.conv4=e2nn.SequentialModule(
            e2nn.R2Conv(in_type,out_type,kernel_size=3,stride=1, padding=1),
            e2nn.InnerBatchNorm(out_type),
            e2nn.ReLU(out_type,inplace=True),
        )

        #maxpool 2
        in_type=self.conv4.out_type
        self.maxpool2=e2nn.SequentialModule(
            e2nn.PointwiseMaxPool(out_type,stride=2,kernel_size=2, padding=0),
            e2nn.PointwiseDropout(in_type,p=0.25)

        )

        #block3
        #defining out put type for block
        out_type = e2nn.FieldType(self.r2_act,mult*4*[self.r2_act.regular_repr])
        #conv 5
        self.conv5=e2nn.SequentialModule(
            e2nn.R2Conv(in_type,out_type,kernel_size=3,stride=1, padding=1),
            e2nn.InnerBatchNorm(out_type),
            e2nn.ReLU(out_type,inplace=True),
        )

        #conv 6
        in_type=self.conv5.out_type
        self.conv6=e2nn.SequentialModule(
            e2nn.R2Conv(in_type,out_type,kernel_size=3,stride=1, padding=1),
            e2nn.InnerBatchNorm(out_type),
            e2nn.ReLU(out_type,inplace=True),
        )

        #conv 7
        in_type=self.conv6.out_type
        self.conv7=e2nn.SequentialModule(
            e2nn.R2Conv(in_type,out_type,kernel_size=3,stride=1, padding=1),
            e2nn.InnerBatchNorm(out_type),
            e2nn.ReLU(out_type,inplace=True),
        )

        #maxpool 3
        in_type=self.conv7.out_type
        self.maxpool3=e2nn.SequentialModule(
            e2nn.PointwiseMaxPool(out_type,stride=2,kernel_size=2, padding=0),
            e2nn.PointwiseDropout(in_type,p=0.30)

        )

        #block 4

        #conv8
        out_type = e2nn.FieldType(self.r2_act,mult*8*[self.r2_act.regular_repr])
        self.conv8=e2nn.SequentialModule(
            e2nn.R2Conv(in_type,out_type,kernel_size=3,stride=1, padding=1),
            e2nn.InnerBatchNorm(out_type),
            e2nn.ReLU(out_type,inplace=True),
        )

        #conv9
        in_type=self.conv8.out_type
        self.conv9=e2nn.SequentialModule(
            e2nn.R2Conv(in_type,out_type,kernel_size=3,stride=1, padding=1),
            e2nn.InnerBatchNorm(out_type),
            e2nn.ReLU(out_type,inplace=True),
        )

        #conv10
        in_type=self.conv9.out_type
        self.conv10=e2nn.SequentialModule(
            e2nn.R2Conv(in_type,out_type,kernel_size=3,stride=1, padding=1),
            e2nn.InnerBatchNorm(out_type),
            e2nn.ReLU(out_type,inplace=True),
        )

        #maxpool 4
        in_type=self.conv10.out_type
        self.maxpool4=e2nn.SequentialModule(
            e2nn.PointwiseMaxPool(out_type,stride=4,kernel_size=4, padding=0),
            e2nn.PointwiseDropout(in_type,p=0.35)

        )

        #block 5
        #conv11
        out_type = e2nn.FieldType(self.r2_act,mult*8*[self.r2_act.regular_repr])
        self.conv11=e2nn.SequentialModule(
            e2nn.R2Conv(in_type,out_type,kernel_size=3,stride=1, padding=1),
            e2nn.InnerBatchNorm(out_type),
            e2nn.ReLU(out_type,inplace=True),
        )

        #conv12
        in_type=self.conv11.out_type
        self.conv12=e2nn.SequentialModule(
            e2nn.R2Conv(in_type,out_type,kernel_size=3,stride=1, padding=1),
            e2nn.InnerBatchNorm(out_type),
            e2nn.ReLU(out_type,inplace=True),
        )

        #conv13
        in_type=self.conv12.out_type
        self.conv13=e2nn.SequentialModule(
            e2nn.R2Conv(in_type,out_type,kernel_size=4,stride=2, padding=0),
            e2nn.InnerBatchNorm(out_type),
            e2nn.ReLU(out_type,inplace=True),
        )

        #maxpool 5
        self.maxpool5=e2nn.SequentialModule(
            e2nn.PointwiseMaxPool(out_type,stride=1,kernel_size=3, padding=0),
            e2nn.PointwiseDropout(out_type,p=0.40)
            )

        #group pooling
        #out_type=self.conv13.out_type
        self.gpool=e2nn.GroupPooling(out_type)
        c=self.gpool.out_type.size

        #fully connected layer 1
        self.fully_net=nn.Sequential(
            nn.Linear(c,4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(4096),
            nn.Dropout(p=0.5),
            nn.Linear(4096,4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(4096),
            nn.Dropout(p=0.5),
            nn.Linear(4096,n_classes),
            nn.Softmax()
        )

    def forward(self, input: torch.Tensor):
        # wrap the input tensor in a GeometricTensor
        # (associate it with the input type)
        x = e2nn.GeometricTensor(input, self.input_type)
        
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

class G_VGG16_no_batchnorm(nn.Module):

    def __init__(self, n_classes,angular_resolution=4,mult=32):
        
        super(G_VGG16_no_batchnorm, self).__init__()
        
        # the model is equivariant under rotations by 90 degrees, modelled by C4
        self.r2_act = gspaces.FlipRot2dOnR2(N=angular_resolution)
        
        # the input image is a scalar field, corresponding to the trivial representation
        in_type = e2nn.FieldType(self.r2_act, [self.r2_act.trivial_repr])
        # we store the input type for wrapping the images into a geometric tensor during the forward pass
        self.input_type = in_type

        #block1
        #defining out put type for block
        out_type = e2nn.FieldType(self.r2_act,mult*[self.r2_act.regular_repr])

        #conv1
        self.conv1=e2nn.SequentialModule(
            e2nn.R2Conv(in_type,out_type,kernel_size=3,stride=1, padding=1),
            #e2nn.InnerBatchNorm(out_type),
            e2nn.ReLU(out_type,inplace=True),
        )

        #conv2
        in_type=self.conv1.out_type
        self.conv2=e2nn.SequentialModule(
            e2nn.R2Conv(in_type,out_type,kernel_size=3,stride=1, padding=1),
            #e2nn.InnerBatchNorm(out_type),
            e2nn.ReLU(out_type,inplace=True),
        )

        #maxpool 1
        in_type=self.conv2.out_type
        self.maxpool1=e2nn.SequentialModule(
            e2nn.PointwiseMaxPool(out_type,stride=2,kernel_size=2, padding=0),
            e2nn.PointwiseDropout(out_type,p=0.20)
        )

        #block 2
        #defining out put type for block
        out_type = e2nn.FieldType(self.r2_act,mult*2*[self.r2_act.regular_repr])

        #conv3
        self.conv3=e2nn.SequentialModule(
            e2nn.R2Conv(in_type,out_type,kernel_size=3,stride=1, padding=1),
            #e2nn.InnerBatchNorm(out_type),
            e2nn.ReLU(out_type,inplace=True),
        )

        #conv4
        in_type=self.conv3.out_type
        self.conv4=e2nn.SequentialModule(
            e2nn.R2Conv(in_type,out_type,kernel_size=3,stride=1, padding=1),
            #e2nn.InnerBatchNorm(out_type),
            e2nn.ReLU(out_type,inplace=True),
        )

        #maxpool 2
        self.maxpool2=e2nn.SequentialModule(
            e2nn.PointwiseMaxPool(out_type,stride=2,kernel_size=2, padding=0),
            e2nn.PointwiseDropout(out_type,p=0.25)

        )

        #block3
        #defining out put type for block
        in_type=self.conv4.out_type
        out_type = e2nn.FieldType(self.r2_act,mult*4*[self.r2_act.regular_repr])


        #conv 5
        self.conv5=e2nn.SequentialModule(
            e2nn.R2Conv(in_type,out_type,kernel_size=3,stride=1, padding=1),
            #e2nn.InnerBatchNorm(out_type),
            e2nn.ReLU(out_type,inplace=True),
        )

        #conv 6
        in_type=self.conv5.out_type
        self.conv6=e2nn.SequentialModule(
            e2nn.R2Conv(in_type,out_type,kernel_size=3,stride=1, padding=1),
            #e2nn.InnerBatchNorm(out_type),
            e2nn.ReLU(out_type,inplace=True),
        )

        #conv 7
        in_type=self.conv6.out_type
        self.conv7=e2nn.SequentialModule(
            e2nn.R2Conv(in_type,out_type,kernel_size=3,stride=1, padding=1),
            #e2nn.InnerBatchNorm(out_type),
            e2nn.ReLU(out_type,inplace=True),
        )

        #maxpool 3
        self.maxpool3=e2nn.SequentialModule(
            e2nn.PointwiseMaxPool(out_type,stride=2,kernel_size=2, padding=0),
            e2nn.PointwiseDropout(out_type,p=0.30)

        )

        #block 4

        #conv8
        in_type=self.conv7.out_type
        out_type = e2nn.FieldType(self.r2_act,mult*8*[self.r2_act.regular_repr])
        self.conv8=e2nn.SequentialModule(
            e2nn.R2Conv(in_type,out_type,kernel_size=3,stride=1, padding=1),
            #e2nn.InnerBatchNorm(out_type),
            e2nn.ReLU(out_type,inplace=True),
        )

        #conv9
        in_type=self.conv8.out_type
        self.conv9=e2nn.SequentialModule(
            e2nn.R2Conv(in_type,out_type,kernel_size=3,stride=1, padding=1),
            #e2nn.InnerBatchNorm(out_type),
            e2nn.ReLU(out_type,inplace=True),
        )

        #conv10
        in_type=self.conv9.out_type
        self.conv10=e2nn.SequentialModule(
            e2nn.R2Conv(in_type,out_type,kernel_size=3,stride=1, padding=1),
            #e2nn.InnerBatchNorm(out_type),
            e2nn.ReLU(out_type,inplace=True),
        )

        #maxpool 4
        self.maxpool4=e2nn.SequentialModule(
            e2nn.PointwiseMaxPool(out_type,stride=4,kernel_size=4, padding=0),
            e2nn.PointwiseDropout(out_type,p=0.35)

        )

        #block 5
        #conv11
        in_type=self.conv10.out_type
        out_type = e2nn.FieldType(self.r2_act,mult*8*[self.r2_act.regular_repr])
        self.conv11=e2nn.SequentialModule(
            e2nn.R2Conv(in_type,out_type,kernel_size=3,stride=1, padding=1),
            #e2nn.InnerBatchNorm(out_type),
            e2nn.ReLU(out_type,inplace=True),
        )

        #conv12
        in_type=self.conv11.out_type
        self.conv12=e2nn.SequentialModule(
            e2nn.R2Conv(in_type,out_type,kernel_size=3,stride=1, padding=1),
            #e2nn.InnerBatchNorm(out_type),
            e2nn.ReLU(out_type,inplace=True),
        )

        #conv13
        in_type=self.conv12.out_type
        self.conv13=e2nn.SequentialModule(
            e2nn.R2Conv(in_type,out_type,kernel_size=4,stride=2, padding=0),
            #e2nn.InnerBatchNorm(out_type),
            e2nn.ReLU(out_type,inplace=True),
        )

        #maxpool 5
        self.maxpool5=e2nn.SequentialModule(
            e2nn.PointwiseMaxPool(out_type,stride=1,kernel_size=3, padding=0),
            e2nn.PointwiseDropout(out_type,p=0.40)

            )

        #group pooling
        #out_type=self.conv13.out_type
        self.gpool=e2nn.GroupPooling(out_type)
        c=self.gpool.out_type.size

        #fully connected layer 1
        self.fully_net=nn.Sequential(
            nn.Linear(c,4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(4096),
            nn.Dropout(p=0.5),
            nn.Linear(4096,4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(4096),
            nn.Dropout(p=0.5),
            nn.Linear(4096,n_classes),
            nn.Softmax()
        )

    def forward(self, input: torch.Tensor):
        # wrap the input tensor in a GeometricTensor
        # (associate it with the input type)
        x = e2nn.GeometricTensor(input, self.input_type)
        
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

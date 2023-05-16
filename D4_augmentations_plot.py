#script used in main_function to plot all D4-transformations of an image.
import numpy as np

import torch
import torch.nn as nn
import matplotlib.pyplot as pyplot

from torchvision.transforms import ToTensor
from dataset import TEM_dataset

from PIL import Image
import matplotlib.pyplot as plt

totensor=ToTensor()

tem_test = TEM_dataset(mode='test')
x, y = next(iter(tem_test))

def D4_augmentations_plot(x: Image):
    fig,ax =plt.subplots(nrows=2, ncols=4)

    for r in range(8):
        if r <= 3:
            angle = r * (90)
            x_transformed = totensor(x.rotate(angle, Image.BILINEAR)).reshape(1, 1, 256, 256)
        else:
            #flip-rotate the image
            angle = (r-4) * (90)
            x_transformed = totensor(x.rotate(angle,Image.BILINEAR).transpose(Image.FLIP_LEFT_RIGHT)).reshape(1,1,256,256)
        
        #for image plot
        y_axis= 0 if r <=3 else 1
        x_axis= r % 4
        ax[y_axis,x_axis].imshow(x_transformed.squeeze(), interpolation='nearest',cmap='gray')
        ax[y_axis,x_axis].set_yticklabels([])
        ax[y_axis,x_axis].set_xticklabels([])
        ax[y_axis,x_axis].set_xticks([])
        ax[y_axis,x_axis].set_yticks([])

    ax[0,0].set_ylabel('not flipped')
    ax[1,0].set_ylabel('flipped')
    ax[1,0].set_xlabel('0'+u'\N{DEGREE SIGN}')
    ax[1,1].set_xlabel('90'+u'\N{DEGREE SIGN}')
    ax[1,2].set_xlabel('180'+u'\N{DEGREE SIGN}')
    ax[1,3].set_xlabel('270'+u'\N{DEGREE SIGN}')
    plt.subplots_adjust(left=0.1,right=0.9,bottom=0.1,top=0.6)
    plt.savefig(f'./d4_whole_plot.png')
D4_augmentations_plot(x)

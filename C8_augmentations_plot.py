#script used in main to plot all C8-transformations of an image.
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

def C8_augmentation_plot(x: Image):
    fig,ax =plt.subplots(nrows=2, ncols=4)

    for r in range(8):
        angle = r * (45)
        x_transformed = totensor(x.rotate(angle, Image.BILINEAR)).reshape(1, 1, 256, 256)
        #flip-rotate the image        
        #for image plot
        y_axis= 0 if r <=3 else 1
        x_axis= r % 4
        ax[y_axis,x_axis].imshow(x_transformed.squeeze(), interpolation='nearest',cmap='gray')
        ax[y_axis,x_axis].set_yticklabels([])
        ax[y_axis,x_axis].set_xticklabels([])
        ax[y_axis,x_axis].set_xticks([])
        ax[y_axis,x_axis].set_yticks([])

    for r,ax in enumerate(ax.flat):
        ax.set(xlabel=str(r*45)+u'\N{DEGREE SIGN}')

    plt.subplots_adjust(left=0.1,right=0.9,bottom=0.1,top=0.6)
    plt.savefig('./c8_whole_plot.png')

#Calling the function
C8_augmentation_plot(x)

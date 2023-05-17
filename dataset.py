import numpy as np
import pandas as pd
from PIL import Image

#insert your own path to respective data set partition

class TEM_dataset():
    def __init__(self, mode, transform=None):
        assert mode in ['train','validation', 'test','augmented_train']

        FILE_PATH='./data/'
        
        if mode == "augmented_train":
            data = np.load(FILE_PATH+'TEM_augmented_train_compressed.npy',allow_pickle=True)

        elif mode == 'train':
            data = np.load(FILE_PATH+'TEM_train_compressed.npy',allow_pickle=True)

        elif mode == 'validation':
            data = np.load(FILE_PATH+'TEM_validation_compressed.npy',allow_pickle=True)

        else:
            data = np.load(FILE_PATH+'TEM_test_compressed.npy',allow_pickle=True)
        
        self.transform = transform
        
        self.images=data[:,0]
        self.labels=data[:,1]
                
    def __getitem__(self, index):
        image, label = self.images[index], self.labels[index]
        image = Image.fromarray(image)
        if self.transform is not None:
            image = self.transform(image)
        return image, label
    
    def __len__(self):
        return len(self.labels)
    
    def subset_maker(self,rate):
        indexes=np.random.choice(len(self.images),int(len(self.images)*rate),replace=False)
        images=self.images[indexes]
        labels=self.labels[indexes]
        self.images=images
        self.labels=labels
        return self

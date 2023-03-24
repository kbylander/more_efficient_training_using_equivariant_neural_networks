import numpy as np
import pandas as pd
from PIL import Image


class TEM_dataset():
    def __init__(self, mode, transform=None):
        assert mode in ['train','validation', 'test','augmented_train']
            
        if mode == "augmented_train":
            file='/mimer/NOBACKUP/groups/naiss2023-22-69/data/TEM-virus/Cells_train_rot90flipaug_93samplesperclass_14classes.txt'
            
        elif mode == 'train':
            file='/mimer/NOBACKUP/groups/naiss2023-22-69/data/TEM-virus/Cells_train_noaug_93samplesperclass_14classes.txt'

        elif mode == 'validation':
            file='/mimer/NOBACKUP/groups/naiss2023-22-69/data/TEM-virus/Cells_validation_14classes.txt'
            
        else:
            file='/mimer/NOBACKUP/groups/naiss2023-22-69/data/TEM-virus/Cells_test_14classes.txt'

        
        self.transform = transform

        data = np.loadtxt(file, delimiter=' ')
        
        if mode == "train":
            self.images = data[:, :-1].reshape(-1, 256, 256).astype(np.float32)
            self.labels = data[:, -1].astype(np.int64)
            self.num_samples = len(self.labels)
        else:
            self.images = data[:, :-1].reshape(-1, 256, 256).astype(np.float32)
            self.labels = data[:, -1].astype(np.int64)
            self.num_samples = len(self.labels)    
        
    def __getitem__(self, index):
        image, label = self.images[index], self.labels[index]
        image = Image.fromarray(image)
        if self.transform is not None:
            image = self.transform(image)
        return image, label
    
    def __len__(self):
        return len(self.labels)

NAME_CLASSES = ['Adenovirus', 'Astrovirus', 'CCHF', 'Cowpox', 'Ebola', 'Influenza', 'Lassa', 'Marburg', 'Nipah', 'Norovirus', 'Orf', 'Papilloma', 'Rift Valley', 'Rotavirus']

def count_instances(data):
    no_inst={}
    for i,(img,lab) in enumerate(data):
        if no_inst.get(lab)==None:
            no_inst[lab]= 1
        else: 
            no_inst[lab] += 1
    return no_inst

print('finished reading TEM_dataset class')
data = TEM_dataset(mode='test')
print('test data',len(data))
print(count_instances(data))

data = TEM_dataset(mode='train')
print('train data',len(data))
print(count_instances(data))

data = TEM_dataset(mode='validation')
print('validation data',len(data))
print(count_instances(data))

data = TEM_dataset(mode='augmented_train')
print('augmented train data',len(data))
print(count_instances(data))



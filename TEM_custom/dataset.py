
from torch.utils.data import Dataset
import numpy as np
from PIL import Image


class TEM_dataset(Dataset):
    def __init__(self,mode,data_aug=False,transform=None) -> None:
        print("fetching data")
        assert mode in ['train','validation','test']
        if mode == 'train':
            if data_aug:
                file='/mimer/NOBACKUP/groups/naiss2023-22-69/data/TEM-virus/Cells_train_rot90flipaug_93samplesperclass_14classes.txt'
            else:
                file='/mimer/NOBACKUP/groups/naiss2023-22-69/data/TEM-virus/Cells_train_noaug_93samplesperclass_14classes.txt'
        elif mode == 'validation':
            #might be changed at a later stage
            file='/mimer/NOBACKUP/groups/naiss2023-22-69/data/TEM-virus/Cells_validation_14classes.txt'
        else:
            file='/mimer/NOBACKUP/groups/naiss2023-22-69/data/TEM-virus/Cells_test_14classes.txt'

        self.transform=transform
        data = np.loadtxt(file,delimiter=' ')

        self.images=data[:, :-1].reshape(-1,256,256).astype(np.float32)
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

class MNIST_dataset(Dataset):
    def __init__(self,mode,data_aug=False,transform=None) -> None:
        print("fetching data")
        assert mode in ['train','test']
        if mode == 'train':
            file='/cephyr/users/karlby/Alvis/vgg16_related/mnist_data/mnist_all_rotation_normalized_float_train_valid.amat'
        else:
            file='/cephyr/users/karlby/Alvis/vgg16_related/mnist_data/mnist_all_rotation_normalized_float_test.amat'

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
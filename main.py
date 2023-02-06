import sys
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor, Compose
import numpy as np
from PIL import Image
import torch
from sklearn.metrics import confusion_matrix
from models import VGG16, C4_VGG16
from for_models import train,eval_test,test_model
from datasets import MnistRotDataset


"""
MODELS AVALIABLE: VGG16 and C4_VGG16 (1nd argument)
NUM_CLASSES: MNIST: 10, TEM_VIRUS: 14 (2nd argument)
NUM_EPOCHS (3rd argument)

"""

if __name__ == "__main__":
    device='cuda' if torch.cuda.is_available() else 'cpu'

    MODEL=sys.argv[1]
    NUM_CLASSES=int(sys.argv[2])
    NUM_EPOCHS=int(sys.argv[3])
    NAME_CLASSES=['zero','one','two','three','four','five','six','seven','eight','nine']


    predicted=[]
    target=[]

    totensor=ToTensor()

    #build model
    if MODEL=="VGG16":
        model = VGG16().to(device)
    else:
        model= C4_VGG16().to(device)
    # build the test set    
    
    raw_mnist_test = MnistRotDataset(mode='test')

    # retrieve the first image from the test set
    x, y = next(iter(raw_mnist_test))

    # evaluate the model without training
    test_model(model, x)

    train_transform = Compose([totensor])

    mnist_train= MnistRotDataset(mode='train',transform=train_transform)
    train_loader = torch.utils.data.DataLoader(mnist_train,batch_size=64)

    test_transform= Compose([totensor])
    mnist_test = MnistRotDataset(mode='test', transform=test_transform)
    test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=64)

    loss_function=torch.nn.CrossEntropyLoss()
    optimizer=torch.optim.Adam(model.parameters(),lr=5e-5,weight_decay=1e-5)

    print("entered training")
    accuracy=[]

    for epoch in range(NUM_EPOCHS):
        train(model,optimizer,loss_function,train_loader,accuracy)
        if True or epoch % 10 == 0: 
            eval_test(model,train_loader,accuracy,epoch)

    test_acc=[]
    #run test set when training is complete
    eval_test(model,test_loader,test_acc,predicted,target)

    # retrieve the first image from the test set
    x, y = next(iter(raw_mnist_test))

    # evaluate the model
    test_model(model, x)
    import pandas as pd
    import seaborn as sn
    import matplotlib.pyplot as plt

    cf_matrix=confusion_matrix(target,predicted)
    df_cm=pd.DataFrame(cf_matrix/np.sum(cf_matrix,axis=1),index = NAME_CLASSES, columns = NAME_CLASSES)    
    plt.figure(figsize=(15,10))
    sn.heatmap(df_cm,annot=True)
    plt.xlabel("Predicted class",fontsize=25)
    plt.ylabel("True class",fontsize=25)

    plt.savefig('cf_matrix_names_2.png')

    plt.figure()
    plt.plot(accuracy)
    plt.xlabel("epoch nr")
    plt.ylabel("accuracy")
    plt.savefig("p4_3.png")

from PIL import Image
from torchvision.transforms import ToTensor
import torch

device='cuda' if torch.cuda.is_available() else 'cpu'

totensor=ToTensor()

def test_model(model: torch.nn.Module, x: Image):
    print("entered testing")
    # evaluate the `model` on 4 rotated versions of the input image `x`
    model.eval()
    
    wrmup = model(torch.randn(1, 1, 28, 28).to(device))
    del wrmup
        
    print()
    print('##########################################################################################')
    header = 'angle |  ' + '  '.join(["{:5d}".format(d) for d in range(10)])
    print(header)
    with torch.no_grad():
        for r in range(4):
            x_transformed = totensor(x.rotate(r*90., Image.Resampling.BILINEAR)).reshape(1, 1, 28, 28)

            x_transformed = x_transformed.to(device)

            y = model(x_transformed)
            y = y.to('cpu').numpy().squeeze()
            
            angle = r * 90
            print("{:4d} : {}".format(angle, y))
    print('##########################################################################################')
    print()


def eval_test(model,datas,accuracy=None,predicted=None,target=None,epoch=0):
    total = 0
    correct = 0
    with torch.no_grad():
        model.eval()
        for i, (x, t) in enumerate(datas):

            x = x.to(device)
            t = t.to(device)
            
            y = model(x)

            _, prediction = torch.max(y.data, 1)
            total += t.shape[0]
            correct += (prediction == t).sum().item()
            if type(predicted) is list and type(target) is list:
                [predicted.append(i) for i in prediction]
                [target.append(i) for i in t]
    if type(accuracy)==list:
        accuracy.append(correct/total*100.)
    print(f"epoch {epoch} | test accuracy: {accuracy[-1]}")


def train(model,optimizer,loss_function,train_loader,accuracy):
    model.train()
    for i, (x, t) in enumerate(train_loader):
        
        optimizer.zero_grad()

        x = x.to(device)
        t = t.to(device)

        y = model(x)

        loss = loss_function(y, t)

        loss.backward()

        optimizer.step()

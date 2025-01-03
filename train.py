import torchvision
import torch
import torchvision.transforms as transforms
from model import ANN
import torch.optim as optim
import torch.nn as nn
import os
import sys
from focal_loss import FocalLoss

if len(sys.argv)==2 and (sys.argv[1]=='true' or sys.argv[1]=='false'):
    save_dir='model_parameters'
    epoch=50
    batch_size=64
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    lr=0.001

    if bool(sys.argv[1]):
        for file in os.listdir(save_dir):
            os.remove(f'{save_dir}/{file}')
    model=ANN().to(device)
    optimizer=optim.Adam(model.parameters(),lr)
    criteron=nn.CrossEntropyLoss()
    # criteron=FocalLoss(gamma=2)
    os.makedirs(save_dir,exist_ok=True)

    train_dataset = torchvision.datasets.MNIST(root='./data', 
                                            train=True, 
                                            transform=transforms.ToTensor(),  
                                            download=True)

    test_dataset = torchvision.datasets.MNIST(root='./data', 
                                            train=False, 
                                            transform=transforms.ToTensor())

    # Data loader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                            batch_size=batch_size, 
                                            shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                            batch_size=batch_size, 
                                            shuffle=False)

    for i in range(epoch):
        epoch_loss=0
        for iter,(image,label) in enumerate(train_loader):
            image=image.to(device)
            label=label.to(device)
            prediction=model(image)
            loss=criteron(prediction,label)
            epoch_loss+=loss
            if iter%10==0:
                print(f'epoch:{i}/{epoch} iteration:{iter}/{len(train_dataset)//batch_size+1} batch loss is :{loss:.4f}')
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        epoch_loss/=(len(train_dataset)//batch_size+1)
        print(f'Epoch:{i}/{epoch} Loss is :{epoch_loss:.4f}')
        if i%1==0:
            val_loss=0
            for image,label in test_loader:
                image=image.to(device)
                label=label.to(device)
                val_prediction=model(image)
                loss=criteron(val_prediction,label)
                val_loss+=loss
            val_loss/=(len(test_dataset)//batch_size+1)
            print(f'VAL loss :{val_loss:.4f}')
            
            torch.save(model.state_dict(),f'{save_dir}/epoch{i}_val_{val_loss}_train_{epoch_loss}')
            
    print('training completed !!')
    
elif len(sys.argv)==2:
    print('the command line argument must be boolean')
               
else:
    print("incomplete command line arguments, can not proceed further!!")
        
        
        
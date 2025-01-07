import argparse
import torchvision
import torch
from torchvision.transforms import transforms
from vae_loss import Combined_Loss
import torch.optim as optim
from vae_model import VAE
import os
from tqdm import tqdm
import logging

parser = argparse.ArgumentParser(description="train arguments")

parser.add_argument("--lr", type=float, default=0.0001, help="learning rate")
parser.add_argument('--batch',type=float,default=32,help='batch size')
parser.add_argument('--epoch',type=int,default=50,help='number of epoch')
parser.add_argument('--data_dir',type=str,default='/home/akshit/Desktop/workspace/python/MNIST/data',help="path of data")
parser.add_argument('--save_dir',type=str,default='model_parameter_VAE')

args = parser.parse_args()

logging.basicConfig(
    level=logging.INFO,                  
    format='%(asctime)s - %(levelname)s - %(message)s',  
    datefmt='%Y-%m-%d %H:%M:%S',        
    handlers=[
        # logging.StreamHandler(),        
        logging.FileHandler("train.log")  
    ]
)


os.makedirs(args.save_dir,exist_ok=True)
model=VAE()
criteron=Combined_Loss(2)
optimizer=optim.Adam(model.parameters(),args.lr)
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_dataset = torchvision.datasets.MNIST(root=args.data_dir, 
                                        train=True, 
                                        transform=transforms.Compose([transforms.ToTensor(),
                                                                      transforms.Lambda(lambda x:x*255)]),  
                                        download=True)

test_dataset = torchvision.datasets.MNIST(root=args.data_dir, 
                                        train=False, 
                                        transform=transforms.Compose([transforms.ToTensor(),
                                                                      transforms.Lambda(lambda x:x*255)]))


train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                        batch_size=args.batch, 
                                        shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                        batch_size=args.batch, 
                                        shuffle=False)


for i in tqdm(range(args.epoch)):
        epoch_loss=0
        for iter,(image,label) in enumerate(train_loader):
            image=image.to(device)
            label=label.to(device)
            prediction,mu,log_var=model(image,return_moments=True)
            loss=criteron(prediction,image,mu,log_var)
            epoch_loss+=loss
            if iter%10==0:
                logging.info(f'epoch:{i}/{args.epoch} iteration:{iter}/{len(train_dataset)//args.batch+1} batch loss is :{loss:.4f}')
                # print(f'epoch:{i}/{args.epoch} iteration:{iter}/{len(train_dataset)//args.batch+1} batch loss is :{loss:.4f}')
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        epoch_loss/=(len(train_dataset)//args.batch+1)
        logging.info(f'Epoch:{i}/{args.epoch} Loss is :{epoch_loss:.4f}')
        # print(f'Epoch:{i}/{args.epoch} Loss is :{epoch_loss:.4f}')
        if i%5==0:
            val_loss=0
            for image,label in test_loader:
                image=image.to(device)
                label=label.to(device)
                val_prediction,mu,log_var=model(image,return_moments=True)
                loss=criteron(val_prediction,image,mu,log_var)
                val_loss+=loss
            val_loss/=(len(test_dataset)//args.batch+1)
            # print(f'VAL loss :{val_loss:.4f}')
            logging.info(f'VAL loss :{val_loss:.4f}')
            
            torch.save(model.state_dict(),f'{args.save_dir}/epoch{i}_val_{val_loss}_train_{epoch_loss}')
            
print('training completed !!')
logging.info('training complete!!!')
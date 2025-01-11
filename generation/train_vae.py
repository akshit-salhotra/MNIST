import argparse
import torchvision
import torch
from torchvision.transforms import transforms
from vae_loss import Combined_Loss
import torch.optim as optim
from vae_model import VAE_conv
import os
from tqdm import tqdm
import logging
parser = argparse.ArgumentParser(description="train arguments")

# torch.cuda.manual_seed_all(21) 
# torch.manual_semanual_seed_all(21) 
# torch.manual_seed(21)ed(21)   

parser.add_argument("--lr", type=float, default=0.0001, help="learning rate")
parser.add_argument('--batch',type=float,default=32,help='batch size')
parser.add_argument('--epoch',type=int,default=50,help='number of epoch')
parser.add_argument('--data_dir',type=str,default='/home/akshit/Desktop/workspace/python/MNIST/data',help="path of data")
parser.add_argument('--save_dir',type=str,default='model_parameter_VAE')
parser.add_argument('--kl_weight',type=float,default=0.3,help='weight factor of kl divergence loss')
parser.add_argument('--save_freq',type=int,default=2,help='after how many epochs are the parameters saved')
parser.add_argument('--log_dir',type=str,default='logs',help='the directory in which training logs are to be saved')
parser.add_argument('--gamma',type=float,default=0.1,help='gamma for learning rate decay')
parser.add_argument('--step_size',type=int,default=2,help='number of epochs after which learning rate is to be decayed')
parser.add_argument('--model_path',type=str,default=None,help='path of model parameters to be loaded')
parser.add_argument('--latent_dim',type=int,default=128,help='dimensions of z')
args = parser.parse_args()

if os.path.exists(args.save_dir):
    num_folder=str(int(sorted(os.listdir(args.save_dir),key=lambda x:int(x))[-1])+1)
else:
    os.makedirs(args.save_dir,exist_ok=False)
    num_folder=0
save_path=args.save_dir+os.sep+num_folder

os.makedirs(args.log_dir,exist_ok=True)
logging.basicConfig(
    level=logging.INFO,                  
    format='%(asctime)s - %(levelname)s - %(message)s',  
    datefmt='%Y-%m-%d %H:%M:%S',        
    handlers=[
        # logging.StreamHandler(),        
        logging.FileHandler(args.log_dir+os.sep+f"train_{num_folder}.log")  
    ]
)
logging.info('------------------------------------------------------------------------------------------------')
logging.info('starting new training !!!!')
logging.info('------------------------------------------------------------------------------------------------')

logging.info(args)

    
os.makedirs(save_path)
logging.info(f'parameters are being saved at :{save_path}')
model=VAE_conv(28,args.latent_dim)
logging.info(model)
criteron=Combined_Loss(args.kl_weight)
optimizer=optim.Adam(model.parameters(),args.lr)
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

if args.model_path:
    model.load_state_dict(torch.load(args.model_path))
    print('loaded model parameters!')

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

pbar=tqdm(range(args.epoch))
for i in pbar:
        epoch_loss=0
        epoch_recons_loss=0
        epoch_kld_loss=0
        for iter,(image,label) in enumerate(train_loader):
            image=image.to(device)
            label=label.to(device)
            prediction,mu,log_var=model(image,return_moments=True)
            loss,[recons_loss,kl_d]=criteron(prediction,image,mu,log_var,return_sep=True)
            epoch_loss+=loss
            epoch_recons_loss+=recons_loss
            epoch_kld_loss+=kl_d
            if iter%50==0:
                logging.info(f'epoch:{i}/{args.epoch} iteration:{iter}/{len(train_dataset)//args.batch+1} batch loss is :{loss:.4f}')
                # print(f'epoch:{i}/{args.epoch} iteration:{iter}/{len(train_dataset)//args.batch+1} batch loss is :{loss:.4f}')
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        epoch_loss/=(len(train_dataset)//args.batch+1)
        epoch_kld_loss/=(len(train_dataset)//args.batch+1)
        epoch_recons_loss/=(len(train_dataset)//args.batch+1)
        logging.info(f'Epoch:{i}/{args.epoch} Loss is :{epoch_loss:.4f}')
        logging.info(f'average reconstruction loss is :{epoch_recons_loss:.4f}')
        logging.info(f'average kl divergence is : {epoch_kld_loss:.4f}')
        pbar.set_postfix({'average epoch loss':f'{epoch_loss:.4f}','recons loss':f'{epoch_recons_loss:.4f}','kld_loss':f'{epoch_kld_loss:.4f}'})
        # print(f'Epoch:{i}/{args.epoch} Loss is :{epoch_loss:.4f}')
        scheduler.step()
        if i%args.save_freq==0:
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
            
            torch.save(model.state_dict(),f'{save_path}/epoch{i}_val_{val_loss}_train_{epoch_loss}')
            
print('training completed !!')
logging.info('training complete!!!')
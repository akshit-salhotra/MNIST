from vae_model import VAE_conv
import torch
import argparse
import torchvision
import torchvision.transforms as transforms
import cv2
import numpy as np

parser = argparse.ArgumentParser(description="inference arguments")

parser.add_argument('--model_path',type=str,default='model_parameter_VAE/30/epoch12_val_344.32470703125_train_346.8779602050781',help='path of model parameters')
parser.add_argument('--data_dir',type=str,default='data',help='path of data')
parser.add_argument('--batch',type=int,default=32,help='batch size')
parser.add_argument('--latent_dim',type=int,default=128,help='dimensions of z')

args=parser.parse_args()

print(args)
model=VAE_conv(28,args.latent_dim)
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.load_state_dict(torch.load(args.model_path))


test_dataset = torchvision.datasets.MNIST(root=args.data_dir, 
                                        train=False, 
                                        transform=transforms.Compose([transforms.ToTensor(),
                                                                      transforms.Lambda(lambda x:x*255)]))


test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                        batch_size=args.batch, 
                                        shuffle=False)

with torch.no_grad():
    flag=0
    for image,label in test_loader:
        image=image.to(device)
        prediction=model(image)
        for im,pred in zip(image,prediction):

            
            cv2.namedWindow('images',cv2.WINDOW_NORMAL)
            
            images=cv2.hconcat([np.transpose((im.numpy()).astype(np.uint8),(1,2,0)),np.transpose((pred*255).numpy().astype(np.uint8),(1,2,0))])

            cv2.imshow('images',images)

            k=cv2.waitKey(0)
            if k==ord('q'):
                flag=1
                break
        if flag:
            break
        
    cv2.destroyAllWindows()
        
        




from vae_model import VAE
import torch
import argparse
import torchvision
import torchvision.transforms as transforms
import cv2
import numpy as np
import sys

parser = argparse.ArgumentParser(description="inference arguments")

parser.add_argument('--model_path',type=str,default='model_parameter_VAE/epoch20_val_2085.669189453125_train_2094.3955078125',help='path of model parameters')
parser.add_argument('--data_dir',type=str,default='data',help='path of data')
parser.add_argument('--batch',type=int,default=32,help='batch size')

args=parser.parse_args()

model=VAE()
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
    for image,label in test_loader:
        image=image.to(device)
        prediction=model(image)
        for im,pred in zip(image,prediction):
            # print(torch.unique(im))
            # sys.exit()
            # cv2.namedWindow('image',cv2.WINDOW_NORMAL)
            # cv2.namedWindow('pred',cv2.WINDOW_NORMAL)
            # cv2.imshow('image',np.transpose((im.numpy()).astype(np.uint8),(1,2,0)))
            # cv2.imshow('pred',np.transpose((pred.numpy()*255).astype(np.uint8),(1,2,0))
            
            cv2.namedWindow('images',cv2.WINDOW_NORMAL)
            
            images=cv2.hconcat([np.transpose((im.numpy()).astype(np.uint8),(1,2,0)),np.transpose((pred*255).numpy().astype(np.uint8),(1,2,0))])

            cv2.imshow('images',images)

            k=cv2.waitKey(0)
            if k==ord('q'):
                break
        
    cv2.destroyAllWindows()
        
        




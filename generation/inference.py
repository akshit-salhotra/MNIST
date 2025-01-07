from vae_model import VAE_conv
import torch
import argparse
import torchvision
import torchvision.transforms as transforms
import cv2
import numpy as np

parser = argparse.ArgumentParser(description="inference arguments")

parser.add_argument('--model_path',type=str,default='/home/akshit/Desktop/workspace/python/MNIST/model_parameter_VAE/5/epoch15_val_485.9570617675781_train_487.7302551269531',help='path of model parameters')
parser.add_argument('--data_dir',type=str,default='data',help='path of data')
parser.add_argument('--batch',type=int,default=32,help='batch size')

args=parser.parse_args()

model=VAE_conv(28,128)
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

            
            cv2.namedWindow('images',cv2.WINDOW_NORMAL)
            
            images=cv2.hconcat([np.transpose((im.numpy()).astype(np.uint8),(1,2,0)),np.transpose((pred*255).numpy().astype(np.uint8),(1,2,0))])

            cv2.imshow('images',images)

            k=cv2.waitKey(0)
            if k==ord('q'):
                break
        
    cv2.destroyAllWindows()
        
        




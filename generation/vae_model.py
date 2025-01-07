import torch.nn as nn
import torch
from torchsummary import summary

class VAE(nn.Module):
    def __init__(self,latent_dim=64):
        super().__init__()
        self.ReLU=nn.ReLU()
        self.encode0=nn.Linear(28**2,512)
        self.encode1=nn.Linear(512,256)
        self.encode2=nn.Linear(256,128)
        self.encode3=nn.Linear(128,64)
        
        
        self.encoder=nn.Sequential(self.encode0,
                                   self.ReLU,
                                   self.encode1,
                                   self.ReLU,
                                   self.encode2,
                                   self.ReLU,
                                   self.encode3,
                                   self.ReLU)
        
        self.decode0=nn.Linear(64,128)
        self.decode1=nn.Linear(128,256)
        self.decode2=nn.Linear(256,512)
        self.decode3=nn.Linear(512,28**2)
        
        self.project_back=nn.Linear(latent_dim,64)
        
        self.decoder=nn.Sequential(self.project_back,
                                   self.ReLU,
                                   self.decode0,
                                   self.ReLU,
                                   self.decode1,
                                   self.ReLU,
                                   self.decode2,
                                   self.ReLU,
                                   self.decode3,
                                   self.ReLU)
        
        self.mu=nn.Linear(64,latent_dim)
        self.log_var=nn.Linear(64,latent_dim)    
        
    @staticmethod
    def reparametrize(mu,log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu
        
    def forward(self,x,return_moments=False):
        x=x.reshape(-1,28**2)
        encoded=self.encoder(x)
        mu=self.mu(encoded)
        log_var=self.log_var(encoded)
        z=VAE.reparametrize(mu,log_var)
        output=self.decoder(z)
        output=output.reshape(-1,1,28,28)
        if return_moments:
            return output,mu,log_var
        return output           
        
if __name__=='__main__':
    model=VAE()
    summary(model,(1,28,28),2)   
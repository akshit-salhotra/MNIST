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
        
class VAE_conv(nn.Module):
    def __init__(self,input_size,latent_dim,in_channel=1):
        super().__init__()
        self.latent_dim=latent_dim
        self.input_size=input_size
        hidden_ch=[in_channel,32,64]
        self.hidden_ch=hidden_ch
        self.hidden_layers=len(hidden_ch)-1
        
        assert input_size/self.hidden_layers==input_size//self.hidden_layers,"improper input size"
        
        encoder=[VAE_conv.encoder_layer(hidden_ch[i],hidden_ch[i+1]) for i in range(len(hidden_ch)-1)]
        for i in range(2):
            encoder.append(nn.Sequential(nn.Conv2d(hidden_ch[-1],hidden_ch[-1],3,padding=1),nn.LeakyReLU()))
        self.encoder=nn.Sequential(*encoder)
        
        self.mu=nn.Linear(input_size**2//((2*(len(hidden_ch)-1))**2)*hidden_ch[-1],self.latent_dim)
        self.log_var=nn.Linear(input_size**2//((2*(len(hidden_ch)-1))**2)*hidden_ch[-1],self.latent_dim)
        self.project_back=nn.Linear(self.latent_dim,input_size**2//((2*(len(hidden_ch)-1))**2)*hidden_ch[-1])
        # print('the values of latent in:',input_size**2//((2*(len(hidden_ch)-1))**2)*hidden_ch[-1])
        decoder=[]
        for i in range(2):
            decoder.append(nn.Sequential(nn.Conv2d(hidden_ch[-1],hidden_ch[-1],3,padding=1),nn.LeakyReLU()))
        decoder.append(nn.Sequential(*[VAE_conv.decoder_layer(hidden_ch[i],hidden_ch[i-1]) for i in range(len(hidden_ch)-1,0,-1)]))
        self.decoder=nn.Sequential(*decoder)
              
    @staticmethod
    def encoder_layer(in_ch,out_ch):
        return nn.Sequential(nn.Conv2d(in_ch,out_ch,3,padding=1),nn.LeakyReLU(),nn.MaxPool2d(2,2))
    
    @staticmethod
    def decoder_layer(in_ch,out_ch):    
        return nn.Sequential(nn.ConvTranspose2d(in_ch,out_ch,2,2),nn.LeakyReLU())
    
    @staticmethod
    def reparametrize(mu,log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu
    
    def forward(self,x,return_moments=False):
        assert x.shape[-1]==x.shape[-2],"the image must have same height and width"
        
        batch,_,_,_=x.shape
        x=self.encoder(x)
        # print('the shape is :',x.shape)
        x=x.reshape(batch,-1)
        mu=self.mu(x)
        log_var=self.log_var(x)
        z=VAE_conv.reparametrize(mu,log_var)
        x=self.project_back(z)
        x=x.reshape(batch,self.hidden_ch[-1],self.input_size//(2*self.hidden_layers),self.input_size//(2*self.hidden_layers))
        x=self.decoder(x)
        
        if return_moments:
            return x,mu,log_var
        else:
            return x
         
if __name__=='__main__':
    model=VAE_conv(28,128)
    model(torch.ones((2,1,28,28)))
    summary(model,(1,28,28),2)   
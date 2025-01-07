import torch.nn as nn
import torch

class Combined_Loss(nn.Module):
    def __init__(self,weight_factor):
        super().__init__()
        self.weight=weight_factor
        self.MSE=nn.MSELoss()
    def forward(self,reconstructed,label,mu,log_var):
        recons_loss=self.MSE(reconstructed,label)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
        return recons_loss+ self.weight*kld_loss

        
    
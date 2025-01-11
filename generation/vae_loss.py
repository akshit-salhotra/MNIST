import torch.nn as nn
import torch

class Combined_Loss(nn.Module):
    def __init__(self,weight_factor):
        super().__init__()
        self.weight=weight_factor
        self.MSE=nn.MSELoss()
    def forward(self,reconstructed,label,mu,log_var,return_sep=False):
        recons_loss=self.MSE(reconstructed,label)
        if self.weight:
            kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
        else:
            kld_loss=0
        if return_sep:
            return recons_loss+ self.weight*kld_loss,[recons_loss,kld_loss]
        return recons_loss+ self.weight*kld_loss

        
    
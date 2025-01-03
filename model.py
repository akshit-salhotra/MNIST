import torch.nn as nn
import torch
class ANN(nn.Module):
    def __init__(self):
        super(ANN,self).__init__()
        self.layer1=nn.Linear(28**2,256)
        self.layer2=nn.Linear(256,128)
        self.layer3=nn.Linear(128,128)
        self.layer4=nn.Linear(128,128)
        self.layer5=nn.Linear(128,128)
        self.layer6=nn.Linear(128,64)
        self.final=nn.Linear(64,10)
        self.Relu=nn.ReLU()
    
    def forward(self,x):
        x=x.reshape(-1,28**2)
        # print(x.shape)
        x=self.Relu(self.layer1(x))
        x=self.Relu(self.layer2(x))
        x=self.Relu(self.layer3(x))
        x=self.Relu(self.layer4(x))
        x=self.Relu(self.layer5(x))
        x=self.Relu(self.layer6(x))
        x=self.Relu(self.final(x))
        return x

if __name__=='__main__':
    from torchsummary import summary
    model=ANN()
    model(torch.ones((2,28**2)))
    summary(model,(1,28,28),2)
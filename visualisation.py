from model import ANN
import matplotlib.pyplot as plt
import torch
model_path='model_parameters/epoch2_val_0.10381928086280823_train_0.09713428467512131'
model=ANN()
model.load_state_dict(torch.load(model_path))
weights=None
for name,parameter in model.named_parameters():
    if name=='layer1.weight':
        weights=parameter
print(weights.shape)
for i in range(0,392,4):
    for j in range(4):
        plt.subplot(2,2,j+1)
        weights_n=weights[i]
        weights_n=torch.reshape(weights_n,(28,28))
        plt.imshow(torch.sigmoid(weights_n).detach().numpy(), cmap='viridis', interpolation='nearest')
    plt.show()
    
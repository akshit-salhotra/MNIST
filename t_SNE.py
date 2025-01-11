import numpy as np
from sklearn.manifold import TSNE
from model import ANN
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from tqdm import tqdm

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model=ANN()
model_path='model_parameters_ANN/epoch18_val_0.09495199471712112_train_0.01756012998521328'
batch_size=128
model.load_state_dict(torch.load(model_path))

test_dataset = torchvision.datasets.MNIST(root='./data', 
                                            train=False, 
                                            transform=transforms.ToTensor())


test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                            batch_size=batch_size, 
                                            shuffle=False)

embeddings=[]
labels=[]

print('number of examples:',len(test_dataset))
with torch.no_grad():
    for image,label in tqdm(test_loader,desc='inference'):
        prediction=model.embedding(image).cpu().numpy()
        for pred,l in zip(prediction,label):
            embeddings.append(pred)
            labels.append(l)
            
X_embedded = TSNE(n_components=2, learning_rate='auto',
                  init='random', perplexity=50).fit_transform(np.array(embeddings))

print(X_embedded.shape)

data_dict={}
for i in range(10):
    data_dict[i]=[]
for i,label in zip(X_embedded,labels):
    data_dict[label.item()].append(list(i))
colors = ['red', 'blue', 'green', 'yellow', 'orange', 'purple', 'cyan', 'magenta', 'black', 'brown']
for i in range(10):
    print(data_dict[i])
    arr=np.array(data_dict[i])
    # print(len(data_dict[i]))
    plt.scatter(arr[:,0],arr[:,1],c=colors[i])
    
plt.show()

        

    
    

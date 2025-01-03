from model import ANN
import torch
import torchvision
import torchvision.transforms as transforms
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size=64
model_path='model_parameters/epoch2_val_0.10381928086280823_train_0.09713428467512131'
test_dataset = torchvision.datasets.MNIST(root='./data', 
                                            train=False, 
                                            transform=transforms.ToTensor())

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                            batch_size=batch_size, 
                                            shuffle=False)
model=ANN().to(device)
model.load_state_dict(torch.load(model_path))
print('model weights loaded')
predictions=[]
labels=[]
for image,label in test_loader:
    output_prediction=model(image)
    pred=torch.argmax(output_prediction,-1)
    # print(pred.shape)
    for val,la in zip(pred,label):
        predictions.append(val)
        labels.append(la)

matrix=confusion_matrix(labels,predictions)
disp = ConfusionMatrixDisplay(confusion_matrix=matrix)
disp.plot(cmap=plt.cm.Greens)

val_accuracy=(torch.tensor(torch.tensor(predictions)-torch.tensor(labels))==0).sum().item()/len(test_dataset)
print('validation accuracy is :',val_accuracy)

plt.show()
    

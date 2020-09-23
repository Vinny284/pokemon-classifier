import numpy as np
import torch
import torch.nn as nn 
from torch.utils.data import DataLoader
from models import *

test_size = 90 # max 90

# load test dataset
dataset = GraphicsDataset('images_test.csv', 'IMAGES_TEST', transform = transforms.Compose([transforms.Resize((100,100)),transforms.ToTensor()]))
testloader = DataLoader(dataset, batch_size=test_size, shuffle=True)

dataiter = iter(testloader)
images, labels = dataiter.next()

images = images.cuda()
labels = labels.cuda()

model = torch.load('poke_class90.pt')
model.eval()

# pass test set through model
label = model.forward(images, test_size)


soft = nn.Softmax(dim=1)
confusion_matrix = np.zeros((3,3))

for i in range(len(images)):
    
    actual = labels[i].item()
    pred = torch.argmax(soft(label)[i]).item()
    
    confusion_matrix[actual,pred] += 1
    
accuracy = 100*np.trace(confusion_matrix)/test_size

print(confusion_matrix)
print(accuracy)



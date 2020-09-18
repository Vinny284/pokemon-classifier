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


# performance metric
soft = nn.Softmax(dim=1)
correct = 0
for i in range(len(images)):
    if torch.argmax(soft(label)[i]).item() == labels[i].item():
        correct += 1
    else:
        display_image(images[i])
        print(dataset.classes[torch.argmax(soft(label)[i]).item()])
        pass

print(100*correct/test_size)

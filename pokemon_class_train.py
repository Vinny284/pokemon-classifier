import numpy as np
import torch
from torchvision import transforms
import torch.nn as nn 
from torch.utils.data import DataLoader
from models import *


train_size = 706 # max 706

# load in test dataset
dataset = GraphicsDataset('images.csv', 'IMAGES_TRAIN', transform = transforms.Compose([transforms.Resize((100,100)),transforms.ToTensor()]))
trainloader = DataLoader(dataset, batch_size=train_size, shuffle=True)

dataiter = iter(trainloader)
images, labels = dataiter.next()

images = images.cuda()
labels = labels.cuda()
    
    
# set learning parameters and loss function
learning_rate = 0.1
n_iters = 2000

model = ConvModel().cuda()
optimiser = torch.optim.SGD(model.parameters(), lr=learning_rate)
loss = nn.CrossEntropyLoss()


# execute training loop
for i in range(n_iters):
    label_pred = model.forward(images, train_size)
    l = loss(label_pred, labels)
    l.backward()
    optimiser.step()
    optimiser.zero_grad()
    if i % 100 == 0:
        print(l)


torch.save(model, 'poke_class.pt')















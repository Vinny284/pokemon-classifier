import numpy as np
import torch
from torchvision import transforms
import torch.nn as nn 
from torch.utils.data import DataLoader
from models import *

# the number of saamples used to train the model
train_size = 706 # max 706

# The images are resized to 100x100 pixels
# I am using Pytorch's built in data loader
dataset = GraphicsDataset('images.csv', 'IMAGES_TRAIN', transform = transforms.Compose([transforms.Resize((100,100)),transforms.ToTensor()]))
trainloader = DataLoader(dataset, batch_size=train_size, shuffle=True)

dataiter = iter(trainloader)
images, labels = dataiter.next()

# assign features and labels and put them ont he GPU using cuda
images = images.cuda()
labels = labels.cuda()
    
    
# set learning parameters and loss function
learning_rate = 0.1
n_iters = 2000


# We are using a convolution neural network, a cross entopy loss function and gradient descent optimiser
model = ConvModel().cuda()
optimiser = torch.optim.SGD(model.parameters(), lr=learning_rate)
loss = nn.CrossEntropyLoss()


# execute training loop
for i in range(n_iters):
    # make a forward pass
    label_pred = model.forward(images, train_size)
    
    # calculate loss function
    l = loss(label_pred, labels)
    
    #calculate gradient for optimisation
    l.backward()
    
    # update model paramters using SGD
    optimiser.step()
    
    # reset gradient variables for next loop
    optimiser.zero_grad()
    
    # print number of steps during training per 100
    if i % 100 == 0:
        print(l)

# save the model
torch.save(model, 'poke_class.pt')















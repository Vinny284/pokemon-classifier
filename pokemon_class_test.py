import numpy as np
import torch
import torch.nn as nn 
from torch.utils.data import DataLoader
from models import *

# the numebr of images used for testing 
test_size = 90 # max 90

# load test dataset and resize to 100x100
dataset = GraphicsDataset('images_test.csv', 'IMAGES_TEST', transform = transforms.Compose([transforms.Resize((100,100)),transforms.ToTensor()]))
testloader = DataLoader(dataset, batch_size=test_size, shuffle=True)

dataiter = iter(testloader)
images, labels = dataiter.next()

# assign features and labels
images = images.cuda()
labels = labels.cuda()

model = torch.load('poke_class90.pt')
model.eval()

# pass test set through model
label = model.forward(images, test_size)

# The network outputs 3 numbers and a softmax function is applied to give a probability distribution
# during training this is done in the loss function so here we need to apply it manually
soft = nn.Softmax(dim=1)
confusion_matrix = np.zeros((3,3))

# create confusion matrix
for i in range(len(images)):
    actual = labels[i].item()
    pred = torch.argmax(soft(label)[i]).item()
    confusion_matrix[actual,pred] += 1

# accuracy is a good performance metric as all the classes are balanced
accuracy = 100*np.trace(confusion_matrix)/test_size

#print(confusion_matrix)
print('Accuracy: ' + str(accuracy) + '%')



import numpy as np
import torch
from torchvision import transforms
import torch.nn as nn 
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import os
import pandas as pd
from PIL import Image

# a utility function that displays and image
def display_image(image):
    trans = transforms.ToPILImage()
    plt.imshow(trans(image.cpu()))
    plt.show()


# Convolutional Neural Network
class ConvModel(nn.Module):
    def __init__(self):
        super(ConvModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 5)
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.conv3 = nn.Conv2d(32, 64, 5)
        self.pool = nn.MaxPool2d(2, 2)
        
        self.fc1 = nn.Linear(5184, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 3)
        self.relu = nn.ReLU()
        
        # the forward pass takes 2 paramters, the input tensor and the sample size
        # the input must be a whole number of 100x100 pixel RGB images
        # the value 5184 is the number of features after flattening and depends on the input image size
    def forward(self, x, batch):
        out = self.pool(self.relu(self.conv1(x)))
        out = self.pool(self.relu(self.conv2(out)))
        out = self.pool(self.relu(self.conv3(out)))
        
        out = out.view(batch, 5184)
        out = self.relu(self.fc1(out))
        out = self.relu(self.fc2(out))
        out = self.fc3(out)
        return out



# Dataset construction
# I am using Pytorch's built in data loader so I needed to construct a dataset
# This was the easiest method for loading in images I could find
# this class requires a csv file with the names of each image and it's associated label
class GraphicsDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file, header=None)
        self.transform = transform
        self.root_dir = root_dir
        self.classes = self.annotations[1].unique()
     
    # returns number of samples in the dataset
    def __len__(self):
        return len(self.annotations)
    
    # returns an image in tensor form along with the associated label
    def __getitem__(self, index):
        # find image path
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 0])
        # open the image
        image = Image.open(img_path).convert('RGB')
        
        # assign y-label 0, 1 or 2
        for i in range(len(self.classes)):
            if self.annotations.iloc[index, 1] == self.classes[i]:
                y_label = torch.tensor(i)
                break
        # apply transformation 
        # the transform given should convert to tensor and resize to 100x100
        if self.transform: 
            image = self.transform(image)
            
        return (image, y_label)
    

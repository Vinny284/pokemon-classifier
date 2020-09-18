import numpy as np
import torch
from torchvision import transforms
import torch.nn as nn 
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import os
import pandas as pd
from PIL import Image


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
class GraphicsDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file, header=None)
        self.transform = transform
        self.root_dir = root_dir
        self.classes = self.annotations[1].unique()
        
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 0])
        image = Image.open(img_path).convert('RGB')
        
        for i in range(len(self.classes)):
            if self.annotations.iloc[index, 1] == self.classes[i]:
                y_label = torch.tensor(i)
                break
        
        if self.transform: 
            image = self.transform(image)
            
        return (image, y_label)
    

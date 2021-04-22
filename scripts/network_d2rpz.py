import torch.nn as nn
import torch
import numpy as np
import random

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.relu = nn.LeakyReLU(0.1, inplace=True)
        self.sigm = nn.Sigmoid()
        self.conv1 = nn.Conv2d(1, 32, [3, 5], 1, bias=False)
        self.conv2 = nn.Conv2d(32, 64, [3, 3], 1, padding=0, bias=False)
        self.conv3 = nn.Conv2d(64, 128, [3, 3], 1, padding=0, bias=False)
        self.fc3 = nn.Conv2d(256,126,1,1,bias=False)
        self.fc4 = nn.Conv2d(128,3,1,1,bias=False)

    def forward(self, input):
        x = self.relu(self.conv1(input))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.fc4(x)
        return x

def mse_loss(output, target):
    loss = torch.sqrt(torch.mean((output - target) ** 2))
    return loss


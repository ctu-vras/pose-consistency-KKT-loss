import torch.nn as nn
import torch
import numpy as np
import random

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        chanells = 16

        self.relu = nn.LeakyReLU(0.1, inplace=True)
        self.pool = nn.MaxPool2d(2, 2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')

        # encoding layers 
        self.conv1_en0 = nn.Conv2d(2, chanells, 5, 1, 2)
        self.conv1_en1 = nn.Conv2d(chanells, chanells, 5, 1, 2, bias=True)
        self.conv1_en2 = nn.Conv2d(chanells, chanells, 5, 1, 2, bias=True)

        self.conv2_en0 = nn.Conv2d(chanells, 2*chanells, 5, 1, 2)
        self.conv2_en1 = nn.Conv2d(2*chanells, 2*chanells, 5, 1, 2, bias=True)
        self.conv2_en2 = nn.Conv2d(2*chanells, 2*chanells, 5, 1, 2, bias=True)
        self.conv2_en3 = nn.Conv2d(2*chanells, 2*chanells, 5, 1, 2, bias=True)

        self.conv3_en0 = nn.Conv2d(2*chanells, 4*chanells, 5, 1, 2)
        self.conv3_en1 = nn.Conv2d(4*chanells, 4*chanells, 5, 1, 2, bias=True)
        self.conv3_en2 = nn.Conv2d(4*chanells, 4*chanells, 5, 1, 2, bias=True)
        self.conv3_en3 = nn.Conv2d(4*chanells, 4*chanells, 5, 1, 2, bias=True)

        # decoding layers
        self.conv3_de = nn.Conv2d(chanells, 2, 3, 1, 1, bias=False)
        self.conv2_de = nn.Conv2d(2*chanells, chanells, 3, 1, 1,bias=False)
        self.conv1_de = nn.Conv2d(4*chanells, 2*chanells, 3, 1, 1,bias=False)

        # scipped connections
        self.conv3_sc = nn.Conv2d(chanells, 1, 3, 1, 1, bias=False)
        self.conv2_sc = nn.Conv2d(2 * chanells, chanells, 3, 1, 1, bias=False)
        self.conv1_sc = nn.Conv2d(4 * chanells, 2 * chanells, 3, 1, 1, bias=False)

	#confidence layers
        self.conv1_en0_conf = nn.Conv2d(2, chanells, 5, 1, 2, bias=True)
        self.conv2_en0_conf = nn.Conv2d(chanells, 2*chanells, 5, 1, 2,bias=True)
        self.conv1_de_conf = nn.Conv2d(2*chanells, chanells, 3, 1, 1,bias=True)
        self.conv2_de_conf = nn.Conv2d(chanells, 1, 3, 1, 1,bias=True)

    def forward(self, input):
        x = self.pool(self.relu(self.conv1_en0(input)))
        x = self.relu(self.conv1_en1(x))
        x1 = self.relu(self.conv1_en2(x))

        x = self.pool(self.relu(self.conv2_en0(x1)))
        x = self.relu(self.conv2_en1(x))
        x = self.relu(self.conv2_en2(x))
        x2 = self.relu(self.conv2_en3(x))

        x = self.pool(self.relu(self.conv3_en0(x2)))
        x = self.relu(self.conv3_en1(x))
        x = self.relu(self.conv3_en2(x))
        x3 = self.relu(self.conv3_en3(x))

        x = self.upsample(x3)
        x = self.relu(self.conv1_de(x))
        x2_cat = torch.cat([x, x2], 1)
        x = self.relu(self.conv1_sc(x2_cat))

        x = self.upsample(x2)
        x = self.relu(self.conv2_de(x))
        x1_cat = torch.cat([x, x1], 1)
        x = self.relu(self.conv2_sc(x1_cat))

        x = self.upsample(x)
        x_inp_cat = torch.cat([x, input], 1)

        x = (self.conv3_de(x))

        return x


def weighted_mse_loss(output, target, weight):


    pred = (output[:, 0, :, :][np.newaxis]).permute([1, 0, 2, 3])
    loss_prediction = torch.sqrt((torch.sum(weight * (pred - target) ** 2)/weight.sum()))

    loss_bce = nn.BCELoss(reduction='mean')
    sigm = nn.Sigmoid()
    conf = (output[:, 1, :, :][np.newaxis]).permute([1, 0, 2, 3])
    loss_confidence = loss_bce(sigm(conf), (weight != 0).float())
    loss = loss_prediction + loss_confidence
    return loss, loss_prediction, loss_confidence




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

        #self.sp_conv1 = Sparse_Conv2d(1, 16, 5, 1, 2)
        #self.sp_conv2 = Sparse_Conv2d(16, 32, 5, 1, 2)
        #self.sp_conv3 = Sparse_Conv2d(32, 64, 5, 1, 2)
        #self.sp_conv4 = Sparse_Conv2d(64, 128, 5, 1, 2)


        # encoding layers   in_channels, out_channels, kernel_size, stride=1,
        #                  padding=0, dilation=1, groups=1, bias=True)
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
        self.conv3_sc = nn.Conv2d(4, 2, 3, 1, 1, bias=False)
        self.conv2_sc = nn.Conv2d(2 * chanells, chanells, 3, 1, 1, bias=False)
        self.conv1_sc = nn.Conv2d(4 * chanells, 2 * chanells, 3, 1, 1, bias=False)
    '''
	#confidence layers
        self.conv1_en0_conf = nn.Conv2d(2, chanells, 5, 1, 2, bias=True)
        self.conv2_en0_conf = nn.Conv2d(chanells, 2*chanells, 5, 1, 2,bias=True)
        self.conv1_de_conf = nn.Conv2d(2*chanells, chanells, 3, 1, 1,bias=True)
        self.conv2_de_conf = nn.Conv2d(chanells, 2, 3, 1, 1,bias=True)
    '''

    def forward(self, input):
        x = self.pool(self.relu(self.conv1_en0(input)))
        x = self.relu(self.conv1_en1(x))
        x1 = self.relu(self.conv1_en2(x))
        # m = self.pool(mask)

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

        x = self.upsample(x)
        x = self.relu(self.conv2_de(x))
        x1_cat = torch.cat([x, x1], 1)
        x = self.relu(self.conv2_sc(x1_cat))

        x = self.upsample(x)
        x = self.relu(self.conv3_de(x))
        x_inp_cat = torch.cat([x, input[:,0:2,:,:]], 1)
        x = self.conv3_sc(x_inp_cat)


        return x


class Sparse_Conv2d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride, padding):
        super(Sparse_Conv2d, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.conv_data = nn.Conv2d(self.in_ch, self.out_ch, self.kernel_size, self.stride, self.padding, bias=False)
        self.conv_mask = nn.Conv2d(1, self.out_ch, self.kernel_size, self.stride, self.padding, bias=False)
        self.conv_mask.weight.data = torch.ones(self.conv_mask.weight.data.shape)
        self.conv_mask.weight.requires_grad = False
        self.kernel_num_elem = kernel_size*kernel_size

    def forward(self, input, mask):
        x = self.conv_data(input)
        m = self.conv_mask(mask)
        sp = torch.div(m, self.kernel_num_elem)
        output = torch.mul(x, sp)
        return output


def weighted_mse_loss(output, target, weight,input):


    #loss_prediction = torch.sqrt(torch.mean(weight * (output[:, 0, :, :] - target) ** 2)*weight.sum())
    pred = (output[:, 0, :, :][np.newaxis]).permute([1, 0, 2, 3])
    loss_prediction = torch.sqrt((torch.sum(weight * (pred - target) ** 2)/weight.sum()))

    loss_bce = nn.BCELoss(reduction='mean')
    sigm = nn.Sigmoid()
    conf = (output[:, 1, :, :][np.newaxis]).permute([1, 0, 2, 3])
    loss_confidence = loss_bce(sigm(conf), (weight != 0).float())

    loss_pred_nonseen = torch.sqrt((torch.sum(weight[input==0] * (pred[input==0] - target[input==0]) ** 2)/weight[input==0].sum()))

    loss = loss_prediction + loss_confidence + loss_pred_nonseen
    return loss, loss_prediction, loss_confidence, loss_pred_nonseen


if __name__ == '__main__':
    model_s2d = Net()
    inp = torch.Tensor(np.ones([1,2,256,256]))
    out = model_s2d(inp)

    #print out.shape

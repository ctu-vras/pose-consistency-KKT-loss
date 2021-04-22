import torch
import network_d2rpz
import dataset_d2rpz
from time import gmtime, strftime
from tensorboardX import SummaryWriter
import os
import torch.optim as optim
import numpy as np
import torch.nn as nn

if __name__ == '__main__':
    epochs = 1000
    batch_size = 1

    runtime = strftime("%Y-%m-%d_%H:%M:%S", gmtime())
    output_file = "../data/d2rpz_network/run_"+runtime
    writer = SummaryWriter('../data/d2rpz_network/tensorboardX/run_' + runtime)

    if not os.path.exists(output_file):
        os.makedirs(output_file)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    net = network_d2rpz.Net()
    net.to(device)

    dataset_trn = dataset_d2rpz.Dataset("../data/d2rpz_labels/")
    trainloader = torch.utils.data.DataLoader(dataset_trn, batch_size=batch_size, shuffle=True, num_workers=0)

    optimizer = optim.Adam(net.parameters(), lr=0.0001)
    seen = 0
    for epoch in range(epochs):  # loop over the dataset multiple times

        epoch_loss = 0
        for i, data in enumerate(trainloader):
            input = data['input']
            label = data['label']
            #input_mask = data['mask']
            #weights = data['weights']
            input, label = input.to(device), label.to(device)

            output = net(input)
            loss = network_d2rpz.mse_loss(output, label)

            loss.backward()
            epoch_loss += loss
            optimizer.step()
            optimizer.zero_grad()
            writer.add_scalar('data/loss', loss, seen)


            seen += 1

        if epoch % 50 == 0:
            torch.save(net.state_dict(), output_file + '/net_epoch_{:06}'.format(epoch))
            input = input[0, :, :, :]
            input = input.cpu().numpy()
            # input = input.transpose((1, 2, 0))
            input = input - input.min()
            input = input / input.max()
            writer.add_image('data/Image', input, seen)
            label = label[0, :, :, :]
            label = label.cpu().numpy()
            label = label - label.min()
            label = label / label.max()
            writer.add_image('data/Label', label, seen)
            out = output[0, :, :, :].detach().cpu().numpy()
            out = out - out.min()
            out = out / out.max()
            writer.add_image('data/Output', out, seen)

        writer.add_scalar('data/epoch_loss', epoch_loss/dataset_trn.size, epoch)
        print(epoch_loss)
        epoch_loss = 0
        epoch_val_loss = 0


import torch
import network_s2d
import dataset_s2d
from time import gmtime, strftime
from tensorboardX import SummaryWriter
import os
import torch.optim as optim
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import math
from shutil import copyfile



if __name__ == '__main__':
    epochs = 200
    batch_size = 48
    learning_rate = np.ones([epochs, 1])*0.1


    runtime = strftime("%Y-%m-%d_%H:%M:%S", gmtime())
    output_file = "../data/s2d_network/run_"+runtime
    writer = SummaryWriter('../data/s2d_network/tensorboardX/run_' + runtime)

    if not os.path.exists(output_file):
        os.makedirs(output_file)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    net = network_s2d.Net()
    net.to(device)

    dataset_trn = dataset_s2d.Dataset("../data/s2d_trn/")
    trainloader = torch.utils.data.DataLoader(dataset_trn, batch_size=batch_size, shuffle=True, num_workers=2)

    dataset_val = dataset_s2d.Dataset("../data/s2d_val/")
    valloader = torch.utils.data.DataLoader(dataset_val, batch_size=batch_size, shuffle=True, num_workers=2)

    copyfile('train_s2d.py', output_file + '/train_script.py')
    copyfile('network_s2d.py', output_file + '/network_s2d.py')

    seen = 0
    for epoch in range(epochs):  # loop over the dataset multiple times
        # optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=0.1)
        optimizer = optim.Adam(net.parameters(), lr=0.001)

        epoch_loss = 0
        epoch_loss_pred = 0
        epoch_loss_conf = 0
        for i, data in enumerate(trainloader):
            input = data['input']
            label = data['label_d']
            input_mask = data['mask']
            weights = data['weights']
            input, label, input_mask, weights = input.to(device), label.to(device), input_mask.to(device), weights.to(device)
            input_w_mask = torch.cat([input,input_mask],1)
            output = net(input_w_mask)
            loss, loss_pred, loss_conf = network_s2d.weighted_mse_loss(output, label, weights)

            loss.backward()
            epoch_loss += loss
            epoch_loss_pred += loss_pred
            epoch_loss_conf += loss_conf
            optimizer.step()
            optimizer.zero_grad()
            writer.add_scalar('data/loss', loss, seen)
            writer.add_scalar('data/loss_pred', loss_pred, seen)
            writer.add_scalar('data/loss_conf', loss_conf, seen)

            if i%100==0:
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
                # out = torch.sigmoid(output[0,:, :, :].clone())
                out = output[0, 0, :, :].detach().cpu().numpy()
                out = out - out.min()
                out = out / out.max()
                writer.add_image('data/Output', out[np.newaxis], seen)
                out = output[0, 1, :, :].detach().cpu().numpy()
                out = 1 / (1 + np.exp(-out))
                out = out - out.min()
                out = out / out.max()
                writer.add_image('data/Output_conf', out[np.newaxis], seen)
            seen += 1

        writer.add_scalar('data/epoch_loss', epoch_loss/dataset_trn.size, epoch)
        writer.add_scalar('data/epoch_loss_pred', epoch_loss_pred/dataset_trn.size, epoch)
        writer.add_scalar('data/epoch_loss_conf', epoch_loss_conf/dataset_trn.size, epoch)
        print(epoch_loss)
        epoch_loss = 0
        epoch_val_loss = 0
        epoch_val_loss_pred = 0
        epoch_val_loss_conf = 0
        for i, data in enumerate(valloader):
            input = data['input']
            label = data['label_d']
            input_mask = data['mask']
            weights = data['weights']
            input, label, input_mask, weights = input.to(device), label.to(device), input_mask.to(device), weights.to(device)
            input_w_mask = torch.cat([input,input_mask],1)
            with torch.no_grad():
                output = net(input_w_mask)
                loss,loss_pred,loss_conf = network_s2d.weighted_mse_loss(output, label, weights)
            epoch_val_loss += loss
            epoch_val_loss_pred += loss_pred
            epoch_val_loss_conf += loss_conf

            if i%100==0:
                input = input[0, :, :, :]
                input = input.cpu().numpy()
                # input = input.transpose((1, 2, 0))
                input = input - input.min()
                input = input / input.max()
                writer.add_image('data/val_Image', input, seen)
                label = label[0, :, :, :]
                label = label.cpu().numpy()
                label = label - label.min()
                label = label / label.max()
                writer.add_image('data/val_Label', label, seen)
                # out = torch.sigmoid(output[0,:, :, :].clone())
                out = output[0, 0, :, :].detach().cpu().numpy()
                out = out - out.min()
                out = out / out.max()
                writer.add_image('data/val_Output', out[np.newaxis], seen)
                out = output[0, 1, :, :].detach().cpu().numpy()
                out = 1 / (1 + np.exp(-out))
                out = out - out.min()
                out = out / out.max()
                writer.add_image('data/val_Output_conf', out[np.newaxis], seen)
            seen += 1

        writer.add_scalar('data/val_loss', epoch_val_loss / dataset_val.size, epoch)
        writer.add_scalar('data/val_loss_pred', epoch_val_loss_pred / dataset_val.size, epoch)
        writer.add_scalar('data/val_loss_conf', epoch_val_loss_conf / dataset_val.size, epoch)
        torch.save(net.state_dict(), output_file+'/net_epoch_{:04}'.format(epoch))



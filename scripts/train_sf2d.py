import torch
import network_sf2d
import dataset_sf2d
from time import gmtime, strftime
from tensorboardX import SummaryWriter
import os
import torch.optim as optim
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import math
from shutil import copyfile
#from project_features import Render



if __name__ == '__main__':
    epochs = 500
    batch_size = 8
    learning_rate = np.ones([epochs, 1])*0.1


    runtime = strftime("%Y-%m-%d_%H:%M:%S", gmtime())
    output_file = "../data/s2d_net/run_"+runtime
    writer = SummaryWriter('../data/s2d_net/tensorboardX/run_' + runtime)

    if not os.path.exists(output_file):
        os.makedirs(output_file)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    net = network_sf2d.Net()
    net.to(device)


    dataset_trn = dataset_sf2d.Dataset("../data/sf2d_trn/")
    trainloader = torch.utils.data.DataLoader(dataset_trn, batch_size=batch_size, shuffle=True, num_workers=4)

    dataset_val = dataset_sf2d.Dataset("../data/sf2d_val/")
    valloader = torch.utils.data.DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=4)




    


    seen = 0
    for epoch in range(epochs):  # loop over the dataset multiple times
        optimizer = optim.Adam(net.parameters(), weight_decay=0.001, lr=0.001)

        epoch_loss = 0
        epoch_loss_pred = 0
        epoch_loss_conf = 0
        epoch_loss_pred_ns = 0
        for i, data in enumerate(trainloader):
            input = data['input']
            label = data['label']
            input_mask = data['mask']
            weights = data['weights']
            imgs = data['images']
            T_baselink_zpr = data['T_baselink_zpr'].numpy()[0]
            input, label, input_mask, weights = input.to(device), label.to(device), input_mask.to(device), weights.to(device)
            features = data['features'][:, 0, :, :].to(device)
            input_w_mask = torch.cat([input,input_mask],1)


            output = net(input_w_mask)
            loss, loss_pred, loss_conf, loss_pred_nonseen = network_sf2d.weighted_mse_loss(output, label, weights,input)

            loss.backward()
            epoch_loss += loss
            epoch_loss_pred += loss_pred
            epoch_loss_conf += loss_conf
            epoch_loss_pred_ns += loss_pred_nonseen

            optimizer.step()
            optimizer.zero_grad()
    
            if i%100==0:
                max_val = 2

                input = input[0, :, :, :]
                input = input.cpu().numpy()
                input[0, 0, 0] = max_val
                input[0, 1, 1] = -max_val
                input[input > max_val] = max_val
                input[input < -max_val] = -max_val
                input = input - input.min()
                input = input / input.max()
                writer.add_image('data/Image', input, seen)
                label = label[0, :, :, :]
                label = label.cpu().numpy()
                label[0, 0, 0] = max_val
                label[0, 1, 1] = -max_val
                label[label > max_val] = max_val
                label[label < -max_val] = -max_val
                label = label - label.min()
                label = label / label.max()
                writer.add_image('data/Label', label, seen)
                out = output[0, 0, :, :].detach().cpu().numpy()
                out[0, 0] = max_val
                out[1, 1] = -max_val
                out[out > max_val] = max_val
                out[out < -max_val] = -max_val
                out = out - out.min()
                out = out / out.max()
                writer.add_image('data/Output', out[np.newaxis], seen)
        seen += 1

        writer.add_scalar('data/epoch_loss', epoch_loss/dataset_trn.size, epoch)
        writer.add_scalar('data/epoch_loss_pred', epoch_loss_pred/dataset_trn.size, epoch)
        writer.add_scalar('data/epoch_loss_conf', epoch_loss_conf/dataset_trn.size, epoch)
        writer.add_scalar('data/epoch_loss_ns', epoch_loss_pred_ns/dataset_trn.size, epoch)
        print(seen)
        print(epoch_loss)
        epoch_loss = 0
        epoch_val_loss = 0
        epoch_val_loss_pred = 0
        epoch_val_loss_pred_ns = 0
        epoch_val_loss_conf = 0
        torch.save(net.state_dict(), output_file+'/net_epoch_{:04}'.format(epoch))


        for i, data in enumerate(valloader):

            input = data['input']
            label = data['label']
            input_mask = data['mask']
            weights = data['weights']
            imgs = data['images']
            T_baselink_zpr = data['T_baselink_zpr'].numpy()[0]
            input, label, input_mask, weights = input.to(device), label.to(device), input_mask.to(device), weights.to(
                device)

            features = data['features'][:, 0, :, :].to(device)
            #input_w_mask = torch.cat([input,input_mask,features],1)
            input_w_mask = torch.cat([input,input_mask],1)

            with torch.no_grad():
                output = net(input_w_mask)
                loss, loss_pred, loss_conf, loss_ns = network_sf2d.weighted_mse_loss(output, label, weights,input)

            epoch_val_loss += loss
            epoch_val_loss_pred += loss_pred
            epoch_val_loss_conf += loss_conf
            epoch_val_loss_pred_ns += loss_ns

    
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

        writer.add_scalar('data/epoch_val_loss', epoch_val_loss / dataset_val.size, epoch)
        writer.add_scalar('data/epoch_val_loss_pred', epoch_val_loss_pred / dataset_val.size, epoch)
        writer.add_scalar('data/epoch_val_loss_conf', epoch_val_loss_conf / dataset_val.size, epoch)
        writer.add_scalar('data/epoch_val_loss_ns', epoch_val_loss_pred_ns / dataset_val.size, epoch)



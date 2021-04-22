import torch
import network_s2d
import network_d2rpz
import dataset_real_rpz
from time import gmtime, strftime
from tensorboardX import SummaryWriter
import os
import torch.optim as optim
import rigid_body_s2d_learning_v1
import numpy as np
import torch.nn as nn
import time
from shutil import copyfile

if __name__ == '__main__':
    epochs = 200
    batch_size = 1
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ##### Define ROBOT  #####
    length = 0.8
    width = 0.6
    grid_res = 0.1
    border_x = torch.tensor(length / 2)
    border_y = torch.tensor(width / 2)
    length_t = torch.tensor(length)
    width_t = torch.tensor(width)
    step_t = torch.tensor(grid_res)
    robot_grid = np.meshgrid(np.arange(-length / 2, length / 2 + grid_res, grid_res),
                             np.arange(-width / 2, width / 2 + grid_res, grid_res))
    LT = np.zeros_like(robot_grid[0], dtype=int)
    LT[0:2, :] = 1
    RT = np.zeros_like(robot_grid[0], dtype=int)
    RT[5:, :] = 1
    X = robot_grid[0]
    Y = robot_grid[1]
    Z = np.zeros_like(X)
    Z[(LT == 0) & (RT == 0)] += 0.1
    # Z = Z + 0.1  # Z.mean()
    N = Z.size
    P = np.stack((X.reshape([1, N]).squeeze(), Y.reshape([1, N]).squeeze(), Z.reshape([1, N]).squeeze()), 0)
    points = torch.tensor(P, dtype=torch.float32)

    ##### training outputs #####
    name_prefix = "p5_rpz_o1_"
    runtime = strftime("%Y-%m-%d_%H:%M:%S", gmtime())
    output_file = "../data/s2d2rpz_network/run_" + name_prefix + runtime
    writer = SummaryWriter('../data/s2d2rpz_network/tensorboardX/run_' + name_prefix + runtime)

    if not os.path.exists(output_file):
        os.makedirs(output_file)

    ##### networks init #####
    net_s2d = network_s2d.Net()
    net_s2d.load_state_dict(torch.load("../data/s2d_network/network_weights_s2d", map_location=device))
    net_s2d.to(device)
    net_d2rpz = network_d2rpz.Net()
    net_d2rpz.load_state_dict(torch.load("../data/d2rpz_network/net_weights_d2rpz", map_location=device))
    net_d2rpz.to(device)
    rb_optimizer = rigid_body_s2d_learning_v1.RigidBodyOptimizer(net_s2d)

    dataset_trn = dataset_real_rpz.Dataset("../data/s2d_trn/")
    trainloader = torch.utils.data.DataLoader(dataset_trn, batch_size=batch_size, shuffle=True, num_workers=0)

    dataset_val = dataset_real_rpz.Dataset("../data/s2d_val/")
    valloader = torch.utils.data.DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=2)

    copyfile('train_s2d2rpz.py', output_file + '/train_script_s2d2rpz.py')
    copyfile('rigid_body_s2d_learning_v1.py', output_file + '/rigid_body_s2d_learning.py')
    optimizer_s2d = optim.Adam(net_s2d.parameters(), lr=0.001)
    seen = 0
    for epoch in range(epochs):  # loop over the dataset multiple times
        # optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=0.1)

        epoch_loss_rpz = 0
        epoch_loss_dem = 0
        epoch_loss_dem_d = 0

        for i, data in enumerate(trainloader):

            input = data['input']
            label_rpz = data['label_rpz']
            mask = data['mask']
            label_dem = data['label_dem_d'].to(device)
            label_dem_p = data['label_dem'].to(device)
            weights = data['weights']
            yaw = data['yaw'].detach().cpu().numpy()
            weights = data['weights'].to(device)

            input_w_mask = torch.cat([input, mask], 1)
            input_w_mask = input_w_mask.to(device)
            dense = net_s2d(input_w_mask)
            output_rpz = net_d2rpz((dense[:, 0, :, :][np.newaxis]).permute([1, 0, 2, 3]))

            pos_mask = (~torch.isnan(label_rpz)).type(torch.float).to(device)

            if True:
                loss_dem, _, _ = network_s2d.weighted_mse_loss(dense, label_dem, weights)
                loss_dem.backward(retain_graph=True)

                label_rpz[torch.isnan(label_rpz)] = 0

                l_rpz = 0 * torch.ones(label_rpz.shape)
                rpz = label_rpz.cpu().detach().numpy()


                idx1, idx2 = np.where((rpz[0, 2, :, :])!=0)
                K = np.argmax(np.abs(rpz[:, 1, idx1, idx2] - output_rpz.cpu().detach().numpy()[:, 1, idx1, idx2].ravel()))
                p_mask = 0 * torch.ones(pos_mask.shape)

                l_rpz[:,:,idx1[K], idx2[K]] = label_rpz[:, :, idx1[K], idx2[K]]
                p_mask[:,:,idx1[K], idx2[K]] =1
                loss_rpz = torch.sqrt(
                    (torch.sum(p_mask.to(device) * (output_rpz - l_rpz.to(device)) ** 2) / p_mask.to(device).sum()))

                
                loss_rpz.backward(retain_graph=False)
                # epoch_loss_rpz += loss_rpz
                optimizer_s2d.step()
                optimizer_s2d.zero_grad()



            with torch.no_grad():
                label_rpz[torch.isnan(label_rpz)] = 0
                loss_rpz = torch.sqrt((torch.sum(pos_mask.detach().cpu() * (
                        output_rpz.detach().cpu() - label_rpz.detach().cpu()) ** 2) / pos_mask.detach().cpu().sum()))
                epoch_loss_rpz += loss_rpz.detach().cpu().numpy()
                _, loss_dem, _ = network_s2d.weighted_mse_loss(dense, label_dem_p, weights)
                epoch_loss_dem += loss_dem.detach().cpu().numpy()
                _, loss_dem_d, _ = network_s2d.weighted_mse_loss(dense, label_dem, weights)
                epoch_loss_dem_d += loss_dem_d.detach().cpu().numpy()

                # print(t-time.time())

            # writer.add_scalar('data/loss', loss, seen)
            seen += 1

        if True:  # epoch % 5 == 0:
            torch.save(net_s2d.state_dict(), output_file + '/net_s2d_epoch_{:06}'.format(epoch))
            input = input[0, :, :, :]
            input = input.cpu().numpy()
            # input = input.transpose((1, 2, 0))
            input = input - input.min()
            input = input / input.max()
            writer.add_image('data/Image', input, seen)
            label_dem = label_dem[0, :, :, :]
            label_dem = label_dem.cpu().numpy()
            label_dem = label_dem - label_dem.min()
            label_dem = label_dem / label_dem.max()
            writer.add_image('data/Label', label_dem, seen)
            # out = torch.sigmoid(output[0,:, :, :].clone())
            out = dense.detach().cpu().numpy()[0, 0, :, :][np.newaxis]
            out = out - out.min()
            out = out / out.max()
            writer.add_image('data/Output', out, seen)

        writer.add_scalar('data/epoch_loss_rpz', epoch_loss_rpz / dataset_trn.size, epoch)
        writer.add_scalar('data/epoch_loss_dem', epoch_loss_dem / dataset_trn.size, epoch)
        writer.add_scalar('data/epoch_loss_dem_d', epoch_loss_dem_d / dataset_trn.size, epoch)
        print([epoch, epoch_loss_rpz / dataset_trn.size, epoch_loss_dem / dataset_trn.size])

        val_epoch_loss_rpz = 0
        val_epoch_loss_dem = 0
        val_epoch_loss_dem_d = 0

        for i, data in enumerate(valloader):
            input = data['input']
            label_rpz = data['label_rpz']
            mask = data['mask']
            label_dem = data['label_dem_d'].to(device)
            label_dem_p = data['label_dem'].to(device)
            weights = data['weights']
            yaw = data['yaw'].detach().cpu().numpy()
            weights = data['weights'].to(device)
            input_w_mask = torch.cat([input, mask], 1)
            input_w_mask = input_w_mask.to(device)
            dense = net_s2d(input_w_mask)
            output_rpz = net_d2rpz((dense[:, 0, :, :][np.newaxis]).permute([1, 0, 2, 3]))
            pos_mask = (~torch.isnan(label_rpz)).type(torch.float).to(device)
            with torch.no_grad():
                label_rpz[torch.isnan(label_rpz)] = 0
                loss_rpz = torch.sqrt((torch.sum(pos_mask.detach().cpu() * (
                        output_rpz.detach().cpu() - label_rpz.detach().cpu()) ** 2) / pos_mask.detach().cpu().sum()))
                val_epoch_loss_rpz += loss_rpz.detach().cpu().numpy()
                _, loss_dem, _ = network_s2d.weighted_mse_loss(dense, label_dem_p, weights)
                val_epoch_loss_dem += loss_dem.detach().cpu().numpy()
                _, loss_dem_d, _ = network_s2d.weighted_mse_loss(dense, label_dem, weights)
                val_epoch_loss_dem_d += loss_dem_d.detach().cpu().numpy()
        writer.add_scalar('data/val_epoch_loss_rpz', val_epoch_loss_rpz / dataset_val.size, epoch)
        writer.add_scalar('data/val_epoch_loss_dem', val_epoch_loss_dem / dataset_val.size, epoch)
        writer.add_scalar('data/val_epoch_loss_dem_d', val_epoch_loss_dem_d / dataset_val.size, epoch)


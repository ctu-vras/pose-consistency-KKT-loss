import numpy as np
from network_s2d import Net, weighted_mse_loss
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm
import time
import os
from scipy import ndimage
from scipy.optimize import nnls

INPUT_PATH = '../data/s2d_newdata_rpz/'
OUTPUT_PATH = '../data/s2d_evaldata_rpz/rigid_body/output/003/'
CREATE_MOVIE = False  # True
EPOCHS = 5
# TRAINING_SAMPLES = np.linspace(0,1,1, dtype ='int')
TRAINING_SAMPLES = np.linspace(0, 300, 301, dtype='int')
VISU = False


# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class RigidBodyOptimizer():
    def __init__(self,net):
        self.optimizer = optim.Adam(net.parameters(), lr=0.0001)


    def optimize_network(self, net, input_w_mask, x_in, points, indexes, label_dem, weights, visu=False):
        # initialization
        #self.optimizer = optim.Adam(net.parameters(), lr=0.0001)
        device = torch.device("cpu")
        N = points.shape[1]  # number of robot points
        C = 100  # linear barrier penalty
        r1 = indexes[0]
        r2 = indexes[1]
        c1 = indexes[2]
        c2 = indexes[3]
        lamb = torch.tensor(1 * torch.ones(N, 1), requires_grad=True, dtype=torch.float32).to(device)
        x = torch.tensor(x_in, requires_grad=True, dtype=torch.float32).to(device)
        Ggx = torch.tensor(0 * torch.ones(x.shape[0], N), requires_grad=False, dtype=torch.float32).to(device)
        Lkkt = np.zeros([1, EPOCHS])

        for epoch in range(EPOCHS):
            #print("epoch:", epoch)

            # Compute feedforward pass and crop terrain under robot
            output_dem = net(input_w_mask.unsqueeze(0))

            loss_dem, _, _ =  weighted_mse_loss(output_dem, label_dem, weights)
            loss_dem = 100*loss_dem
            loss_dem.backward(retain_graph=True)
            outputs = output_dem[0, 0, r1:r2, c1:c2]

            # Compute LOSS
            loss, loss_energy, loss_collision, pt, min_heights = loss_KKT(x, points, outputs.squeeze(), visu)

            ##### Estimate optimal lambda wrt fixed heights and dual feasibility constraint
            Lgx = torch.autograd.grad(loss_energy + (lamb.squeeze() * loss_collision).sum(), x, create_graph=True,
                                      retain_graph=True)[0].clone().detach()
            Fgx = torch.autograd.grad(loss_energy, x, create_graph=True, retain_graph=True)[0].clone().detach()
            for ii in range(N):
                Ggx[:, ii] = torch.autograd.grad(loss_collision[ii], x, create_graph=True, retain_graph=True)[0].clone().detach()
            g = loss_collision.detach().numpy()
            bb = np.concatenate((np.reshape(Fgx.detach().numpy(), (6, 1)), np.zeros((63, 1)))).squeeze()
            AA = np.concatenate(
                (-np.tile(Ggx.detach().numpy(), (1, 1)), np.diagflat(np.tile(g, (1, 1)))))  # np.tile(g, (2, 1))
            try:
                s, acc = nnls(AA, bb)
            except:
                print('nnls failed')
                return
            lamb = torch.tensor(s.reshape(N, 1), requires_grad=False, dtype=torch.float32).to(device)

            ###### Optimize network wrt lamda
            LOSS = Lgx.pow(2).sum() + (lamb.squeeze() * loss_collision).pow(2).sum() + C * torch.relu(loss_collision).pow(
                2).sum()
            #print("LOSS-KKT: ", LOSS.detach().numpy(), " = Stationarity:", Lgx.pow(2).sum().detach().numpy()," + Complementary slackness:", (lamb.squeeze() * loss_collision).pow(2).sum().detach().numpy(), " + Primal feasibility:", C * torch.relu(loss_collision).pow(2).sum().detach().numpy())
            Lkkt[0, epoch] = LOSS.detach().numpy()
            LOSS.backward(retain_graph=False)
            if LOSS.detach().numpy() > 100:
                self.optimizer.zero_grad()
                return
            self.optimizer.step()
            # zero the parameter gradients
            self.optimizer.zero_grad()

         
        return

def loss_KKT(x, points, height, visu=False):
    device = torch.device("cpu")

    N = points.shape[1]
    length = 0.8
    width = 0.6
    grid_res = 0.1
    length_t = torch.tensor(length).to(device)
    width_t = torch.tensor(width).to(device)
    step_t = torch.tensor(grid_res).to(device)

    # create rotation matrix and translation vector from variable x
    Rx = torch.tensor(torch.zeros(3, 3), requires_grad=False, dtype=torch.float32).to(device)
    Ry = torch.tensor(torch.zeros(3, 3), requires_grad=False, dtype=torch.float32).to(device)
    Rz = torch.tensor(torch.zeros(3, 3), requires_grad=False, dtype=torch.float32).to(device)
    t = torch.tensor(torch.zeros(3, 1), requires_grad=False, dtype=torch.float32).to(device)

    Rx[0, 0] = 1
    Rx[1, 1] = torch.cos(x[0]).to(device)
    Rx[1, 2] = -torch.sin(x[0]).to(device)
    Rx[2, 1] = torch.sin(x[0]).to(device)
    Rx[2, 2] = torch.cos(x[0]).to(device)

    Ry[0, 0] = torch.cos(x[1]).to(device)
    Ry[0, 2] = torch.sin(x[1]).to(device)
    Ry[1, 1] = 1
    Ry[2, 0] = -torch.sin(x[1]).to(device)
    Ry[2, 2] = torch.cos(x[1]).to(device)

    # yaw does not apply
    Rz[0, 0] = torch.cos(x[2])
    Rz[0, 1] = -torch.sin(x[2])
    Rz[1, 0] = torch.sin(x[2])
    Rz[1, 1] = torch.cos(x[2])
    Rz[2, 2] = 1

    t[0, 0] = x[3]
    t[1, 0] = x[4]
    t[2, 0] = x[5]

    # define loss
    pt = (torch.mm(torch.mm(torch.mm(Rx, Ry), Rz), points.to(device)) + t.repeat(1, N)).to(
        device)  # transform pointcloud
    rr = torch.clamp(((width_t / 2 + pt[1, :]) / step_t).long(), 0, height.shape[0] - 1).to(device)
    cc = torch.clamp(((length_t / 2 + pt[0, :]) / step_t).long(), 0, height.shape[1] - 1).to(device)
    min_heights = height[rr, cc].to(device)  # search for corresponding heights
    loss_collision = (min_heights - pt[2, :])
    loss_energy = (pt[2, :]).sum().to(device)
    loss = (loss_collision.sum() + loss_energy)
    if visu:
        draw_bars = True
        plt.figure(1)
        plt.clf()
        ax = plt.axes(projection='3d')
        Z = height.detach().cpu().numpy()
        if draw_bars:
            # ax.bar3d(X.ravel(), Y.ravel(), Z.ravel(), 0.1, 0.1, -0.01, shade=True, alpha=0.5)
            # ax.bar3d(X.ravel(), Y.ravel(), Z.ravel()/2, 0.1, 0.1, Z.ravel(), shade=True, alpha=0.5)
            # norm = colors.Normalize(lamb.min(), lamb.max())
            ax.bar3d(heightmap_grid[0].ravel(), -heightmap_grid[1].ravel(), Z.ravel(), 0.1, 0.1, -0.01, shade=True,
                     alpha=0.5)  # color=cm.jet(norm(lamb.detach().numpy()).ravel()), shade=True, alpha=0.5)
        else:
            surf = ax.plot_surface(X, Y, height.detach().cpu().numpy(), linewidth=0, antialiased=False, alpha=0.5)
        PT = pt.detach().cpu().numpy()
        idx = (min_heights.detach().cpu() - pt[2, :].detach().cpu()).numpy() > 0.01
        ax.scatter(PT[0, :], -PT[1, :], PT[2, :], marker='o', color='b')
        ax.scatter(PT[0, idx], -PT[1, idx], PT[2, idx], s=80, marker='o', color='r')
        ax.plot_wireframe(np.reshape(PT[0], robot_grid[0].shape), np.reshape(-PT[1], robot_grid[0].shape),
                          np.reshape(PT[2], robot_grid[0].shape), color='k')
        plt.pause(0.001)
        plt.draw()
    return loss, loss_energy, loss_collision, pt, min_heights


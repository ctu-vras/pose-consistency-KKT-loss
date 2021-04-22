import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import nnls
import torch.optim as optim
import time
import os
import torch.nn as nn
import network_d2t


class Robot:
    def __init__(self, border_x, border_y, length, width, step, points):
        self.border_x = border_x
        self.border_y = border_y
        self.length = length
        self.width = width
        self.step = step
        self.points = points

def get_robot_model(length = 0.8, width = 0.6, grid_res = 0.1):

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
    Z = Z - 0.07  # Z.mean()
    N = Z.size
    P = np.stack((X.reshape([1, N]).squeeze(), Y.reshape([1, N]).squeeze(), Z.reshape([1, N]).squeeze()), 0)
    P = P - np.tile(P.mean(axis=1).reshape(3,1), (1, P.shape[1]))
    #points = torch.tensor(P, dtype=torch.float32).to(device)

    robot = Robot(torch.tensor(length / 2), torch.tensor(width / 2), torch.tensor(length), torch.tensor(width), torch.tensor(grid_res), torch.tensor(P, dtype=torch.float32))

     ##### Initialize heightmap grid (just for visualization) #####

    heightmap_grid = np.meshgrid(
        np.arange(-robot_grid[0].shape[1] * grid_res / 2 + grid_res / 2, robot_grid[0].shape[1] * grid_res / 2 + grid_res / 2, grid_res),
        np.arange(-robot_grid[0].shape[0] * grid_res / 2 + grid_res / 2, robot_grid[0].shape[0] * grid_res / 2 + grid_res / 2, grid_res))

    return robot, heightmap_grid

class Optimizer_kkt():
    def __init__(self,parameters, device='cpu', epochs = 25, movie=False, output_path = '../data/kkt'):
        self.CREATE_MOVIE = movie
        self.device = device
        self.robot, self.heightmap_grid = get_robot_model()
        self.epochs = epochs
        self.output_path = output_path
        self.optimizer = optim.Adam(parameters, lr=0.0001)


    ##### NEW KKT Loss #####
    def primal_problem(self, x, points, height, visu=False):
        N = points.shape[1]  # number of robot points
        # create rotation matrix and translation vector from variable x

        x_ = torch.cat((x[0:2],x[5].view(1)))

        Rx = torch.zeros(3, 3, dtype=torch.float32)
        Ry = torch.zeros(3, 3, dtype=torch.float32)
        Rz = torch.zeros(3, 3, dtype=torch.float32)
        t = torch.zeros(3, 1, dtype=torch.float32)

        Rx[0, 0] = 1
        Rx[1, 1] = torch.cos(x_[0]).to(self.device)
        Rx[1, 2] = -torch.sin(x_[0]).to(self.device)
        Rx[2, 1] = torch.sin(x_[0]).to(self.device)
        Rx[2, 2] = torch.cos(x_[0]).to(self.device)

        Ry[0, 0] = torch.cos(x_[1]).to(self.device)
        Ry[0, 2] = torch.sin(x_[1]).to(self.device)
        Ry[1, 1] = 1
        Ry[2, 0] = -torch.sin(x_[1]).to(self.device)
        Ry[2, 2] = torch.cos(x_[1]).to(self.device)

        # yaw does not apply
        Rz[0, 0] = torch.cos(x[2])
        Rz[0, 1] = -torch.sin(x[2])
        Rz[1, 0] = torch.sin(x[2])
        Rz[1, 1] = torch.cos(x[2])
        Rz[2, 2] = 1

        t[0, 0] = x[3]
        t[1, 0] = x[4]
        t[2, 0] = x_[2]

        dzda = torch.zeros(3, 3, dtype=torch.float32)

        dzda[:, 0] = torch.autograd.grad(torch.mm(torch.mm(Rx, Ry), Rz)[2, 0], x_, retain_graph=True)[0]
        dzda[:, 1] = torch.autograd.grad(torch.mm(torch.mm(Rx, Ry), Rz)[2, 1], x_, retain_graph=True)[0]
        dzda[:, 2] = torch.autograd.grad(torch.mm(torch.mm(Rx, Ry), Rz)[2, 2], x_, retain_graph=True)[0]


        # transform robot model
        pt = (torch.mm(torch.mm(torch.mm(Rx, Ry), Rz), points) + t.repeat(1, N)).to(self.device)  # transform pointcloud
        #pt = torch.mm(torch.mm(Rx, Ry), points) + t.repeat(1, N) # transform pointcloud

        # estimate heightmap-robot correspondences
        rr = torch.clamp(((self.robot.width / 2 + pt[1, :]) / self.robot.step).long(), 0, height.shape[0] - 1).to(self.device)
        cc = torch.clamp(((self.robot.length / 2 + pt[0, :]) / self.robot.step).long(), 0, height.shape[1] - 1).to(self.device)
        min_heights = height[rr, cc]  # determine closest height for each robot's point pt

        # define primal problem
        constraints = (min_heights - pt[2, :])
        loss_energy = (pt[2, :]).sum().to(self.device)

        if visu:
            draw_bars = True
            #plt.figure(1)
            plt.clf()
            ax = plt.axes(projection='3d')
            Z = height.detach().cpu().numpy()
            if draw_bars:
                # ax.bar3d(X.ravel(), Y.ravel(), Z.ravel(), 0.1, 0.1, -0.01, shade=True, alpha=0.5)
                # ax.bar3d(X.ravel(), Y.ravel(), Z.ravel()/2, 0.1, 0.1, Z.ravel(), shade=True, alpha=0.5)
                # norm = colors.Normalize(lamb.min(), lamb.max())
                ax.bar3d(self.heightmap_grid[0].ravel(), -self.heightmap_grid[1].ravel(), Z.ravel(), 0.1, 0.1, -0.01, shade=True, alpha=0.5)  # color=cm.jet(norm(lamb.detach().numpy()).ravel()), shade=True, alpha=0.5)
            #else:
            #    surf = ax.plot_surface(X, Y, height.detach().cpu().numpy(), linewidth=0, antialiased=False, alpha=0.5)
            PT = pt.detach().cpu().numpy()
            idx = (min_heights.detach().cpu() - pt[2, :].detach().cpu()).numpy() > 0.01
            ax.scatter(PT[0, :], -PT[1, :], PT[2, :], marker='o', color='b')
            ax.scatter(PT[0, idx], -PT[1, idx], PT[2, idx], s=80, marker='o', color='r')
            ax.plot_wireframe(np.reshape(PT[0], height.shape), np.reshape(-PT[1], height.shape),
                              np.reshape(PT[2], height.shape), color='k')
            plt.pause(0.001)
            plt.draw()
        return loss_energy, constraints, x_, dzda



    def loss_kkt(self, x, points, outputs, visu):
        C = 100             # linear barrier penalty
        N = points.shape[1]  # number of robot points
        lamb = torch.tensor(1 * torch.ones(N, 1), requires_grad=True, dtype=torch.float32).to(self.device)


        # get criterion and contraints of the primal problem
        loss_energy, constraints, x_, dzda = self.primal_problem(x, points, outputs.squeeze(), visu)

        # Estimate optimal lambda wrt fixed heights and dual feasibility constraint
        Fgx = torch.autograd.grad(loss_energy, x_, create_graph=True, retain_graph=True)[0]#.clone()
        #Ggx = - dzda @ points    # dzda stands for precomputed gradient of [R(alpha)*p+t]_z wrt alpha
        Ggx = np.matmul(-dzda,points)
        Ggx[2,:] = -1

        g = constraints.detach().cpu().numpy()
        bb = np.concatenate((np.reshape(Fgx.detach().cpu().numpy(), (3, 1)), np.zeros((63, 1)))).squeeze()
        AA = np.concatenate((-np.tile(Ggx.detach().cpu().numpy(), (1, 1)), np.diagflat(np.tile(g, (1, 1)))))  # np.tile(g, (2, 1))
        try:
            s, acc = nnls(AA, bb)
        except:
            print('nnls failed')
            return 
        lamb = torch.tensor(s.reshape(N, 1), requires_grad=False, dtype=torch.float32).to(self.device)

        Lgx = torch.autograd.grad(loss_energy+(lamb.squeeze() * constraints).sum(), x_, create_graph=True, retain_graph=True)[0].detach()

        # Define KKT loss
        LOSS = Lgx.pow(2).sum() + (lamb.squeeze() * constraints).pow(2).sum() + C * torch.relu(constraints).pow(2).sum()


        #print("LOSS-KKT: ", LOSS.detach().cpu().numpy(), " = Stationarity:", Lgx.pow(2).sum().detach().cpu().numpy(), " + Complementary slackness:", (lamb.squeeze() * constraints).pow(2).sum().detach().cpu().numpy(),
        #      " + Primal feasibility:", C * torch.relu(constraints).pow(2).sum().detach().cpu().numpy())

        return LOSS

    def optimize_network(self,  net_d2t, input_w_features, x_in, patches, weights, class_weights,net_t2rpz, idx1,idx2,label_rpz, visu = False):
        # initialization
        points = self.robot.points
        N = points.shape[1] # number of robot points
        x = torch.tensor(x_in, requires_grad=True, dtype=torch.float32).to(self.device)
        #optimizer = optim.Adam([x], lr=0.01)
        #self.optimizer = optim.Adam(net_d2t.parameters(), lr=0.0001)
        Lkkt = np.zeros([1, self.epochs])

        for epoch in range(self.epochs):
            #print("epoch:", epoch)

            # zero the parameter gradients
            self.optimizer.zero_grad()


            # Compute LOSS
            kkt_time_start = time.time()
            LOSS = 0
            # feedforward pass
            relu = nn.ReLU()
            diff_terrain = relu(net_d2t(input_w_features))
            dem = input_w_features[:,0,:,:].unsqueeze(1)
            support_terrain = dem-diff_terrain

            loss_pred_st = network_d2t.weighted_mse_loss(diff_terrain, torch.zeros_like(diff_terrain), weights, class_weights)
            loss_pred_st = loss_pred_st/10#*100*patches.shape[0]
            loss_pred_st.backward(retain_graph=True)
            print(patches.shape[0])
            print(loss_pred_st)


            #output_rpz = net_t2rpz((support_terrain).permute([1, 0, 2, 3]))
            #loss_rpz = torch.sqrt((torch.sum((output_rpz[0, :, idx1, idx2] - label_rpz[0, :, idx1, idx2].to(self.device)) ** 2) / torch.tensor(idx1.size).to(self.device)))
            #loss_rpz.backward(retain_graph=True)
            #print(loss_rpz)


            for k in range(patches.shape[0]):
                # crop terrain under robot
                #print('PATCH', k, ':')
                if visu:
                    plt.figure(k)
                outputs = support_terrain[0, 0, patches[k,0]:patches[k,2], patches[k,1]:patches[k,3]]
                # Compute KKT-loss
                L = self.loss_kkt(x[k,:], points, outputs, visu)
                if(L is None):
                    print('none')
                else:
                    LOSS += L 
            LOSS = LOSS/patches.shape[0]
            LOSS.backward(retain_graph=True)
            print(LOSS)
            print('---')
            #if LOSS.detach().cpu().numpy() > 500:
            #    print('big loss')
            #    print(LOSS.detach().cpu().numpy())
            #    self.optimizer.zero_grad()
            #    return net_d2t, Lkkt
            #print("KKT estimation time :", time.time() - kkt_time_start, "sec")

            Lkkt[0, epoch] = LOSS.detach().cpu().numpy()


            self.optimizer.step()

            self.optimizer.zero_grad()
            if visu:
                if self.CREATE_MOVIE:
                    plt.savefig(self.output_path + '{:04d}_rigid_body'.format(epoch) + '.png')



        if self.CREATE_MOVIE:
            os.system('rm ' + self.output_path + 'output.mp4')
            os.system('ffmpeg -i ' + self.output_path + '%04d_rigid_body.png -c:v libx264 -vf scale=1280:-2 -pix_fmt yuv420p ' + self.output_path + 'output.mp4')
        return net_d2t, Lkkt, loss_pred_st

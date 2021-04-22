import torch
import network_sf2d
import network_d2t
import network_d2rpz
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
import kkt_loss_optimizer



if __name__ == '__main__':
    epochs = 500
    batch_size = 1
    learning_rate = np.ones([epochs, 1])*0.1


    path_to_init_weights = "../weights/network_weights_s2df"

    name = "kkt_z5_st10_all"
    KKT = True

    runtime = strftime("%Y-%m-%d_%H:%M:%S", gmtime())
    output_file = "../data/df2t_net/"+name+"run_"+runtime
    writer = SummaryWriter('../data/df2t_net/tensorboardX/'+name+'run_' + runtime)

    if not os.path.exists(output_file):
        os.makedirs(output_file)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    net = network_sf2d.Net()
    net.load_state_dict(torch.load(path_to_init_weights, map_location=device))
    net.to(device)

    net_d2t = network_d2t.Net()
    #net_d2t.load_state_dict(torch.load(path_to_init_weights, map_location=device))
    net_d2t.to(device)

    net_t2rpz = network_d2rpz.Net()
    net_t2rpz.load_state_dict(torch.load("../weights/network_weights_d2rpz", map_location=device))
    net_t2rpz.to(device)


    dataset_trn = dataset_sf2d.Dataset("../data/sf2d_trn/",augment=False)

    trainloader = torch.utils.data.DataLoader(dataset_trn, batch_size=batch_size, shuffle=True, num_workers=4)

    dataset_val = dataset_sf2d.Dataset("../data/sf2d_val/",augment=False)
    valloader = torch.utils.data.DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=4)


   

    seen = 0
    optimizer_d2t = optim.Adam(net_d2t.parameters(), weight_decay=0.001, lr=0.0001)
    optimizer_kkt = kkt_loss_optimizer.Optimizer_kkt(net_d2t.parameters(), device,5,False)

    regularizer_st = np.ones(500)*10#np.linspace(100,1,500)

    for epoch in range(epochs):  # loop over the dataset multiple times
        print(epoch)
        epoch_loss = 0
        epoch_loss_pred = 0
        epoch_loss_conf = 0
        epoch_loss_rpz = 0
        epoch_loss_st = 0
        for i, data in enumerate(trainloader):
            input = data['input']
            label_dem = data['label']
            input_mask = data['mask']
            weights = data['weights']
            imgs = data['images']
            label_rpz = data['label_rpz']

            T_baselink_zpr = data['T_baselink_zpr'].numpy()[0]
            input, label_dem, input_mask, weights = input.to(device), label_dem.to(device), input_mask.to(device), weights.to(device)

            features = data['features'][:, 0, :, :].to(device)


            pool = nn.MaxPool2d(3, stride=1, padding=1)
            features = pool(features)

            input_w_mask = torch.cat([input,input_mask,features],1)

            #input_w_mask = torch.cat([input,input_mask,features],1)
            input_w_mask = torch.cat([input,input_mask],1)
            #input_w_mask = torch.cat([input,input_mask,data['features'].to(device)],1)

            output_DEM = net(input_w_mask)
            loss_dem, loss_pred, loss_conf,_ = network_sf2d.weighted_mse_loss(output_DEM, label_dem, weights,input)
            #loss_dem.backward(retain_graph=True)
            pool = nn.MaxPool2d(5, stride=1, padding=2)
            features = pool(features)
            dense = torch.cat([output_DEM[:, 0:1, :, :],input_mask,features],1)
            relu = nn.LeakyReLU(0.0001)
            diff_terrain = relu(net_d2t(dense))

            # experiment with penalty for dem change
            class_weights = data['class_weights'].to(device)

            rpz = label_rpz.cpu().detach().numpy()
            # BATCH has to be 1!!!!!!!!!!
            idx_b, idx1, idx2 = np.where(~np.isnan(rpz[:, 2, :, :]))



            if(KKT==True):

                yaw = data['label_yaw'][0][0].detach().cpu().numpy()
                ###### Find patches with steepest pitch
                rpz[np.isnan(rpz)] = 0
                K = np.argsort(-np.sum(np.abs(rpz[idx_b, 0:2, idx1, idx2]), axis=1))  # sort positions according abs(rpz)

                NUM_PATCHES = K.shape[0]#np.min([10, K.shape[0]])  # number of patches from a single DEM to be considered in kkt loss
                patches = np.zeros([NUM_PATCHES, 4], dtype='int')  # corners of under-robot patches in a given DEM
                x_in = torch.zeros([NUM_PATCHES, 6])  # corresponding robot poses (roll, pitch, yaw, t0,t1,t2)

                r_offset = 3
                c_offset = 4
                for k in range(NUM_PATCHES):
                    r1 = r_offset + idx1[K[k]] - 3
                    r2 = r_offset + idx1[K[k]] + 4
                    c1 = c_offset + idx2[K[k]] - 4
                    c2 = c_offset + idx2[K[k]] + 5
                    patches[k, :] = [r1, c1, r2, c2]
                    RPZ = rpz[0, :, idx1[K[k]], idx2[K[k]]]
                    YAW = yaw[idx1[K[k]]+r_offset, idx2[K[k]]+c_offset]
                    x_in[k, :] = torch.tensor([RPZ[0], RPZ[1], YAW, 0, 0, RPZ[2]+ 0.05])

                net_d2t, Lkkt, loss_pred_st = optimizer_kkt.optimize_network(net_d2t, dense, x_in, patches, weights, class_weights, net_t2rpz, idx1,idx2,label_rpz)
                with torch.no_grad():
                    diff_terrain = relu(net_d2t(dense))
                    support_terrain = output_DEM[:, 0, :, :][np.newaxis]-diff_terrain[:, 0, :, :][np.newaxis]
                    output_rpz = net_t2rpz((support_terrain).permute([1, 0, 2, 3]))
                    loss_rpz = torch.sqrt((torch.sum((output_rpz[idx_b,:,idx1,idx2] - label_rpz[idx_b,:,idx1,idx2].to(device)) ** 2) / torch.tensor(idx1.size).to(device)))


            else:

                support_terrain = output_DEM[:, 0, :, :][np.newaxis]-diff_terrain[:, 0, :, :][np.newaxis]

                loss_pred_st = network_d2t.weighted_mse_loss(diff_terrain, torch.zeros_like(diff_terrain), weights,
                                                             class_weights)
                loss_pred_st = loss_pred_st / regularizer_st[epoch]
                loss_pred_st.backward(retain_graph=True)

                # uncomment for augmentation / derotation
                '''
                for r in range(input.shape[0]):
                    support_terrain[:,r,:,:] = support_terrain[:,r,:,:].rot90(-data['rot'].detach().cpu().numpy()[r],dims=[1,2])
                output_rpz = net_t2rpz((support_terrain).permute([1, 0, 2, 3]))
                '''
                output_rpz = net_t2rpz((support_terrain).permute([1, 0, 2, 3]))

                loss_rpz = torch.sqrt((torch.sum((output_rpz[idx_b,:,idx1,idx2] - label_rpz[idx_b,:,idx1,idx2].to(device)) ** 2) / torch.tensor(idx1.size).to(device)))

                loss_rpz.backward(retain_graph=False)

                optimizer_d2t.step()
                optimizer_d2t.zero_grad()


            epoch_loss += loss_dem.item()
            epoch_loss_pred += loss_pred.item()
            epoch_loss_conf += loss_conf.item()
            epoch_loss_st += loss_pred_st.item()
            epoch_loss_rpz += loss_rpz.item()
            #writer.add_scalar('data/loss', loss_dem, seen)
            #writer.add_scalar('data/loss_pred', loss_pred, seen)
            writer.add_scalar('data/loss_pred_st', loss_pred_st, seen)
            #writer.add_scalar('data/loss_conf', loss_conf, seen)
            writer.add_scalar('data/loss_rpz', loss_rpz, seen)

            if i%100==0:
                max_val = 1

                input = input[0, :, :, :]
                input = input.cpu().numpy()
                input[0,0,0] = max_val
                input[0,1,1] = -max_val
                input[input > max_val] = max_val
                input[input < -max_val] = -max_val
                input = input - input.min()
                input = input / input.max()
                writer.add_image('data/Image', input, seen)
                label = label_dem[0, :, :, :]
                label = label.cpu().numpy()
                label[0,0,0] = max_val
                label[0,1,1] = -max_val
                label[label>max_val] = max_val
                label[label<-max_val] = -max_val
                label = label - label.min()
                label = label / label.max()
                writer.add_image('data/Label', label, seen)
                out = output_DEM[0, 0, :, :].detach().cpu().numpy()
                out[0,0] = max_val
                out[1,1] = -max_val
                out[out>max_val] = max_val
                out[out<-max_val] = -max_val
                out = out - out.min()
                out = out / out.max()
                writer.add_image('data/Output', out[np.newaxis], seen)
                out = np.rot90(support_terrain[0, 0, :, :].detach().cpu().numpy(),data['rot'].detach().cpu().numpy()[0])
                out[0,0] = max_val
                out[1,1] = -max_val
                out[out>max_val] = max_val
                out[out<-max_val] = -max_val
                out = out - out.min()
                out = out / out.max()
                writer.add_image('data/Output_support', out[np.newaxis], seen)
                out = output_DEM[0, 1, :, :].detach().cpu().numpy()
                out = 1 / (1 + np.exp(-out))
                out[0,0] = max_val
                out[1,1] = -max_val
                out[out>max_val] = max_val
                out[out<-max_val] = -max_val
                out = out - out.min()
                out = out / out.max()
                #writer.add_image('data/Output_conf', out[np.newaxis], seen)
            seen += 1

        #writer.add_scalar('data/epoch_loss', (epoch_loss/dataset_trn.size)*batch_size, epoch)
        writer.add_scalar('data/epoch_loss_pred', (epoch_loss_pred/dataset_trn.size)*batch_size, epoch)
        writer.add_scalar('data/epoch_loss_support', (epoch_loss_st/dataset_trn.size)*batch_size, epoch)

        #writer.add_scalar('data/epoch_loss_conf', epoch_loss_conf/dataset_trn.size, epoch)
        writer.add_scalar('data/epoch_loss_rpz', (epoch_loss_rpz/dataset_trn.size)*batch_size, epoch)

        print(epoch_loss)
        epoch_loss = 0
        epoch_val_loss = 0
        epoch_val_loss_pred = 0
        epoch_val_loss_conf = 0
        epoch_val_loss_pred_st = 0
        epoch_val_loss_rpz = 0
        torch.save(net_d2t.state_dict(), output_file+'/net_epoch_{:04}'.format(epoch))


        for i, data in enumerate(valloader):
            input = data['input']
            label_dem = data['label']
            input_mask = data['mask']
            weights = data['weights']
            imgs = data['images']
            label_rpz = data['label_rpz']

            T_baselink_zpr = data['T_baselink_zpr'].numpy()[0]
            input, label_dem, input_mask, weights = input.to(device), label_dem.to(device), input_mask.to(
                device), weights.to(device)

            features = data['features'][:, 0, :, :].to(device)
            #input_w_mask = torch.cat([input,input_mask,features],1)
            input_w_mask = torch.cat([input,input_mask],1)
            
            pool = nn.MaxPool2d(3, stride=1, padding=1)
            features = pool(features)

            #input_w_mask = torch.cat([input,input_mask,features],1)
            input_w_mask = torch.cat([input, input_mask], 1)
            with torch.no_grad():
                output_DEM = net(input_w_mask)
                loss, loss_pred, loss_conf,_ = network_sf2d.weighted_mse_loss(output_DEM, label_dem, weights,input)
                pool = nn.MaxPool2d(5, stride=1, padding=2)
                features = pool(features)

                dense = torch.cat([output_DEM[:, 0:1, :, :],input_mask,features],1)

                relu = nn.LeakyReLU(0.0001)
                diff_terrain = relu(net_d2t(dense))

                support_terrain = output_DEM[:, 0, :, :][np.newaxis] - diff_terrain[:, 0, :, :][np.newaxis]
                class_weights = data['class_weights'].to(device)

                loss_pred_st = network_d2t.weighted_mse_loss(diff_terrain,torch.zeros_like(diff_terrain),
                                                                                     weights,class_weights)

                loss_pred_st = loss_pred_st/regularizer_st[epoch]

                for r in range(input.shape[0]):
                    support_terrain[:, r, :, :] = support_terrain[:, r, :, :].rot90(
                        -data['rot'].detach().cpu().numpy()[r], dims=[1, 2])
                output_rpz = net_t2rpz((support_terrain).permute([1, 0, 2, 3]))

                #label_rpz[torch.isnan(label_rpz)] = 0
                #rpz = label_rpz.cpu().detach().numpy()
                #idx_b, idx1, idx2 = np.where((rpz[:, 2, :, :]) != 0)

                rpz = label_rpz.cpu().detach().numpy()
                idx_b, idx1, idx2 = np.where(~np.isnan(rpz[:, 2, :, :]))

                loss_rpz = torch.sqrt((torch.sum((output_rpz[idx_b, :, idx1, idx2] - label_rpz[idx_b, :, idx1, idx2].to(
                    device)) ** 2) / torch.tensor(idx1.size).to(device)))


                #output_rpz = net_t2rpz((output_DEM[:, 0, :, :][np.newaxis]-support_terrain[:, 0, :, :][np.newaxis]).permute([1, 0, 2, 3]))


            epoch_val_loss += loss
            epoch_val_loss_pred += loss_pred
            epoch_val_loss_conf += loss_conf
            epoch_val_loss_pred_st += loss_pred_st
            epoch_val_loss_rpz += loss_rpz

            if i%10==0:
                input = input[0, :, :, :]
                input = input.cpu().numpy()
                # input = input.transpose((1, 2, 0))
                input[0,0,0] = max_val
                input[0,1,1] = -max_val
                input[input > max_val] = max_val
                input[input < -max_val] = -max_val
                input = input - input.min()
                input = input / input.max()
                writer.add_image('data/val_Image', input, seen)
                label = label_dem[0, :, :, :]
                label = label.cpu().numpy()
                label[0,0,0] = max_val
                label[0,1,1] = -max_val
                label[label>max_val] = max_val
                label[label<-max_val] = -max_val
                label = label - label.min()
                label = label / label.max()
                writer.add_image('data/val_Label', label, seen)
                # out = torch.sigmoid(output[0,:, :, :].clone())
                out = output_DEM[0, 0, :, :].detach().cpu().numpy()
                out[0,0] = max_val
                out[1,1] = -max_val
                out[out>max_val] = max_val
                out[out<-max_val] = -max_val
                out = out - out.min()
                out = out / out.max()
                writer.add_image('data/val_Output', out[np.newaxis], seen)
                out = np.rot90(support_terrain[0, 0, :, :].detach().cpu().numpy(),data['rot'].detach().cpu().numpy()[0])
                out[0,0] = max_val
                out[1,1] = -max_val
                out[out>max_val] = max_val
                out[out<-max_val] = -max_val
                out = out - out.min()
                out = out / out.max()
                writer.add_image('data/val_Output_support', out[np.newaxis], seen)

                out = output_DEM[0, 1, :, :].detach().cpu().numpy()
                out = 1 / (1 + np.exp(-out))
                out[0,0] = max_val
                out[1,1] = -max_val
                out[out>max_val] = max_val
                out[out<-max_val] = -max_val
                out = out - out.min()
                out = out / out.max()
                #writer.add_image('data/val_Output_conf', out[np.newaxis], seen)
            seen += 1

        #writer.add_scalar('data/epoch_val_loss', (epoch_val_loss / dataset_val.size)*batch_size, epoch)
        writer.add_scalar('data/epoch_val_loss_pred', (epoch_val_loss_pred / dataset_val.size)*batch_size, epoch)
        writer.add_scalar('data/epoch_val_loss_st', (epoch_val_loss_pred_st / dataset_val.size)*batch_size, epoch)
        writer.add_scalar('data/epoch_val_loss_rpz', (epoch_val_loss_rpz / dataset_val.size)*batch_size, epoch)



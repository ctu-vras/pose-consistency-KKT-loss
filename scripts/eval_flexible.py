import numpy as np
import matplotlib.pyplot as plt
import torch
import network_sf2d
import network_d2t
import network_d2rpz
import glob
import torch.nn as nn
import dataset_sf2d
from scipy import ndimage, interpolate
import os


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
   # device = "cpu"
    model_s2d = network_sf2d.Net()
    model_s2d.load_state_dict(torch.load("../weights/network_weights_s2df", map_location=device))
    model_s2d.to(device)

    model_d2t = network_d2t.Net()
    model_d2t.load_state_dict(torch.load("../weights/network_weights_df2t_rpz", map_location=device))
    model_d2t.to(device)

    model_d2t_kkt = network_d2t.Net()
    model_d2t_kkt.load_state_dict(torch.load("../weights/network_weights_df2t_kkt", map_location=device))
    model_d2t_kkt.to(device)

    model_d2rpz = network_d2rpz.Net()
    model_d2rpz.load_state_dict(torch.load("../weights/network_weights_d2rpz", map_location=device))
    model_d2rpz.to(device)

    loss_dem = 0
    loss_st = 0

    dataset_val = dataset_sf2d.Dataset("../../data/sf2d_tst/",augment=False)
    valloader = torch.utils.data.DataLoader(dataset_val, batch_size=1, shuffle=False, num_workers=0)

    with torch.no_grad():

        for t in [0,1]:
            if t==0:
                onsparse = True
            else:
                onsparse = False
            dataset_val.set_onsparse(onsparse)
            loss_roll_s2rpz = []
            loss_roll_s2d = []
            loss_roll = []
            loss_roll_kkt = []

            loss_pitch_s2rpz = []
            loss_pitch_s2d = []
            loss_pitch = []
            loss_pitch_kkt = []

            loss_z_s2rpz = []
            loss_z_s2d = []
            loss_z = []
            loss_z_kkt = []

            loss_abs_in = []
            loss_abs = []
            loss_abs_t = []
            loss_abs_k = []
            loss_dem_in = 0
            loss_dem = 0
            loss_dem_t = 0
            loss_dem_k = 0


            for i, data in enumerate(valloader):
                input = data['input']
                label_dem = data['label']
                input_mask = data['mask']
                weights = data['weights']
                label_rpz = data['label_rpz'].to(device)
                label_synth = data['label_synth'].to(device) #+ 0.05
                synth_mask = (~torch.isnan(label_synth)).float()
                label_synth[torch.isnan(label_synth)] = 0

                if onsparse:
                    label_rpz[0, :  , input_mask[0, 0, 3:-3, 4:-4] == 0] = np.nan

                T_baselink_zpr = data['T_baselink_zpr'].numpy()[0]
                input, label_dem, input_mask, weights = input.to(device), label_dem.to(device), input_mask.to(device), weights.to(device)

                #label_dem = torch.zeros([1,1,256,256])

                features = data['features'][:, 0, :, :].to(device)

                pool = nn.MaxPool2d(3, stride=1, padding=1)
                features = pool(features)

                input_w_mask = torch.cat([input, input_mask], 1)

                to_interp = input.cpu().detach().numpy()[0,0,:,:]
                to_interp[input_mask.cpu().detach().numpy()[0,0,:,:]==0] = np.nan
                x = np.arange(0, to_interp.shape[1])
                y = np.arange(0, to_interp.shape[0])
                xx, yy = np.meshgrid(x, y)
                to_interp = np.ma.masked_invalid(to_interp)
                x1 = xx[~to_interp.mask]
                y1 = yy[~to_interp.mask]
                newarr = to_interp[~to_interp.mask]
                GD1 = interpolate.griddata((x1, y1), newarr.ravel(), (xx, yy), method='linear')

                to_interp = GD1
                #to_interp[input_mask.cpu().detach().numpy()[0, 0, :, :] == 0] = np.nan
                x = np.arange(0, to_interp.shape[1])
                y = np.arange(0, to_interp.shape[0])
                xx, yy = np.meshgrid(x, y)
                to_interp = np.ma.masked_invalid(to_interp)
                x1 = xx[~to_interp.mask]
                y1 = yy[~to_interp.mask]
                newarr = to_interp[~to_interp.mask]
                GD1 = interpolate.griddata((x1, y1), newarr.ravel(), (xx, yy), method='nearest')


                input_interp = torch.from_numpy(GD1.astype(np.float32)).to(device).unsqueeze(0).unsqueeze(0)

                output = input_interp
                target = label_dem
                weight = weights
                pred = (output[:, 0, :, :][np.newaxis]).permute([1, 0, 2, 3])
                loss_prediction = torch.sqrt((torch.sum(weight * (pred - target) ** 2) / weight.sum()))
                loss_abs_in.append(abs((weight * (pred - target))[weight == 1].cpu().detach().numpy()))


                output_DEM = model_s2d(input_w_mask)
                relu = nn.LeakyReLU(0.0001)

                output = output_DEM
                target = label_dem
                weight = weights
                pred = (output[:, 0, :, :][np.newaxis]).permute([1, 0, 2, 3])
                loss_prediction = torch.sqrt((torch.sum(weight * (pred - target) ** 2) / weight.sum()))
                loss_abs.append(abs((weight * (pred - target))[weight == 1].cpu().detach().numpy()))

                dense = torch.cat([output_DEM[:, 0:1, :, :], input_mask, features], 1)
                diff_terrain = relu(model_d2t(dense))
                support_terrain = output_DEM[:, 0, :, :][np.newaxis] - diff_terrain[:, 0, :, :][np.newaxis]

                output = support_terrain
                target = label_dem
                weight = weights
                pred = (output[:, 0, :, :][np.newaxis]).permute([1, 0, 2, 3])
                loss_prediction = torch.sqrt((torch.sum(weight * (pred - target) ** 2) / weight.sum()))
                loss_abs_t.append(abs((weight * (pred - target))[weight == 1].cpu().detach().numpy()))

                diff_terrain = relu(model_d2t_kkt(dense))
                support_terrain_kkt = output_DEM[:, 0, :, :][np.newaxis] - diff_terrain[:, 0, :, :][np.newaxis]

                output = support_terrain_kkt
                target = label_dem
                weight = weights
                pred = (output[:, 0, :, :][np.newaxis]).permute([1, 0, 2, 3])
                loss_prediction = torch.sqrt((torch.sum(weight * (pred - target) ** 2) / weight.sum()))
                loss_abs_k.append(abs((weight * (pred - target))[weight == 1].cpu().detach().numpy()))

                output_rpz_s2rpz = model_d2rpz(input_interp)
                output_rpz_s2d = model_d2rpz(output_DEM[:, 0, :, :][np.newaxis])
                output_rpz = model_d2rpz(support_terrain)
                output_rpz_kkt = model_d2rpz(support_terrain_kkt)

                loss_roll_s2rpz.append((label_rpz[0,0,~torch.isnan(label_rpz[0,0,:,:])] - output_rpz_s2rpz[0,0,~torch.isnan(label_rpz[0,0,:,:])]).detach().cpu().numpy() ** 2)
                loss_roll_s2d.append((label_rpz[0,0,~torch.isnan(label_rpz[0,0,:,:])] - output_rpz_s2d[0,0,~torch.isnan(label_rpz[0,0,:,:])]).detach().cpu().numpy() ** 2)
                loss_roll.append((label_rpz[0,0,~torch.isnan(label_rpz[0,0,:,:])] - output_rpz[0,0,~torch.isnan(label_rpz[0,0,:,:])]).detach().cpu().numpy() ** 2)
                loss_roll_kkt.append((label_rpz[0,0,~torch.isnan(label_rpz[0,0,:,:])] - output_rpz_kkt[0,0,~torch.isnan(label_rpz[0,0,:,:])]).detach().cpu().numpy() ** 2)

                loss_pitch_s2rpz.append((label_rpz[0,1,~torch.isnan(label_rpz[0,1,:,:])] - output_rpz_s2rpz[0,1,~torch.isnan(label_rpz[0,1,:,:])]).detach().cpu().numpy() ** 2)
                loss_pitch_s2d.append((label_rpz[0,1,~torch.isnan(label_rpz[0,1,:,:])] - output_rpz_s2d[0,1,~torch.isnan(label_rpz[0,1,:,:])]).detach().cpu().numpy() ** 2)
                loss_pitch.append((label_rpz[0,1,~torch.isnan(label_rpz[0,1,:,:])] - output_rpz[0,1,~torch.isnan(label_rpz[0,1,:,:])]).detach().cpu().numpy() ** 2)
                loss_pitch_kkt.append((label_rpz[0,1,~torch.isnan(label_rpz[0,1,:,:])] - output_rpz_kkt[0,1,~torch.isnan(label_rpz[0,1,:,:])]).detach().cpu().numpy() ** 2)

                loss_z_s2rpz.append((label_rpz[0,2,~torch.isnan(label_rpz[0,2,:,:])] - output_rpz_s2rpz[0,2,~torch.isnan(label_rpz[0,2,:,:])]).detach().cpu().numpy() ** 2)
                loss_z_s2d.append((label_rpz[0,2,~torch.isnan(label_rpz[0,2,:,:])] - output_rpz_s2d[0,2,~torch.isnan(label_rpz[0,2,:,:])]).detach().cpu().numpy() ** 2)
                loss_z.append((label_rpz[0,2,~torch.isnan(label_rpz[0,2,:,:])] - output_rpz[0,2,~torch.isnan(label_rpz[0,2,:,:])]).detach().cpu().numpy() ** 2)
                loss_z_kkt.append((label_rpz[0,2,~torch.isnan(label_rpz[0,2,:,:])] - output_rpz_kkt[0,2,~torch.isnan(label_rpz[0,2,:,:])]).detach().cpu().numpy() ** 2)

                _, loss_dem_in_one, _ ,_= network_sf2d.weighted_mse_loss(torch.cat([input_interp, input_mask], 1),label_synth,synth_mask,input)
                _, loss_dem_one, _,_ = network_sf2d.weighted_mse_loss(output_DEM,label_synth,synth_mask,input)
                _, loss_dem_t_one, _,_ = network_sf2d.weighted_mse_loss(torch.cat([support_terrain, output_DEM[:,1,:,:].unsqueeze(0)], 1),label_synth,synth_mask,input)
                _, loss_dem_k_one, _,_ = network_sf2d.weighted_mse_loss(torch.cat([support_terrain_kkt, output_DEM[:,1,:,:].unsqueeze(0)], 1),label_synth,synth_mask,input)

                loss_dem_in += loss_dem_in_one.cpu().detach().numpy()
                loss_dem += loss_dem_one.cpu().detach().numpy()
                loss_dem_t += loss_dem_t_one.cpu().detach().numpy()
                loss_dem_k += loss_dem_k_one.cpu().detach().numpy()

                data_path = "../data/tomas_pose_all/"
                try:
                    os.listdir(data_path)
                except:
                    os.makedirs(data_path)
                label_name = '{:03}'.format(len(os.listdir(data_path))) + '_label'

                np.savez_compressed(data_path + label_name, input=data['input'].cpu().detach().numpy()[0][0][3:-3,4:-4],
                                    label_rpz=data['label_rpz'].cpu().detach().numpy()[0],
                                    dem_interp = input_interp.detach().cpu().numpy()[0,0][3:-3,4:-4],
                                    dem_s2d = output_DEM.detach().cpu().numpy()[0,0][3:-3,4:-4],
                                    dem_s2d2rpz = support_terrain.detach().cpu().numpy()[0,0][3:-3,4:-4],
                                    dem_s2d2kkt = support_terrain_kkt.detach().cpu().numpy()[0, 0][3:-3,4:-4])

                                    #_,loss_dem_one,_,_ = network_sf2d.weighted_mse_loss(dense,label_dem,weights,input)


                #_,loss_dem_one,_,_ = network_sf2d.weighted_mse_loss(dense,label_dem,weights,input)
                #loss_st_one = network_d2t.weighted_mse_loss(dense-support_terrain,label_dem,weights,torch.ones(input.shape).to(device))

                #loss_dem += loss_dem_one.cpu().detach().numpy()
                #loss_st += loss_st_one.cpu().detach().numpy()

            #loss_abs = np.concatenate(loss_abs)
            num_labels = dataset_val.size
            print(["error",'  li  ','  r  ','  r+p  ','  r+kkt '])
            print(['roll ','{:.3}'.format(np.sqrt(np.mean(np.concatenate(loss_roll_s2rpz)))),'{:.3}'.format(np.sqrt(np.mean(np.concatenate(loss_roll_s2d)))),'{:.3}'.format(np.sqrt(np.mean(np.concatenate(loss_roll)))),'{:.3}'.format(np.sqrt(np.mean(np.concatenate(loss_roll_kkt))))])
            print(['pitch','{:.3}'.format(np.sqrt(np.mean(np.concatenate(loss_pitch_s2rpz)))),'{:.3}'.format(np.sqrt(np.mean(np.concatenate(loss_pitch_s2d)))),'{:.3}'.format(np.sqrt(np.mean(np.concatenate(loss_pitch)))),'{:.3}'.format(np.sqrt(np.mean(np.concatenate(loss_pitch_kkt))))])
            print(['z    ','{:.3}'.format(np.sqrt(np.mean(np.concatenate(loss_z_s2rpz)))),'{:.3}'.format(np.sqrt(np.mean(np.concatenate(loss_z_s2d)))),'{:.3}'.format(np.sqrt(np.mean(np.concatenate(loss_z)))),'{:.3}'.format(np.sqrt(np.mean(np.concatenate(loss_z_kkt))))])
            print(['dem  ',' {:.3}'.format(loss_dem_in/num_labels), '{:.3}'.format(loss_dem/num_labels), '{:.3}'.format(loss_dem_t/num_labels), '{:.3}'.format(loss_dem_k/num_labels)])

            print('---------------------')
            #print(['dem  ','{:.3}'.format(loss_dem/len(dataset_val))])
            #print(['st  ','{:.3}'.format(loss_st/len(dataset_val))])















import numpy as np
import matplotlib.pyplot as plt
import torch
import network_s2d
import network_d2rpz
from scipy import ndimage, interpolate
import glob

if __name__ == '__main__':
    on_sparse = True

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model_s2d = network_s2d.Net()
    model_s2d.load_state_dict(torch.load("../weights/network_weights_s2d", map_location=device))
    model_s2d.to(device)

    model_s2d_rpzt = network_s2d.Net()
    model_s2d_rpzt.load_state_dict(torch.load("../weights/network_weights_s2d2rpz", map_location=device))
    model_s2d_rpzt.to(device)

    model_s2d_rpzk = network_s2d.Net()
    model_s2d_rpzk.load_state_dict(torch.load("../weights/network_weights_s2d_kkt", map_location=device))
    model_s2d_rpzk.to(device)


    model_d2rpz = network_d2rpz.Net()
    model_d2rpz.load_state_dict(torch.load("../weights/network_weights_d2rpz", map_location=device))
    model_d2rpz.to(device)




    data_path = 's2d_tst'



    num_labels = len(glob.glob('../../data/'+data_path + '/*label.npz'))
    for t in [0,1]:
        if t==0:
            onsparse=True
        else:
            onsparse=False
        loss_roll_s2 = []
        loss_pitch_s2 = []
        loss_z_s2 = []
        loss_roll = []
        loss_pitch = []
        loss_roll_t = []
        loss_pitch_t = []
        loss_roll_k = []
        loss_pitch_k = []
        loss_z = []
        loss_z_t = []
        loss_z_k = []
        loss_abs = []
        loss_abs_t = []
        loss_abs_k = []
        loss_dem_interp = 0

        loss_dem = 0
        loss_dem_t = 0
        loss_dem_k = 0
        #num_labels = 17
        for i in range(0,num_labels,1):
            label = np.load('../../data/'+data_path+'/000{:03}'.format(i)+'_label.npz')

            yaw_mask = abs(label['yaw_label'])<(np.pi/8)

            dem_label = label['label']

            yaw_indexes = np.where(yaw_mask)  ## here are indexes of known roll, pitch, z

            roll_label = label['roll_label']
            pitch_label = label['pitch_label']
            z_label = label['z_baselink_label']
            roll_label[~yaw_mask] = np.nan
            pitch_label[~yaw_mask] = np.nan
            z_label[~yaw_mask] = np.nan


            vis = label['visible_mask']
            vis[~np.isnan(vis)] = 0
            vis[np.isnan(vis)] = 1
            bw_dist = ndimage.distance_transform_edt(vis)
            if onsparse:
                #pos_mask = ~np.isnan(yaw_mask) & (bw_dist < 10)  & ~np.isnan(label['input']) # distance in decimeters from visible
                pos_mask = ~np.isnan(yaw_mask) & (bw_dist < 10)  & (ndimage.distance_transform_edt(~np.isnan(label['input'])) > 4)
            else:
                pos_mask = ~np.isnan(yaw_mask) & (bw_dist < 10) 
            pos_indexes = np.where(pos_mask) 

            roll_label[~pos_mask] = np.nan
            pitch_label[~pos_mask] = np.nan
            z_label[~pos_mask] = np.nan

            rpz = np.stack([roll_label,pitch_label,z_label])
            mask = torch.from_numpy((~np.isnan(label['input'])).astype(np.float32)).to(device).unsqueeze(0).unsqueeze(0)
            input = torch.from_numpy(label['input'].astype(np.float32)).to(device).unsqueeze(0).unsqueeze(0)

            dem_weights = ~np.isnan(dem_label) & (bw_dist < 10)  # distance in decimeters from visible

            input[torch.isnan(input)] = 0
            input_w_mask = torch.cat([input, mask], 1)

            to_interp = label['input'].astype(np.float32)
            x = np.arange(0, to_interp.shape[1])
            y = np.arange(0, to_interp.shape[0])
            xx, yy = np.meshgrid(x, y)
            to_interp = np.ma.masked_invalid(to_interp)
            x1 = xx[~to_interp.mask]
            y1 = yy[~to_interp.mask]
            newarr = to_interp[~to_interp.mask]
            GD1 = interpolate.griddata((x1, y1), newarr.ravel(),(xx, yy),method='linear')
            to_interp = GD1
            x = np.arange(0, to_interp.shape[1])
            y = np.arange(0, to_interp.shape[0])
            xx, yy = np.meshgrid(x, y)
            to_interp = np.ma.masked_invalid(to_interp)
            x1 = xx[~to_interp.mask]
            y1 = yy[~to_interp.mask]
            newarr = to_interp[~to_interp.mask]
            GD1 = interpolate.griddata((x1, y1), newarr.ravel(), (xx, yy), method='nearest')



            input_interp = torch.from_numpy(GD1.astype(np.float32)).to(device).unsqueeze(0).unsqueeze(0)
            #interpolate.interp2d(256, 256, label['input'].astype(np.float32), kind='linear')
            output_s2rpz = model_d2rpz(input_interp).permute([1, 0, 2, 3])

            #output_s2rpz = model_d2rpz(input_w_mask[:, 0, :, :][np.newaxis]).permute([1, 0, 2, 3])
            output_s2rpz = output_s2rpz.detach().cpu().numpy()
            output_s2rpz = np.pad(output_s2rpz[:, 0, :, :], [(0, 0), (3, 3), (4, 4)], mode='constant')

            dense = model_s2d(input_w_mask)
            output_rpz = model_d2rpz(dense[:, 0, :, :][np.newaxis]).permute([1, 0, 2, 3])
            output_rpz = output_rpz.detach().cpu().numpy()
            output_rpz = np.pad(output_rpz[:, 0, :, :], [(0, 0), (3, 3), (4, 4)], mode='constant')

            dense_rpzt = model_s2d_rpzt(input_w_mask)
            output_rpzt = model_d2rpz(dense_rpzt[:, 0, :, :][np.newaxis]).permute([1, 0, 2, 3])
            output_rpzt = output_rpzt.detach().cpu().numpy()
            output_rpzt = np.pad(output_rpzt[:, 0, :, :], [(0, 0), (3, 3), (4, 4)], mode='constant')

            dense_rpzk = model_s2d_rpzk(input_w_mask)
            output_rpzk = model_d2rpz(dense_rpzk[:, 0, :, :][np.newaxis]).permute([1, 0, 2, 3])
            output_rpzk = output_rpzk.detach().cpu().numpy()
            output_rpzk = np.pad(output_rpzk[:, 0, :, :], [(0, 0), (3, 3), (4, 4)], mode='constant')

            deg_th = 0
            rpz_mask = (abs(rpz[0,:,:]) >deg_th) | (abs(rpz[1,:,:]) >deg_th)
            if (np.sum(yaw_mask & pos_mask)>0)&(np.sum(np.isnan(rpz[0, yaw_mask&pos_mask].T))==0):
                loss_roll_s2.append((rpz[0, yaw_mask&pos_mask&rpz_mask].T-output_s2rpz[0,yaw_mask&pos_mask&rpz_mask].T)**2)
                loss_pitch_s2.append((rpz[1, yaw_mask&pos_mask&rpz_mask].T-output_s2rpz[1,yaw_mask&pos_mask&rpz_mask].T)**2)
                loss_roll.append((rpz[0, yaw_mask&pos_mask&rpz_mask].T-output_rpz[0,yaw_mask&pos_mask&rpz_mask].T)**2)
                loss_pitch.append((rpz[1, yaw_mask&pos_mask&rpz_mask].T-output_rpz[1,yaw_mask&pos_mask&rpz_mask].T)**2)
                loss_roll_t.append((rpz[0, yaw_mask&pos_mask&rpz_mask].T-output_rpzt[0,yaw_mask&pos_mask&rpz_mask].T)**2)
                loss_pitch_t.append((rpz[1, yaw_mask&pos_mask&rpz_mask].T-output_rpzt[1,yaw_mask&pos_mask&rpz_mask].T)**2)
                loss_roll_k.append((rpz[0, yaw_mask&pos_mask&rpz_mask].T-output_rpzk[0,yaw_mask&pos_mask&rpz_mask].T)**2)
                loss_pitch_k.append((rpz[1, yaw_mask&pos_mask&rpz_mask].T-output_rpzk[1,yaw_mask&pos_mask&rpz_mask].T)**2)
                loss_z_s2.append((rpz[2, yaw_mask & pos_mask].T - output_s2rpz[2, yaw_mask & pos_mask].T) ** 2)
                loss_z.append((rpz[2, yaw_mask & pos_mask].T - output_rpz[2, yaw_mask & pos_mask ].T) ** 2)
                loss_z_t.append((rpz[2, yaw_mask & pos_mask].T - output_rpzt[2, yaw_mask & pos_mask].T) ** 2)
                loss_z_k.append((rpz[2, yaw_mask & pos_mask].T - output_rpzk[2, yaw_mask & pos_mask].T) ** 2)                
    
            dem_label[np.isnan(dem_label)] = 0
            dem_weights[(dem_label>0.3)] = 0
          
            output = dense
            target = torch.from_numpy(dem_label.astype(np.float32)).to(device).unsqueeze(0).unsqueeze(0)
            dem_weights_rpz = dem_weights.copy()
            weight = torch.from_numpy(dem_weights_rpz.astype(np.float32)).to(device).unsqueeze(0).unsqueeze(0)
            pred = (output[:, 0, :, :][np.newaxis]).permute([1, 0, 2, 3])
            loss_prediction = torch.sqrt((torch.sum(weight * (pred - target) ** 2) / weight.sum()))
            loss_abs.append(abs((weight * (pred - target))[weight == 1].cpu().detach().numpy()))

            output = dense_rpzt
            dem_weights_rpzt = dem_weights.copy()
            weight = torch.from_numpy(dem_weights_rpzt.astype(np.float32)).to(device).unsqueeze(0).unsqueeze(0)
            pred = (output[:, 0, :, :][np.newaxis]).permute([1, 0, 2, 3])
            loss_prediction = torch.sqrt((torch.sum(weight * (pred - target) ** 2) / weight.sum()))
            loss_abs_t.append(abs((weight * (pred - target))[weight == 1].cpu().detach().numpy()))

            output = dense_rpzk
            dem_weights_rpzk = dem_weights.copy()
            weight = torch.from_numpy(dem_weights_rpzk.astype(np.float32)).to(device).unsqueeze(0).unsqueeze(0)
            pred = (output[:, 0, :, :][np.newaxis]).permute([1, 0, 2, 3])
            loss_prediction = torch.sqrt((torch.sum(weight * (pred - target) ** 2) / weight.sum()))
            loss_abs_k.append(abs((weight * (pred - target))[weight == 1].cpu().detach().numpy()))

            _,loss_dem_interp_one,_ = network_s2d.weighted_mse_loss(torch.cat([input_interp, mask], 1),torch.from_numpy(dem_label.astype(np.float32)).to(device).unsqueeze(0).unsqueeze(0),torch.from_numpy(dem_weights.astype(np.float32)).to(device).unsqueeze(0).unsqueeze(0))
            _,loss_dem_one,_ = network_s2d.weighted_mse_loss(dense,torch.from_numpy(dem_label.astype(np.float32)).to(device).unsqueeze(0).unsqueeze(0),torch.from_numpy(dem_weights.astype(np.float32)).to(device).unsqueeze(0).unsqueeze(0))
            _,loss_dem_t_one,_= network_s2d.weighted_mse_loss(dense_rpzt,torch.from_numpy(dem_label.astype(np.float32)).to(device).unsqueeze(0).unsqueeze(0),torch.from_numpy(dem_weights.astype(np.float32)).to(device).unsqueeze(0).unsqueeze(0))
            _,loss_dem_k_one,_= network_s2d.weighted_mse_loss(dense_rpzk,torch.from_numpy(dem_label.astype(np.float32)).to(device).unsqueeze(0).unsqueeze(0),torch.from_numpy(dem_weights.astype(np.float32)).to(device).unsqueeze(0).unsqueeze(0))

            loss_dem_interp += loss_dem_interp_one.cpu().detach().numpy()

            loss_dem += loss_dem_one.cpu().detach().numpy()
            loss_dem_t += loss_dem_t_one.cpu().detach().numpy()
            loss_dem_k += loss_dem_k_one.cpu().detach().numpy()        
        loss_abs = np.concatenate(loss_abs)
        loss_abs_t = np.concatenate(loss_abs_t)
        loss_abs_k = np.concatenate(loss_abs_k)

        print(["error", '  li  ', '  r  ', '  r+p  ', '  r+kkt '])

        if onsparse==False:
            print(['roll ','{:.3}'.format(np.sqrt(np.mean(np.concatenate(loss_roll_s2)))), '{:.3}'.format(np.sqrt(np.mean(np.concatenate(loss_roll)))), '{:.3}'.format(np.sqrt(np.mean(np.concatenate(loss_roll_t)))), '{:.3}'.format(np.sqrt(np.mean(np.concatenate(loss_roll_k))))])
            print(['pitch','{:.3}'.format(np.sqrt(np.mean(np.concatenate(loss_pitch_s2)))), '{:.3}'.format(np.sqrt(np.mean(np.concatenate(loss_pitch)))), '{:.3}'.format(np.sqrt(np.mean(np.concatenate(loss_pitch_t)))), '{:.3}'.format(np.sqrt(np.mean(np.concatenate(loss_pitch_k))))])
            print(['z    ','{:.3}'.format(np.sqrt(np.mean(np.concatenate(loss_z_s2)))), '{:.3}'.format(np.sqrt(np.mean(np.concatenate(loss_z)))), '{:.3}'.format(np.sqrt(np.mean(np.concatenate(loss_z_t)))), '{:.3}'.format(np.sqrt(np.mean(np.concatenate(loss_z_k))))])
            print(['dem   ','{:.3}'.format(loss_dem_interp/num_labels), '{:.3}'.format(loss_dem/num_labels), '{:.3}'.format(loss_dem_t/num_labels), '{:.3}'.format(loss_dem_k/num_labels)])
        else:
            print(['roll ','{:.3}'.format(np.sqrt(np.mean(np.concatenate(loss_roll_s2)))),'{:.3}'.format(np.sqrt(np.mean(np.concatenate(loss_roll)))), '{:.3}'.format(np.sqrt(np.mean(np.concatenate(loss_roll_t)))), '{:.3}'.format(np.sqrt(np.mean(np.concatenate(loss_roll_k))))])
            print(['pitch','{:.3}'.format(np.sqrt(np.mean(np.concatenate(loss_pitch_s2)))),'{:.3}'.format(np.sqrt(np.mean(np.concatenate(loss_pitch)))), '{:.3}'.format(np.sqrt(np.mean(np.concatenate(loss_pitch_t)))), '{:.3}'.format(np.sqrt(np.mean(np.concatenate(loss_pitch_k))))])
            print(['z    ','{:.3}'.format(np.sqrt(np.mean(np.concatenate(loss_z_s2)))),'{:.3}'.format(np.sqrt(np.mean(np.concatenate(loss_z)))), '{:.3}'.format(np.sqrt(np.mean(np.concatenate(loss_z_t)))), '{:.3}'.format(np.sqrt(np.mean(np.concatenate(loss_z_k))))])
            print('---------------------------------------------------')













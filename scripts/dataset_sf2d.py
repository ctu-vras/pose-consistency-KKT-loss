from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import os
import glob
from scipy import ndimage
import cv2
import copy
import matplotlib.pyplot as plt


class Dataset(Dataset):
    def __init__(self, path, seen=0, augment=True, transform=None):
        if not os.path.exists(path):
            os.makedirs(path)
        self.root_dir = path
        self.list = glob.glob(self.root_dir + '*label.npz')
        self.size = len(glob.glob(self.root_dir + '*label.npz'))
        self.augment = augment
        self.onsparse = False




    def __getitem__(self, idx):
        label_file = '{:06}'.format(idx) + '_label.npz'
        label_file = self.list[idx]
        # print(label_file)
        data = np.load(label_file)

        input = data['input']
        label = data['label']
        try:
            label_d = data['label_d']
        except:
            label_d = data['label']
        vis = data['visible_mask']
        feat = data['hardness']
        T_baselink_zpr = data["T_baselink_zpr"]
        yaw = data['yaw_label']
        roll_label = data['roll_label']
        pitch_label = data['pitch_label']
        z_label = data['z_baselink_label']

        not_rot_input = copy.deepcopy(input)
        not_rot_input[np.isnan(input)] = 0
        not_rot_bw_dist = ndimage.distance_transform_edt(not_rot_input == 0)
        r = 0
        if (self.augment):
            not_rot_input = copy.deepcopy(input)
            not_rot_input[np.isnan(input)] = 0
            not_rot_bw_dist = ndimage.distance_transform_edt(not_rot_input == 0)
            np.random.seed()
            r = np.random.random_integers(0, 3)
            input = np.rot90(input, r, axes=(0, 1))
            label = np.rot90(label, r, axes=(0, 1))
            label_d = np.rot90(label_d, r, axes=(0, 1))
            vis = np.rot90(vis, r, axes=(0, 1))
            feat = np.rot90(feat, r, axes=(1, 2))
            # spatenka - ale pro uceni nevadi??? T_baselink_zpr = data["T_baselink_zpr"]
            # yaw = np.rot90(yaw,r,axes=(0,1)) #+ r*np.pi/2 # todo
            # roll_label = np.rot90(roll_label,r,axes=(0,1))
            # pitch_label = np.rot90(pitch_label,r,axes=(0,1))
            # z_label = np.rot90(z_label,r,axes=(0,1))


        # vis[~np.isnan(vis)] = 0
        # vis[np.isnan(vis)] = 1
        vis[~np.isnan(input)] = 1

        bw_dist = ndimage.distance_transform_edt(vis == 0)
        vis = ~np.isnan(label) & (bw_dist < 10)  # distance in decimeters from visible
        mask = ~np.isnan(input)

        label[np.isnan(label)] = 0
        # label[label>0.5] = 0.5
        label_d[np.isnan(label_d)] = 0
        # label_d[label_d>0.5] = 0.5
        mask = mask.astype(np.float32)
        vis = vis.astype(np.float32)

        feat[np.isnan(feat)] = 0

        imgs = [cv2.imdecode(data['img0'], cv2.IMREAD_COLOR), cv2.imdecode(data['img1'], cv2.IMREAD_COLOR),
                cv2.imdecode(data['img2'], cv2.IMREAD_COLOR), cv2.imdecode(data['img3'], cv2.IMREAD_COLOR),
                cv2.imdecode(data['img4'], cv2.IMREAD_COLOR)]

        yaw_mask = abs(yaw) < (np.pi / 8)

        roll_label[~yaw_mask] = np.nan
        pitch_label[~yaw_mask] = np.nan
        z_label[~yaw_mask] = np.nan

        # test nahrazeni bw_dist
        if self.onsparse:
            pos_mask = (yaw_mask) & (not_rot_bw_dist < 10)  & (ndimage.distance_transform_edt(~np.isnan(input)) > 4)
        else:
            pos_mask = (yaw_mask) & (not_rot_bw_dist < 10) # distance in decimeters from visible
        # pos_indexes = np.where(pos_mask)
        input[np.isnan(input)] = 0

        roll_label[~pos_mask] = np.nan
        pitch_label[~pos_mask] = np.nan
        z_label[~pos_mask] = np.nan

        z_label = z_label.astype(np.float32) - 0.05
#        print(z_label[~np.isnan(z_label)].size)
#        print(sum(sum((z_label[:,:]==0))))
        rpz = np.stack([roll_label, pitch_label, z_label])

        x_antipad = 3
        y_antipad = 4
        rpz = rpz[:, x_antipad:-x_antipad, y_antipad:-y_antipad]
        # yaw = data['yaw_label']
        # yaw = yaw[x_antipad:-x_antipad, y_antipad:-y_antipad]
        class_mask = np.sum(feat, axis=0) > 0

        max_feat = np.argmax(feat, axis=0)
        # 0 1 3 6 11 12 13 16 20 32 34  53 61 87 102

        class_weight = np.ones_like(max_feat)

        class_weight[(class_mask & ((max_feat == 0) | (max_feat == 1) | (max_feat == 3)
                                    | (max_feat == 6) | (max_feat == 11) | (max_feat == 12) | (max_feat == 13)
                                    | (max_feat == 16) | (max_feat == 20) | (max_feat == 32) | (max_feat == 34)
                                    | (max_feat == 53) | (max_feat == 61) | (max_feat == 87) | (max_feat == 102)))] = 10

        length = 0.8
        width = 0.6
        grid_res = 0.1
        robot_model = np.meshgrid(np.arange(-length / 2, length / 2 + grid_res, grid_res),
                    np.arange(-width / 2, width / 2 + grid_res, grid_res))
        z_robot = np.zeros_like(robot_model[0])

        robot = np.vstack([robot_model[0].ravel(),robot_model[1].ravel(),z_robot.ravel()])

        Rx = np.zeros([3,3])
        Ry = np.zeros([3,3])



        #label_synth = label
        label_synth = z_label
        indexes = np.where(pos_mask)

        for i in range(indexes[0].size):
            Rx[0, 0] = 1
            Rx[1, 1] = np.cos(roll_label[indexes[0][i],indexes[1][i]])
            Rx[1, 2] = -np.sin(roll_label[indexes[0][i],indexes[1][i]])
            Rx[2, 1] = np.sin(roll_label[indexes[0][i],indexes[1][i]])
            Rx[2, 2] = np.cos(roll_label[indexes[0][i],indexes[1][i]])

            Ry[0, 0] = np.cos(pitch_label[indexes[0][i],indexes[1][i]])
            Ry[0, 2] = np.sin(pitch_label[indexes[0][i],indexes[1][i]])
            Ry[1, 1] = 1
            Ry[2, 0] = -np.sin(pitch_label[indexes[0][i],indexes[1][i]])
            Ry[2, 2] = np.cos(pitch_label[indexes[0][i],indexes[1][i]])

            rot_robot = np.matmul(Ry, np.matmul(Rx, robot))
            rot_robot_z = np.reshape(rot_robot[2], [7, 9])
            label_synth[indexes[0][i] - 3:indexes[0][i] + 4, indexes[1][i] - 4:indexes[1][i] + 5] = z_label[indexes[0][i],indexes[1][i]]+rot_robot_z



        sample = {'rot': r, 'class_weights': class_weight.astype(np.float32)[np.newaxis],
                  'input': input.astype(np.float32)[np.newaxis], 'label': label.astype(np.float32)[np.newaxis],
                  'label_d': label_d.astype(np.float32)[np.newaxis], 'label_synth': label_synth.astype(np.float32)[np.newaxis], 'mask': mask[np.newaxis],
                  'weights': vis[np.newaxis], 'images': imgs, 'T_baselink_zpr': T_baselink_zpr.astype(np.float32),
                  'features': feat.astype(np.float32)[np.newaxis], 'label_rpz': rpz.astype(np.float32),
                  'label_yaw': yaw.astype(np.float32)[np.newaxis]}
        # if self.transform:
        #    sample = self.transform(sample)
        return sample

    def __len__(self):
        return self.size


    def set_onsparse(self, Value):
        self.onsparse = Value
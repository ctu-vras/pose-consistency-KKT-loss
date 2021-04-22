from torch.utils.data import Dataset
import numpy as np
import os
import glob
from scipy import ndimage


class Dataset(Dataset):
    def __init__(self, path, seen=0, transform=None):
        if not os.path.exists(path):
            os.makedirs(path)
        self.root_dir = path
        self.list = glob.glob(self.root_dir + '*label.npz')
        self.size = len(glob.glob(self.root_dir + '*label.npz'))

    def __getitem__(self, idx):
        label_file = '{:06}'.format(idx) + '_label.npz'
        data = np.load(self.root_dir + label_file)

        input = data['input']

        yaw_mask = abs(data['yaw_label']) < (np.pi / 8)

        roll_label = data['roll_label']
        pitch_label = data['pitch_label']
        z_label = data['z_baselink_label']
        roll_label[~yaw_mask] = np.nan
        pitch_label[~yaw_mask] = np.nan
        z_label[~yaw_mask] = np.nan

        vis = data['visible_mask']
        vis[~np.isnan(vis)] = 0
        vis[np.isnan(vis)] = 1
        bw_dist = ndimage.distance_transform_edt(vis)
        pos_mask = ~np.isnan(yaw_mask) & (bw_dist < 10)  # distance in decimeters from visible
        # pos_indexes = np.where(pos_mask)

        roll_label[~pos_mask] = np.nan
        pitch_label[~pos_mask] = np.nan
        z_label[~pos_mask] = np.nan

        z_label = z_label - 0.05
        rpz = np.stack([roll_label, pitch_label, z_label])

        x_antipad = 3
        y_antipad = 4
        rpz = rpz[:, x_antipad:-x_antipad, y_antipad:-y_antipad]
        yaw = data['yaw_label']
        yaw = yaw[x_antipad:-x_antipad, y_antipad:-y_antipad]

        label = data['label']
        try:
            label_d = data['label_d']
        except:
            label_d = data['label']

        vis = ~np.isnan(label) & (bw_dist < 10)  # distance in decimeters from visible
        mask = ~np.isnan(input)

        input[np.isnan(input)] = 0
        label[np.isnan(label)] = 0
        label_d[np.isnan(label_d)] = 0
        label[label > 0.5] = 0.5
        label_d[label_d > 0.5] = 0.5
        mask = mask.astype(np.float32)
        vis = vis.astype(np.float32)

        sample = {'input': input.astype(np.float32)[np.newaxis], 'mask': mask.astype(np.float32)[np.newaxis],
                  'label_rpz': rpz.astype(np.float32), 'label_dem': label.astype(np.float32)[np.newaxis],
                  'label_dem_d': label_d.astype(np.float32)[np.newaxis], 'weights': vis[np.newaxis], 'yaw': yaw}
        return sample

    def __len__(self):
        return self.size

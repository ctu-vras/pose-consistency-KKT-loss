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
        data = np.load(self.root_dir+label_file)

        input = data['input']
        label = data['label']
        try:
            label_d = data['label_d']
        except:
            label_d = data['label']
        vis = data['visible_mask']

        vis[~np.isnan(vis)] = 0
        vis[np.isnan(vis)] = 1
        bw_dist = ndimage.distance_transform_edt(vis)
        vis = ~np.isnan(label) & (bw_dist < 10) # distance in decimeters from visible
        mask = ~np.isnan(input)

        input[np.isnan(input)] = 0
        label[np.isnan(label)] = 0
        label[label>0.5] = 0.5
        label_d[np.isnan(label_d)] = 0
        label_d[label_d>0.5] = 0.5
        mask = mask.astype(np.float32)
        vis = vis.astype(np.float32)

        sample = {'input': input.astype(np.float32)[np.newaxis], 'label': label.astype(np.float32)[np.newaxis],'label_d': label_d.astype(np.float32)[np.newaxis], 'mask': mask[np.newaxis], 'weights': vis[np.newaxis]}
        return sample

    def __len__(self):
        return self.size


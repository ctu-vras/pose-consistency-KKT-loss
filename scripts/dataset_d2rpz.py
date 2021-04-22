from torch.utils.data import Dataset
import numpy as np
import os
import glob

class Dataset(Dataset):
    def __init__(self, path, seen=0, transform=None):
        if not os.path.exists(path):
            os.makedirs(path)
        self.root_dir = path
        self.list = glob.glob(self.root_dir + '*label.npz')
        self.size = len(glob.glob(self.root_dir + '*label.npz'))

    def __getitem__(self, idx):
        label_file = '{:04}'.format(idx) + '_label.npz'
        data = np.load(self.root_dir+label_file)

        input = data['input']
        label = data['label']
        x_antipad = 3
        y_antipad = 4
        label = label[[0,1,5],x_antipad:-x_antipad,y_antipad:-y_antipad]

        sample = {'input': input.astype(np.float32)[np.newaxis], 'label': label.astype(np.float32)}
        return sample

    def __len__(self):
        return self.size



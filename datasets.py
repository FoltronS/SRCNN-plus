import random

import h5py
import numpy as np
from torch.utils.data import Dataset


class TrainDataset(Dataset):
    def __init__(self, h5_file, augment=True):
        super(TrainDataset, self).__init__()
        self.h5_file = h5_file
        self.augment = augment

    def __getitem__(self, idx):
        with h5py.File(self.h5_file, 'r') as f:
            lr = f['lr'][idx] / 255.
            hr = f['hr'][idx] / 255.

        if self.augment:
            if random.random() > 0.5:
                lr = np.fliplr(lr).copy()
                hr = np.fliplr(hr).copy()
            if random.random() > 0.5:
                lr = np.flipud(lr).copy()
                hr = np.flipud(hr).copy()
            k = random.randint(0, 3)
            lr = np.rot90(lr, k).copy()
            hr = np.rot90(hr, k).copy()

        return np.expand_dims(lr, 0), np.expand_dims(hr, 0)

    def __len__(self):
        with h5py.File(self.h5_file, 'r') as f:
            return len(f['lr'])


class EvalDataset(Dataset):
    def __init__(self, h5_file):
        super(EvalDataset, self).__init__()
        self.h5_file = h5_file

    def __getitem__(self, idx):
        with h5py.File(self.h5_file, 'r') as f:
            return np.expand_dims(f['lr'][str(idx)][:, :] / 255., 0), np.expand_dims(f['hr'][str(idx)][:, :] / 255., 0)

    def __len__(self):
        with h5py.File(self.h5_file, 'r') as f:
            return len(f['lr'])

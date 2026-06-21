import torch
from torch.utils.data import Dataset
import h5py


class PhaseAwareDataset(Dataset):
    def __init__(self, h5_file):
        self.h5_file = h5_file
        with h5py.File(self.h5_file, 'r') as f:
            self.length = f['input'].shape[0]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        with h5py.File(self.h5_file, 'r') as f:
            inputs = torch.from_numpy(f['input'][idx])
            gts = torch.from_numpy(f['ground_truth'][idx])
            meta = torch.from_numpy(f['metadata'][idx])
        return inputs, gts, meta
import h5py
import torch
from torch.utils.data import Dataset


class PhaseAwareDataset(Dataset):
    def __init__(self, h5_file):
        self.h5_file = str(h5_file)
        self._h5 = None

        with h5py.File(self.h5_file, "r") as f:
            self.length = int(f["input"].shape[0])

    def _get_file(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_file, "r")
        return self._h5

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        f = self._get_file()

        x = torch.from_numpy(f["input"][idx]).float()
        y = torch.from_numpy(f["ground_truth"][idx]).float()
        meta = torch.from_numpy(f["metadata"][idx]).float()

        return x, y, meta

    def __del__(self):
        try:
            if self._h5 is not None:
                self._h5.close()
        except Exception:
            pass
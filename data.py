from torch.utils.data import Dataset, ConcatDataset, DataLoader, random_split
from scipy.io import wavfile
import os
import torch
from torch import nn
import numpy as np
import pytorch_lightning as L


def make_pairings(i):
    i = [x for x in i if "67_near" in x or "nt1_middle" in x]
    D = {}
    for x in i:
        sp = x.split("/")[-1].split(".")[0].split("_")[-1]
        if not (sp in D):
            D[sp] = []
        D[sp].append(x)
    o = []
    for x in D.values():
        ii = 0 if "nt1_middle" in x[0] else 1
        ti = 0 if ii == 1 else 1
        o.append((x[ii], x[ti]))
    return o


class AudioData(Dataset):
    def __init__(self, window_size, stride, i, t):
        self.window_size = window_size
        self.stride = stride
        self.mask = nn.Transformer.generate_square_subsequent_mask(window_size)
        _, i = wavfile.read(i)
        _, t = wavfile.read(t)
        assert i.dtype == np.int16
        assert t.dtype == np.int16
        i, t = i.astype(np.int32), t.astype(np.int32)
        assert i.min() < 0
        assert t.min() < 0
        i, t = i + 32768, t + 32768
        assert i.min() >= 0
        assert t.min() >= 0
        min_l = min(i.shape[0], t.shape[0])
        self.i = i[:min_l]
        self.t = t[:min_l]

    def __len__(self):
        li = len(self.i)
        n = (li - 1 - self.window_size) // self.stride
        return n

    def __getitem__(self, idx):
        st = idx * self.stride
        ed = st + self.window_size
        return self.i[st:ed], self.t[st:ed], self.t[st + 1 : ed + 1], self.mask


class AudioDataModule(L.LightningDataModule):
    def __init__(self, data_dir, batch_size, window_size, stride):
        super().__init__()
        self.save_hyperparameters()

    # no stage needed
    def setup(self, stage):

        data_path = self.params.data_dir

        day1 = [
            os.path.join(data_path, "day1", x)
            for x in os.listdir(os.path.join(data_path, "day1"))
        ]
        day2 = [
            os.path.join(data_path, "day2", x)
            for x in os.listdir(os.path.join(data_path, "day2"))
        ]

        files = make_pairings(day1) + make_pairings(day2)

        dataset = ConcatDataset(
            [
                AudioData(self.params.window_size, self.params.stride, i, t)
                for i, t in files
            ]
        )
        self.train, self.val, self.test = random_split(
            dataset, [0.85, 0.1, 0.5], generator=torch.Generator().manual_seed(42)
        )

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.params.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.params.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.params.batch_size)

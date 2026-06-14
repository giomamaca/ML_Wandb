import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd

class FERDataset(Dataset):
    def __init__(self, df, augment=False, has_labels=True):
        self.pixels = df['pixels'].values
        self.has_labels = has_labels and ('emotion' in df.columns)
        self.labels = df['emotion'].values if self.has_labels else None
        self.augment = augment

    def __len__(self):
        return len(self.pixels)

    def __getitem__(self, idx):
        img = np.array(self.pixels[idx].split(), dtype=np.uint8).reshape(48, 48)
        img = torch.tensor(img, dtype=torch.float32).unsqueeze(0) / 255.0
        if self.augment:
            if torch.rand(1) < 0.5:
                img = torch.flip(img, dims=[2])
            i, j = torch.randint(0, 5, (2,))
            img = torch.nn.functional.pad(img[:, i:i+44, j:j+44], (2, 2, 2, 2))
        if self.has_labels:
            return img, torch.tensor(self.labels[idx], dtype=torch.long)
        return img

def get_loaders(train_csv, batch_size=64, augment_train=False, val_ratio=0.1, seed=42):
    df = pd.read_csv(train_csv)
    rng = np.random.default_rng(seed)
    val_idx = []
    for c in df['emotion'].unique():
        idx = df.index[df['emotion'] == c].to_numpy()
        rng.shuffle(idx)
        val_idx.extend(idx[:int(len(idx) * val_ratio)])
    val_mask = df.index.isin(val_idx)
    train_ds = FERDataset(df[~val_mask], augment=augment_train)
    val_ds   = FERDataset(df[val_mask],  augment=False)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=2)
    val_loader   = torch.utils.data.DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, val_loader

def get_test_loader(test_csv, batch_size=64):
    df = pd.read_csv(test_csv)
    test_ds = FERDataset(df, augment=False, has_labels=False)
    return torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2)
import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd

class FERDataset(Dataset):
    def __init__(self, df, augment=False):
        self.pixels = df['pixels'].values
        self.labels = df['emotion'].values
        self.augment = augment

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = np.fromstring(self.pixels[idx], dtype=np.uint8, sep=' ').reshape(48, 48)
        img = torch.tensor(img, dtype=torch.float32).unsqueeze(0) / 255.0
        if self.augment:
            if torch.rand(1) < 0.5:
                img = torch.flip(img, dims=[2])
            i, j = torch.randint(0, 5, (2,))
            img = torch.nn.functional.pad(img[:, i:i+44, j:j+44], (2, 2, 2, 2))
        return img, torch.tensor(self.labels[idx], dtype=torch.long)

def get_loaders(csv_path, batch_size=64, augment_train=False):
    df = pd.read_csv(csv_path)
    train_ds = FERDataset(df[df['Usage'] == 'Training'], augment=augment_train)
    val_ds   = FERDataset(df[df['Usage'] == 'PublicTest'])
    test_ds  = FERDataset(df[df['Usage'] == 'PrivateTest'])
    return (torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2),
            torch.utils.data.DataLoader(val_ds, batch_size=batch_size, num_workers=2),
            torch.utils.data.DataLoader(test_ds, batch_size=batch_size, num_workers=2))

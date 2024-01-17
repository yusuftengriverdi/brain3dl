import torch
from torch.utils.data import Dataset, DataLoader
from utils import normalization_0_1

class CustomDataset(Dataset):
    def __init__(self, X, y, transform = None, mask_transform = None, preprocessing = None):
            # Add channel information.
        self.X = torch.Tensor(X).unsqueeze(1)
        self.y = torch.Tensor(y).unsqueeze(1)

        self.transform = transform
        self.mask_transform = mask_transform
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        
        volume = self.X[idx]
        label = self.y[idx]

        if self.transform:
            volume = self.transform(volume)

        if self.mask_transform:
            label = self.mask_transform(label)

        sample = {'volume': volume, 'label': label}
        return sample


def get_dataloaders(dataset, batch_size=16, shuffle=False):

    # Create DataLoader instances for training and validation sets
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4)

    return dataloader

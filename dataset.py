import torch
from torch.utils.data import Dataset, DataLoader
from albumentations.pytorch import ToTensorV2


def min_max_normalization_tensor(tensor):
    """
    Perform 0-1 normalization (Min-Max scaling) on a PyTorch tensor along a specified axis.

    Parameters:
    - tensor: PyTorch tensor containing the feature values.
    - axis: Axis along which normalization is performed (default is 0 for column-wise normalization).

    Returns:
    - Normalized tensor.
    """
    tensor = tensor.squeeze(1)

    min_val = torch.min(tensor)
    max_val = torch.max(tensor)

    if not max_val - min_val == 0:
        return (tensor - min_val) / (max_val - min_val)
    else: 
        return tensor

class CustomDataset(Dataset):
    def __init__(self, X, y, transform = None, preprocessing = None):
            # Add channel information.
        self.X = X
        self.y = y

        self.transform = transform
        self.preprocessing = preprocessing

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        
        image = self.X[idx]
        mask = self.y[idx]

        sample = {'image': image, 'mask': mask}

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            sample = {'image': min_max_normalization_tensor(augmented['image']), 'mask': augmented['mask']}
        
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            sample = {'image': torch.tensor(sample['image']).unsqueeze(0), 'mask': torch.tensor(sample['mask']).unsqueeze(0)}
        return sample


def get_dataloaders(dataset, batch_size=16, shuffle=False):

    # Create DataLoader instances for training and validation sets
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4)

    return dataloader

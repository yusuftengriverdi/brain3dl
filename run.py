import utils, metrics, phases, dataset
from models import segnet

import torchvision.transforms as T

if __name__ == '__main__':

    PATCH_SIZE = (256, 256, 128)
    BATCH_SIZE = 16
    # Load Train Data
    image_size = (256, 256, 128)  # Replace with your actual image size
    lr_rate = 0.001

    volume_list = [1, 3, 4, 5, 6, 7, 8, 9, 16, 18] 
    fname_pattern = 'data/Training_Set/IBSR_{0:02d}/IBSR_{0:02d}{1:}.nii.gz' # Replace with your actual file name pattern
    volumes, labels = utils.load_data(volume_list, image_size, fname_pattern)
    train_X, train_y = utils.extract_useful_patches(volumes, labels, patch_size=PATCH_SIZE)
    
    mean, std = utils.calculate_mean_std(train_X)
    normalize = T.Normalize(mean=mean, std=std)

    train_dataset = dataset.CustomDataset(train_X, train_y, transform=normalize)

    train_loader = dataset.get_dataloaders(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    del train_X, train_y, train_dataset

    volume_list = [11, 12, 13, 14, 17]  # Replace with your actual volume IDs
    fname_pattern = 'data/Validation_Set/IBSR_{0:02d}/IBSR_{0:02d}{1:}.nii.gz' # Replace with your actual file name pattern
    volumes, labels = utils.load_data(volume_list, image_size, fname_pattern)
    val_X, val_y = utils.extract_useful_patches(volumes, labels, patch_size=PATCH_SIZE)

    val_dataset = dataset.CustomDataset(val_X, val_y, transform=normalize)

    val_loader = dataset.get_dataloaders(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    del val_X, val_y, val_dataset

    model = segnet.SegNet(n_classes=3, n_input_channels=1, scaling_factor=2
                        )

    phases.train_and_validate(model, train_loader, val_loader, num_epochs=5, learning_rate=lr_rate, device='cpu')





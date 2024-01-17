import utils, metrics, phases, dataset
from models import segnet3d

import torchvision.transforms as T

if __name__ == '__main__':

    PATCH_SIZE = (64, 64, 16)
    BATCH_SIZE = 16
    # Load Train Data
    image_size = (256, 256, 128)  # Replace with your actual image size
    lr_rate = 0.1

    volume_list = [1, 3, 4, 5, 6, 7, 8, 9, 16, 18] 
    fname_pattern = 'data/Training_Set/IBSR_{0:02d}/IBSR_{0:02d}{1:}.nii.gz' # Replace with your actual file name pattern
    volumes, labels = utils.load_data(volume_list, image_size, fname_pattern, key='_denoised_v2')
    train_X, train_y, _ = utils.extract_useful_patches(volumes, labels, patch_size=PATCH_SIZE)
    
    mean, std = utils.calculate_mean_std(train_X)
    normalize = T.Normalize(mean=mean, std=std)

    # Augmentation
    augment = T.Compose([T.RandomHorizontalFlip(),
                         T.RandomVerticalFlip(),
                         T.RandomRotation(degrees=[0, 135]),
                         normalize,
                         ])


    train_dataset = dataset.CustomDataset(train_X, train_y, transform=None)

    train_loader = dataset.get_dataloaders(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    del train_X, train_y, train_dataset

    volume_list = [11, 12, 13, 14, 17]  # Replace with your actual volume IDs
    fname_pattern = 'data/Validation_Set/IBSR_{0:02d}/IBSR_{0:02d}{1:}.nii.gz' # Replace with your actual file name pattern
    volumes, labels = utils.load_data(volume_list, image_size, fname_pattern)
    val_X, val_y, _, val_X_all_patches, val_y_all_patches, _ = utils.extract_useful_patches(volumes, labels, patch_size=PATCH_SIZE, return_all=True)

    val_dataset = dataset.CustomDataset(val_X, val_y, transform=None)

    val_loader = dataset.get_dataloaders(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    del val_X, val_y, val_dataset

    model = segnet3d.SegNet3D(n_classes=3, n_input_channels=1, scaling_factor=2
                        )

    # current problems --> max 0.45 dice (batch 16, patch 32,32,16 lr 0.01), training loss does not drop altho dice score increases up to .46.
    # why doesn't dice work? 
    
    # apply lr scheduler ? 

    # improve model ?  how? 

    # augmentation to increase -- instead of on air? 


    # TODO list.

    # unpatchify val patches --> reconstructed_image = unpatchify(patches, image.shape)

    # predict with whole image and/ or predict all patches and merge.

    # calculate metrics. select best method (whole or merge)

    # predict with tests and save the results. 

    phases.train_and_validate(model, train_loader, val_loader, num_epochs=30, learning_rate=lr_rate, device='cpu')





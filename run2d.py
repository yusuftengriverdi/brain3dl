import utils, metrics, phases, dataset
from models import segnet2d
from torch.hub import load 
import torch.nn as nn
import segmentation_models_pytorch as smp
import torchvision.transforms as T

if __name__ == '__main__':

    PATCH_SIZE = (128, 128)
    BATCH_SIZE = 8
    # Load Train Data
    image_size = (256, 256, 128)  # Replace with your actual image size
    lr_rate = 0.1

    volume_list = [1, 3, 4, 5, 6, 7, 8, 9, 16, 18] 
    fname_pattern = 'data/Training_Set/IBSR_{0:02d}/IBSR_{0:02d}{1:}.nii.gz' # Replace with your actual file name pattern
    volumes, labels, fnames = utils.load_data_2d_slices(volume_list, image_size, fname_pattern, key='_denoised_v3')
    train_X, train_y, train_fnames = utils.extract_useful_patches(volumes, labels, patch_size=PATCH_SIZE, fnames=fnames)
    
    # mean, std = utils.calculate_mean_std(train_X)
    # normalize = T.Normalize(mean=mean, std=std)

    # Augmentation -to mask as well. 
    augment = T.Compose([T.RandomHorizontalFlip(),
                         T.RandomVerticalFlip(),
                         T.RandomRotation(degrees=[0, 135]),
                         ])

    print(train_X.shape, train_y.shape, len(train_fnames))

    train_dataset = dataset.CustomDataset(train_X, train_y, transform=None, mask_transform=None)

    train_loader = dataset.get_dataloaders(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    del train_X, train_y, train_dataset

    volume_list = [11, 12, 13, 14, 17]  # Replace with your actual volume IDs
    fname_pattern = 'data/Validation_Set/IBSR_{0:02d}/IBSR_{0:02d}{1:}.nii.gz' # Replace with your actual file name pattern
    volumes, labels, fnames = utils.load_data_2d_slices(volume_list, image_size, fname_pattern, key='_denoised_v3')
    val_X, val_y, val_fnames, val_X_all_patches, val_y_all_patches, val_fnames_info = utils.extract_useful_patches(volumes, labels, patch_size=PATCH_SIZE, return_all=True, fnames=fnames)

    val_dataset = dataset.CustomDataset(val_X, val_y, transform=None, mask_transform=None)

    val_loader = dataset.get_dataloaders(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    del val_X, val_y, val_dataset

    # model = segnet2d.SegNet2D(n_classes=3, n_input_channels=1, scaling_factor=2
    #                     )

    # model = load('mateuszbuda/brain-segmentation-pytorch', 'unet',
    # in_channels=1, out_channels=3, init_features=32, pretrained=False)
    # # # Add a new Conv2d layer on top
    # # additional_conv_layer = nn.Conv2d(in_channels=32, out_channels=3, kernel_size=3, stride=1, padding=1)

    # # # Create a new model that includes the original UNet and the additional Conv2d layer
    # # model = nn.Sequential(
    # #     model,
    # #     additional_conv_layer
    # # )

    ENCODER = 'resnet50'
    CLASSES = [0, 1, 2]
    ACTIVATION = 'softmax2d' # could be None for logits or 'softmax2d' for multiclass segmentation

        # create segmentation model with pretrained encoder
    model = smp.Unet(
        encoder_name=ENCODER, 
        encoder_weights=None, 
        classes=len(CLASSES), 
        activation=ACTIVATION,
        in_channels=1,
    )


    print(model)

    phases.train_and_validate(model, train_loader, val_loader, num_epochs=30, learning_rate=lr_rate, device='cpu')

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






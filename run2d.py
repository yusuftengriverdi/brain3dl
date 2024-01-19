import os
import datetime
import logging
import utils
import phases
import dataset
from models import segnet2d
from torch.hub import load 
import segmentation_models_pytorch as smp
import torchvision.transforms as T
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
from tqdm import tqdm
from patchify import patchify, unpatchify


def get_preprocessing(preprocessing_fn=None):
    """Construct preprocessing transform    
    Args:
        preprocessing_fn (callable): data normalization function 
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    """   
    _transform = []
    if preprocessing_fn:
        _transform.append(A.Lambda(image=preprocessing_fn))
        
    return A.Compose(_transform)

def setup_logging(date):
    log_folder = 'runs/'
    os.makedirs(log_folder, exist_ok=True)
    
    log_filename = os.path.join(log_folder, f'log_{date}.txt')
    
    logging.basicConfig(filename=log_filename, level=logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler())  # Print logs to console as well


if __name__ == '__main__':

    date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    setup_logging(date=date)

    # ... (rest of your code)

    PATCH_SIZE = (32, 32)
    BATCH_SIZE = 64
    # Load Train Data
    image_size = (256, 256, 128)  # Replace with your actual image size
    lr_rate = 0.1

    ENCODER = 'resnet50'
    CLASSES = [0, 1, 2]
    ACTIVATION = 'softmax2d' # could be None for logits or 'softmax2d' for multiclass segmentation

        # create segmentation model with pretrained encoder
    # model = smp.Unet(
    #     encoder_name=ENCODER, 
    #     encoder_weights='imagenet', 
    #     classes=len(CLASSES), 
    #     activation=ACTIVATION,
    #     in_channels=1,
    # )

    # preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, 'imagenet')
    
    # preprocessing_ = get_preprocessing(preprocessing_fn)

    preprocessing_ = None

    logging.info(preprocessing_)

    volume_list = [1, 3, 4, 5, 6, 7, 8, 9, 16, 18] 
    fname_pattern = 'data/Training_Set/IBSR_{0:02d}/IBSR_{0:02d}{1:}.nii.gz' # Replace with your actual file name pattern
    volumes, labels, fnames = utils.load_data_2d_slices(volume_list, image_size, fname_pattern, key='_denoised_v3')
    train_X, train_y, train_fnames = utils.extract_useful_patches(volumes, labels, patch_size=PATCH_SIZE, fnames=fnames)


    # Augmentation -to mask as well. 
    # Define the augmentation pipeline
    augmentation = A.Compose([
        A.HorizontalFlip(p=0.5),  # Adjust the probability as needed
        A.VerticalFlip(p=0.5),    # Adjust the probability as needed
        A.Rotate(limit=135, p=0.5),  # Adjust the limit and probability as needed
        # A.Normalize(),  # You can add more augmentations as needed
        # A.RandomBrightness(),
        ToTensorV2(),
    ])

    print(train_X.shape, train_y.shape, len(train_fnames))

    train_dataset = dataset.CustomDataset(train_X, train_y, transform=ToTensorV2(), preprocessing=preprocessing_)

    logging.info(f"AUGMENT: {train_dataset.transform}")

    train_loader = dataset.get_dataloaders(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    del train_X, train_y, train_dataset

    volume_list = [11, 12, 13, 14, 17]  # Replace with your actual volume IDs
    fname_pattern = 'data/Validation_Set/IBSR_{0:02d}/IBSR_{0:02d}{1:}.nii.gz' # Replace with your actual file name pattern
    volumes, labels, fnames = utils.load_data_2d_slices(volume_list, image_size, fname_pattern, key='_denoised_v3')
    val_X, val_y, val_fnames, val_X_all_patches, val_y_all_patches, val_fnames_info = utils.extract_useful_patches(volumes, labels, patch_size=PATCH_SIZE, return_all=True, fnames=fnames)

    val_dataset = dataset.CustomDataset(val_X, val_y, transform=ToTensorV2(), preprocessing=preprocessing_)

    val_loader = dataset.get_dataloaders(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    del val_X, val_y, val_dataset

    # model = segnet2d.SegNet2D(n_classes=3, n_input_channels=1, scaling_factor=2
    #                     )


    model = smp.Unet(
        encoder_name=ENCODER, 
        encoder_weights=None, 
        classes=len(CLASSES), 
        activation=ACTIVATION,
        in_channels=1,
    )

    # model = smp.UnetPlusPlus(
    #     encoder_name='resnet34',
    #     encoder_weights=None,
    #     classes=3,
    #     activation=ACTIVATION,
    #     in_channels=1
    # )

    # model = smp.PSPNet(
    #     encoder_name='resnet34',
    #     encoder_weights=None,
    #     classes=3,
    #     activation=ACTIVATION,
    #     in_channels=1
    # )

    # model = smp.FPN(
    #     encoder_name='resnet34',
    #     encoder_weights=None,
    #     classes=3,
    #     activation=ACTIVATION,
    #     in_channels=1
    # )

    
    # model = smp.MAnet(
    #     encoder_name='resnet34',
    #     encoder_weights=None,
    #     classes=3,
    #     activation=ACTIVATION,
    #     in_channels=1
    # )
    
    logging.info(model)
        # log hyperparameters
    logging.info("Hyperparameters:")
    logging.info(f"PATCH_SIZE: {PATCH_SIZE}")
    logging.info(f"BATCH_SIZE: {BATCH_SIZE}")
    logging.info(f"image_size: {image_size}")
    logging.info(f"lr_rate: {lr_rate}")
    logging.info(f"volume_list (Train): {volume_list}")
    logging.info(f"volume_list (Validation): {volume_list}")
    logging.info(f"ENCODER: {ENCODER}")
    logging.info(f"CLASSES: {CLASSES}")
    logging.info(f"ACTIVATION: {ACTIVATION}")


    phases.train_and_validate(model, train_loader, val_loader, num_epochs=30, learning_rate=lr_rate, device='cpu', date=date)


    def predict(test_imgs, patch_size, image_size, model, device):

        model.eval()

        pred_labels = []

        with torch.no_grad():
            
            pbar = tqdm(enumerate(test_imgs), total=len(test_imgs), desc="Initial Description")

            for batch, img in pbar:
                
                img = torch.tensor(img).unsqueeze(1).unsqueeze(1)
                print(img.shape)
                X_test_patches_raw = patchify(img, (patch_size), step=patch_size[0] )
                X_test_patches = X_test_patches_raw.reshape((-1, *patch_size))

                yhat_patches = []

                for X in X_test_patches:
                    X = dataset.min_max_normalization_tensor(X)

                    X = X.to(device)

                    yhat_patches.append(model(X))

                yhat_patches = yhat_patches.reshape(*X_test_patches_raw.shape)
                
                pred_label = unpatchify(patches=yhat_patches, imsize=image_size)
                pred_labels.append(pred_label)
                
        return pred_labels
    
    fname_pattern = 'data/Test_Set/IBSR_{0:02d}/IBSR_{0:02d}{1:}.nii.gz' # Replace with your actual file name pattern

    test_volumes = utils.load_data_test(volume_list, fname_pattern, key='_denoised_v3')

    predict(volumes, patch_size=PATCH_SIZE, image_size=image_size, model=model, device='cpu')

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






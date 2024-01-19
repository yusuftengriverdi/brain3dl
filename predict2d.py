import utils
import dataset
import torch
from tqdm import tqdm
from patchify import patchify, unpatchify
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt

volume_list = [1]
PATCH_SIZE = (32, 32)
image_size = (256, 256, 128)

model_path = 'runs/best_model_val_dice_2024-01-18_04-15-13.pth'

model = torch.load(model_path)
print(model)

def predict(test_imgs, patch_size, image_size, model, device):

    model.eval()

    pred_labels = []

    with torch.no_grad():
        
        pbar = tqdm(enumerate(test_imgs), total=len(test_imgs), desc="Initial Description")

        for batch, img in pbar:

            X_test_patches_raw = patchify(img, (patch_size), step=patch_size[0] )
            X_test_patches = X_test_patches_raw.reshape((-1, *patch_size))

            X_test_patches = torch.tensor(X_test_patches).unsqueeze(1)
            # print("X_test_patches ", X_test_patches.shape)

            for idx, X in enumerate(X_test_patches):
                X = dataset.min_max_normalization_tensor(X)
                X_test_patches[idx] = X

            X_test_patches = X_test_patches.to(device)
            pred = model(X_test_patches)
            yhat_patches = np.array( torch.argmax(pred, dim=1))
            plt.imshow(pred[0, 2])
            plt.show()
            # print("Predicted patches ", yhat_patches.shape)
            yhat_patches = yhat_patches.reshape(X_test_patches_raw.shape)
            
            pred_label = unpatchify(patches=yhat_patches, imsize=(image_size[0], image_size[1]))
            print(pred_label.shape)
            # print("Predicted label ", pred_label.shape)
            pred_labels.append(pred_label)

    pred_labels = np.array(pred_labels)   
    print(pred_labels.shape)   

    pred_volumes = pred_labels.reshape(len(volume_list), *image_size)
    print(pred_volumes.shape)


    plt.imshow(pred_volumes[0, 20, :, :])
    plt.show()

    for idx, vol in enumerate(pred_volumes):
        vol = sitk.GetImageFromArray(np.transpose(vol))
        sitk.WriteImage(vol, f'test{idx}.nii.gz')
    return pred_volumes

fname_pattern = 'data/Training_Set/IBSR_{0:02d}/IBSR_{0:02d}{1:}.nii.gz' # Replace with your actual file name pattern

test_imgs = utils.load_data_test(volume_list, image_size, fname_pattern, key='_denoised_v3')

pred_labels = predict(test_imgs=test_imgs, patch_size=PATCH_SIZE, image_size=image_size, model=model, device='cpu')


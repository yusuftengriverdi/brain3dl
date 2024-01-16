import torch.nn.functional as F
import numpy as np
import SimpleITK as sitk

import torch.nn.functional as F
from patchify import patchify 
import matplotlib.pyplot as plt
import os 
import torch 
from tqdm import tqdm 

def load_data(volume_list, image_size, fname_pattern):
    """
    Load MRI volumes and corresponding segmentation labels.

    Args:
        volume_list (list): List of volume IDs.
        image_size (tuple): Size of the input volumes.
        fname_pattern (str): File name pattern for loading volumes.

    Returns:
        tuple: A tuple containing volumes and labels arrays.
    """
    n_volumes = len(volume_list)
    volumes = np.zeros((n_volumes, *image_size), dtype=np.float32)
    labels = np.zeros((n_volumes, *image_size), dtype=np.float32)
    
    for iFile, iID in enumerate(volume_list):
        img_data = sitk.ReadImage(fname_pattern.format(iID, '_denoised'))
        volumes[iFile, ...] = np.transpose(sitk.GetArrayFromImage(img_data), (2, 0, 1))

        seg_data = sitk.ReadImage(fname_pattern.format(iID, '_seg'))
        labels[iFile, ...] = np.transpose(sitk.GetArrayFromImage(seg_data), (2, 0, 1))

    return (volumes, labels)

def extract_useful_patches(volumes, labels, patch_size=(32, 32, 32), threshold=0.5):
    
    X_patches = []
    y_patches = []
    for X, y in zip(volumes, labels):
        #This will split the image into small images of shape [3,3,3]
        # patches = patchify(image, (3, 3, 3), step=1)
        X = patchify(X, (patch_size), step=patch_size[0] ).reshape((-1, patch_size[0], patch_size[1], patch_size[2]))
        y = patchify(y, (patch_size), step=patch_size[0] ).reshape((-1, patch_size[0], patch_size[1], patch_size[2]))

        # Check if m_patch contains values other than 0
        foreground = y != 0

        useful_patches = foreground.sum(axis=(1, 2, 3)) / len(foreground[0]) > threshold

        X_patches.append(X[useful_patches])
        y_patches.append(y[useful_patches])


    X_patches_flatten = []
    y_patches_flatten = []

    for idx in range(len(X_patches)):
        for vol_patch, seg_patch in zip(X_patches[idx], y_patches[idx]):
            X_patches_flatten.append(vol_patch)
            y_patches_flatten.append(seg_patch)
    
    X_patches_flatten = np.array(X_patches_flatten)
    y_patches_flatten = np.array(y_patches_flatten)
    
    print("Number of useful patches found: ", X_patches_flatten.shape[0])
    print("Number of volumes:", len(volumes))

    return X_patches_flatten, y_patches_flatten

def visualize_patches(X_patches, y_patches, rows, cols):
    fig, axes = plt.subplots(rows, cols, figsize=(cols*2, rows*2))

    n_slice = 20
    for i in range(rows):
        for j in range(cols):
            index = i * cols + j
            if index < len(X_patches):
                patch = X_patches[index, :, n_slice, :]  # Extract the image from the patch
                label = y_patches[index, :, n_slice, :] 

                axes[i, j].imshow(patch, cmap='gray')
                axes[i, j].contour(label, colors='red', levels=[0.5])  

            axes[i, j].axis('off')

    plt.tight_layout()
    plt.show()

def calculate_mean_std(data, cache_file="mean_std_cache.pth"):
    if os.path.exists(cache_file):
        # Load cached mean and std from file
        mean_std_dict = torch.load(cache_file)
        mean = mean_std_dict["mean"]
        std = mean_std_dict["std"]
    else:
        # Calculate mean and std
        means = []
        stds = []
        pbar = tqdm(data, total=len(data), desc="Calculating mean and std")
        for img in pbar:
            means.append(np.mean(img))
            stds.append(np.std(img))

        mean = np.mean(means)
        std = np.mean(stds)

        # Save mean and std to cache file
        mean_std_dict = {"mean": mean, "std": std}
        torch.save(mean_std_dict, cache_file)

    print(mean, std)
    return mean, std

# def main():
#     # Example usage
#     volume_list = [1, 3, 4, 5, 6, 7, 8, 9, 16, 18]  # Replace with your actual volume IDs
#     image_size = (256, 256, 128)  # Replace with your actual image size
#     fname_pattern = 'data/Training_Set/IBSR_{0:02d}/IBSR_{0:02d}{1:}.nii.gz' # Replace with your actual file name pattern

#     volumes, labels = load_data(volume_list, image_size, fname_pattern)

#     vol_patches, seg_patches = extract_useful_patches(volumes, labels)


#     # Print some information
#     print("Number of volumes:", len(volume_list))
#     print("Number of useful patches:", len(vol_patches))

#     visualize_patches(vol_patches, seg_patches, rows=5, cols=5)


# if __name__ == "__main__":
#     main()
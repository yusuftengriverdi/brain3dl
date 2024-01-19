import torch.nn.functional as F
import numpy as np
import SimpleITK as sitk

import torch.nn.functional as F
from patchify import patchify 
import matplotlib.pyplot as plt
import os 
import torch 
from tqdm import tqdm 

import albumentations as album

def load_data(volume_list, image_size, fname_pattern, key='_denoised_v2'):
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
    
    voxel_spacing = []
    for iFile, iID in enumerate(volume_list):
        img_data = sitk.ReadImage(fname_pattern.format(iID, key))
        volumes[iFile, ...] = np.transpose(sitk.GetArrayFromImage(img_data), (2, 0, 1))

        seg_data = sitk.ReadImage(fname_pattern.format(iID, '_seg'))
        labels[iFile, ...] = np.transpose(sitk.GetArrayFromImage(seg_data), (2, 0, 1))

        voxel_spacing += [img_data.GetSpacing()]
    
    np.savetxt('voxel_spacing.txt', voxel_spacing)

    return (volumes, labels)

def load_data_test(volume_list, image_size, fname_pattern, key='_denoised_v2'):
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
    
    for iFile, iID in enumerate(volume_list):
        img_data = sitk.ReadImage(fname_pattern.format(iID, key))
        volumes[iFile, ...] = np.transpose(sitk.GetArrayFromImage(img_data), (2, 0, 1))

    volumes = volumes.reshape(-1, volumes.shape[1], volumes.shape[2])

    return volumes


def load_data_2d_slices(volume_list, image_size, fname_pattern, key='_denoised_v2'):
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
    fnames = np.zeros((n_volumes, image_size[-1]), dtype=np.float32) # save fnames along z-axis.
    
    voxel_spacing = []
    for iFile, iID in enumerate(volume_list):
        img_data = sitk.ReadImage(fname_pattern.format(iID, key))
        volumes[iFile, ...] = np.transpose(sitk.GetArrayFromImage(img_data), (2, 0, 1))

        seg_data = sitk.ReadImage(fname_pattern.format(iID, '_seg'))
        labels[iFile, ...] = np.transpose(sitk.GetArrayFromImage(seg_data), (2, 0, 1))

        fnames[iFile, :] = iID

        voxel_spacing += [img_data.GetSpacing()]
    
    np.savetxt('voxel_spacing.txt', voxel_spacing)

    volumes, labels, fnames = volumes.reshape(-1, volumes.shape[1], volumes.shape[2]), labels.reshape(-1, labels.shape[1], labels.shape[2]), fnames.flatten()
    
    return volumes, labels, fnames


def extract_useful_patches(volumes, labels, patch_size=(32, 32, 32), threshold=0.5, return_all = False, fnames=None):
    
    if (not fnames is None) and len(patch_size) == 3:
        raise ValueError("In 2D slices, please provide only H and W of patches.")
    
    X_patches = []
    y_patches = []
    f_info_patches = []

    Xs, ys, fs = [], [], []  

    for idx, (X, y) in enumerate(zip(volumes, labels)):
        #This will split the image into small images of shape [3,3,3]
        # patches = patchify(image, (3, 3, 3), step=1)

        if (not fnames is None):
            f = fnames[idx]

        X = patchify(X, (patch_size), step=patch_size[0] ).reshape((-1, *patch_size))
        y = patchify(y, (patch_size), step=patch_size[0] ).reshape((-1, *patch_size))

        if return_all:
            Xs.append(X)
            ys.append(y)
            if (not fnames is None):
                fs.append(f)

        # Check if m_patch contains values other than 0
        foreground = y != 0

        if (fnames is None):
            useful_patches = foreground.sum(axis=(1, 2, 3)) / len(foreground[0]) > threshold
        else:
            useful_patches = foreground.sum(axis=(1, 2)) / len(foreground[0]) > threshold
            f_info_patches.append([f for i in useful_patches])

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
    print("Number of fnames:", len(f_info_patches))

    if return_all:
        return X_patches_flatten, y_patches_flatten, f_info_patches, Xs, ys, fs

    return X_patches_flatten, y_patches_flatten, f_info_patches


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


def main():
    # Example usage
    volume_list = [1, 3, 4, 5, 6, 7, 8, 9, 16, 18]  # Replace with your actual volume IDs
    image_size = (256, 256, 128)  # Replace with your actual image size
    fname_pattern = 'data/Training_Set/IBSR_{0:02d}/IBSR_{0:02d}{1:}.nii.gz' # Replace with your actual file name pattern

    volumes, labels = load_data(volume_list, image_size, fname_pattern)

    vol_patches, seg_patches, _ = extract_useful_patches(volumes, labels)


    # Print some information
    print("Number of volumes:", len(volume_list))
    print("Number of useful patches:", len(vol_patches))

    visualize_patches(vol_patches, seg_patches, rows=5, cols=5)


if __name__ == "__main__":
    main()

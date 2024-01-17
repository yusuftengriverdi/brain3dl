import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
from pathlib import Path
from tqdm import tqdm

def histogram_match(moving, target):
    """ Matches histogram of moving image to target image """
    matcher = sitk.HistogramMatchingImageFilter()
    if target.GetPixelID() in (sitk.sitkUInt8, sitk.sitkInt8):
        matcher.SetNumberOfHistogramLevels(128)
    else:
        matcher.SetNumberOfHistogramLevels(1024)
    matcher.SetNumberOfMatchPoints(7)
    matcher.ThresholdAtMeanIntensityOn()
    return matcher.Execute(moving, target)

def resample2target(moving, target):
    """Resamples moving image to target image"""
    # resample mask too.

    return sitk.Resample(moving, target.GetSize(),
                                    sitk.Transform(), 
                                    sitk.sitkLinear,
                                    target.GetOrigin(),
                                    target.GetSpacing(),
                                    target.GetDirection(),
                                    0,
                                    target.GetPixelID())


def bias_field_correction(test_img, shrinkFactor=1):
    """Performs bias field correction on image"""
    inputImage = sitk.Cast(test_img, sitk.sitkFloat32)
    image = inputImage

    if shrinkFactor > 1:
        image = sitk.Shrink(
            inputImage, [shrinkFactor] * inputImage.GetDimension()
        )

    corrector = sitk.N4BiasFieldCorrectionImageFilter()        

    return corrector.Execute(image)

# Denoising filter
def denoise(inputImage):
    inputImage = sitk.Cast(inputImage, sitk.sitkFloat32)
    return sitk.DiscreteGaussian(inputImage)

# Intensity normalization
def intensity_normalize(inputImage):
    inputImage = sitk.Cast(inputImage, sitk.sitkFloat32)
    return sitk.Normalize(inputImage)

if __name__ == '__main__':
    train_path = Path().resolve()/'data/Training_Set/'
    val_path = Path().resolve()/'data/Validation_Set'
    test_path = Path().resolve()/'data/Test_Set'

    target = sitk.ReadImage(str(train_path/'IBSR_04/IBSR_04.nii.gz'))
    
    target_bias = bias_field_correction(target)
    

    for val_set in tqdm(val_path.iterdir(), desc='Val Set'):
        val_img = sitk.ReadImage(str(val_set/f'{val_set.name}.nii.gz'))
        val_img = bias_field_correction(val_img)
        #val_img=preprocess_image(val_img,target_histogram_bias)   
        # create results folder, in this case we will overwrite the image
        results_path = val_set
        results_path.mkdir(exist_ok=True)
        sitk.WriteImage(val_img, str(results_path/f'{val_set.name}_bias_removed.nii.gz'))

        # val_img = resample2target(val_img, target)
        # val_img = histogram_match(val_img, target)
        val_img = denoise(val_img)
        sitk.WriteImage(val_img, str(results_path/f'{val_set.name}_denoised_v2.nii.gz'))
        val_img = intensity_normalize(val_img)
        sitk.WriteImage(val_img, str(results_path/f'{val_set.name}_denoised_v3.nii.gz'))


    for train_set in tqdm(train_path.iterdir(), desc='Train Set'):
        train_img = sitk.ReadImage(str(train_set/f'{train_set.name}.nii.gz'))
        train_img = bias_field_correction(train_img)
        # create results folder, in this case we will overwrite the image
        results_path = train_set
        results_path.mkdir(exist_ok=True)
        sitk.WriteImage(train_img, str(results_path/f'{train_set.name}_bias_removed.nii.gz'))

        # train_img = resample2target(train_img, target)
        # train_img = histogram_match(train_img, target)
        train_img = denoise(train_img)
        sitk.WriteImage(train_img, str(results_path/f'{train_set.name}_denoised_v2.nii.gz'))
        train_img = intensity_normalize(train_img)
        sitk.WriteImage(train_img, str(results_path/f'{train_set.name}_denoised_v3.nii.gz'))

    for test_set in tqdm(test_path.iterdir(), desc='Test Set'):
        test_img = sitk.ReadImage(str(test_set/f'{test_set.name}.nii.gz'))
        test_img = bias_field_correction(test_img) 
        # create results folder, in this case we will overwrite the image
        results_path = test_set
        results_path.mkdir(exist_ok=True)
        
        sitk.WriteImage(test_img, str(results_path/f'{test_set.name}_bias_removed.nii.gz'))

        # test_img = resample2target(test_img, target)
        # test_img = histogram_match(test_img, target)
        test_img = denoise(test_img)
        sitk.WriteImage(test_img, str(results_path/f'{test_set.name}_denoised_v2.nii.gz'))
        test_img = intensity_normalize(test_img)
        sitk.WriteImage(test_img, str(results_path/f'{test_set.name}_denoised_v3.nii.gz'))



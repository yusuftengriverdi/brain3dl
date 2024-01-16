import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
from pathlib import Path
from tqdm import tqdm

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


if __name__ == '__main__':
    train_path = Path().resolve()/'data/Training_Set/'
    val_path = Path().resolve()/'data/Validation_Set'
    test_path = Path().resolve()/'data/Test_Set'

    target_histogram = sitk.ReadImage(str(train_path/'IBSR_04/IBSR_04.nii.gz'))
    target_histogram_bias=bias_field_correction(target_histogram)
    

    for val_set in tqdm(val_path.iterdir(), desc='Val Set'):
        val_img = sitk.ReadImage(str(val_set/f'{val_set.name}.nii.gz'))
        val_img = bias_field_correction(val_img)
        #val_img=preprocess_image(val_img,target_histogram_bias)   
        # create results folder, in this case we will overwrite the image
        results_path = val_set
        results_path.mkdir(exist_ok=True)
        sitk.WriteImage(val_img, str(results_path/f'{val_set.name}_bias_removed.nii.gz'))

        val_img = denoise(val_img)
        sitk.WriteImage(val_img, str(results_path/f'{val_set.name}_denoised.nii.gz'))


    for train_set in tqdm(train_path.iterdir(), desc='Train Set'):
        train_img = sitk.ReadImage(str(train_set/f'{train_set.name}.nii.gz'))
        train_img = bias_field_correction(train_img)
        # create results folder, in this case we will overwrite the image
        results_path = train_set
        results_path.mkdir(exist_ok=True)
        sitk.WriteImage(train_img, str(results_path/f'{train_set.name}_bias_removed.nii.gz'))

        train_img = denoise(train_img)
        sitk.WriteImage(train_img, str(results_path/f'{train_set.name}_denoised.nii.gz'))
  

    for test_set in tqdm(test_path.iterdir(), desc='Test Set'):
        test_img = sitk.ReadImage(str(test_set/f'{test_set.name}.nii.gz'))
        test_img = bias_field_correction(test_img) 
        # create results folder, in this case we will overwrite the image
        results_path = test_set
        results_path.mkdir(exist_ok=True)
        
        sitk.WriteImage(test_img, str(results_path/f'{test_set.name}_bias_removed.nii.gz'))

        test_img = denoise(test_img)
        sitk.WriteImage(test_img, str(results_path/f'{test_set.name}_denoised.nii.gz'))
  

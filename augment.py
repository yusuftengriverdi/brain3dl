import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt

# Example image and mask (replace with your actual data)
image = plt.imread('path_to_your_image.jpg')
mask = plt.imread('path_to_your_mask.png')

# Define the augmentation pipeline
augmentation = A.Compose([
    A.HorizontalFlip(p=0.5),  # Adjust the probability as needed
    A.VerticalFlip(p=0.5),    # Adjust the probability as needed
    A.Rotate(limit=135, p=0.5),  # Adjust the limit and probability as needed
    # A.Normalize(),  # You can add more augmentations as needed
    ToTensorV2(),
])

# Apply the augmentation to the image and mask
augmented = augmentation(image=image, mask=mask)

# Retrieve the augmented image and mask
augmented_image = augmented['image']
augmented_mask = augmented['mask']

# Convert augmented_image and augmented_mask to torch tensors if needed
# augmented_image_tensor = torch.from_numpy(augmented_image.transpose(2, 0, 1))
# augmented_mask_tensor = torch.from_numpy(augmented_mask)

# Display the original and augmented images and masks
plt.subplot(2, 2, 1)
plt.imshow(image)
plt.title('Original Image')

plt.subplot(2, 2, 2)
plt.imshow(mask, cmap='gray')
plt.title('Original Mask')

plt.subplot(2, 2, 3)
plt.imshow(augmented_image)
plt.title('Augmented Image')

plt.subplot(2, 2, 4)
plt.imshow(augmented_mask, cmap='gray')
plt.title('Augmented Mask')

plt.show()

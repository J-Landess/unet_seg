# Dataset Structure

Place your images and corresponding segmentation masks in the following structure:

train/images/    - Training images
train/masks/     - Training segmentation masks
val/images/      - Validation images  
val/masks/       - Validation segmentation masks
test/images/     - Test images
test/masks/      - Test segmentation masks

Supported formats: .jpg, .jpeg, .png, .bmp, .tiff
Masks should be grayscale images with pixel values representing class IDs.

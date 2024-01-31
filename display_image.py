folder_path = "/mnt/fast/nobackup/users/ds01502/MajorProjectCustomizableConditioningGANS/GANSketching/output/teaser_cat_lsketch_limage_aug"  
import torch
import torchvision.transforms as transforms
import torchvision.utils as vutils
from PIL import Image
import os
# Folder containing the images
image_folder = folder_path



# Get a list of image filenames in the folder
image_files = [os.path.join(image_folder, file) for file in os.listdir(image_folder) if file.endswith(('.jpg', '.png', '.jpeg'))]

# Check if there are images in the folder
if len(image_files) == 0:
    print("No images found in the folder.")
else:
    # Create a list to store loaded images as tensors
    image_tensors = []

    # Define a transformation to convert images to tensors
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Resize images to a consistent size
        transforms.ToTensor()           # Convert images to tensors
    ])

    # Load and convert each image to a tensor
    for image_file in image_files[:50]:
        img = Image.open(image_file)
        img_tensor = transform(img)
        image_tensors.append(img_tensor)

    # Create a grid of images
    image_grid = vutils.make_grid(image_tensors, nrows=10, padding=10, normalize=True)

    # Save the image grid as a single image
    vutils.save_image(image_grid, 'n_output/teaser_cat_lsketch_limage_aug.png')


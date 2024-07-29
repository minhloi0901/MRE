from PIL import Image
import numpy as np
import argparse
import os

def resize_image(image_input):
    image = Image.open(image_input)
    
    # Resize the image to 512x512
    image = image.resize((512, 512))
    
    return image

def cut_image_into_patches(image_input, patch_size):
    # Open the image file
    image = image_input
    image_array = np.array(image)
    
    # Get image dimensions
    img_height, img_width = image_array.shape[:2]
    
    # Compute the number of patches along each dimension
    num_patches_x = img_width // patch_size
    num_patches_y = img_height // patch_size
    
    patches = []
    
    # Iterate through the image and extract patches
    for i in range(num_patches_y):
        for j in range(num_patches_x):
            top = i * patch_size
            left = j * patch_size
            bottom = top + patch_size
            right = left + patch_size
            
            # Crop the patch from the image
            patch = image_array[top:bottom, left:right]
            patches.append(patch)
    
    return patches

parse = argparse.ArgumentParser()
parse.add_argument("--image_path", type=str, required=True)
parse.add_argument("--patch_size", type=int, default=16)

if __name__ == '__main__':
    args = parse.parse_args()
    image = resize_image(args.image_path)
    os.makedirs('resized', exist_ok=True)
    image.save('resized/resized_image.png')
    
    patches = cut_image_into_patches(image, args.patch_size)

    # Save patches to foler patches/patch_idx.png
    os.makedirs('patches', exist_ok=True)
    for idx, patch in enumerate(patches):
        patch_image = Image.fromarray(patch)
        patch_image.save(f'patches/patch_{idx}.png')
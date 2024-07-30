# -*- coding: utf-8 -*-
"""
Created on Sun May 28 19:44:44 2023

@author: scorp
"""

import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
# !pip install dipy
from dipy.denoise.nlmeans import nlmeans
from dipy.denoise.noise_estimate import estimate_sigma

# Load the nii image
nii_image_swi = nib.load('images/sub-01_ses-forrestgump_anat_sub-01_ses-forrestgump_acq-mag_veno.nii')
nii_image_tof = nib.load('images/sub-01_ses-forrestgump_anat_sub-01_ses-forrestgump_angio.nii')
nii_image_t1 = nib.load('images/sub-01_ses-forrestgump_anat_sub-01_ses-forrestgump_T1w.nii')
nii_image_t2 = nib.load('images/sub-01_ses-forrestgump_anat_sub-01_ses-forrestgump_T2w.nii')
nii_image_dwi = nib.load('images/sub-01_ses-forrestgump_dwi_sub-01_ses-forrestgump_dwi.nii')
nii_image_bold = nib.load('images/sub-01_ses-forrestgump_func_sub-01_ses-forrestgump_task-forrestgump_acq-dico_run-01_bold.nii')

# Get the image data from nii_image
swi_img = nii_image_swi.get_fdata()
tof_img = nii_image_tof.get_fdata()
t1_img = nii_image_t1.get_fdata()
t2_img = nii_image_t2.get_fdata()
dwi_img = nii_image_dwi.get_fdata()
bold_img = nii_image_bold.get_fdata()

def slice_image(image_name,image):
    if image_name == "swi_img":
        # Slicing the middle z-slice for swi image and formatting it
        return np.rot90(image[:, :, image.shape[2]//2])
    elif image_name == "tof_img":
        # Slicing the middle z-slice for tof image and formatting it
        return np.fliplr(np.rot90(np.transpose(image[:, :, image.shape[2]//2]),2))
    elif image_name == "t1_img":
        # Slicing the middle z-slice for t1 image and formatting it
        return np.fliplr(np.rot90(np.transpose(image[:, :, image.shape[2]//2]),2))
    elif image_name == "t2_img":
        # Slicing the middle z-slice for t2 image and formatting it
        return np.fliplr(np.rot90(np.transpose(image[:, :, image.shape[2]//2]),2))
    elif image_name == "dwi_img":
        # Slicing the middle z-slice for dwi image and formatting it
        return np.fliplr(np.rot90(np.transpose(image[:, :, image.shape[2]//2, image.shape[3]//2]),2))
    elif image_name == "bold_img":
        # Slicing the middle z-slice for bold image and formatting it
        return np.fliplr(np.rot90(np.transpose(image[:, :, image.shape[2]//2, image.shape[3]//2]),2))


image_list = {"swi_img":swi_img, "tof_img":tof_img, "t1_img":t1_img, "t2_img":t2_img, "dwi_img":dwi_img, "bold_img":bold_img}

subplot_index=1
plt.figure(figsize=(10, 7))

for image_name,image in image_list.items():
    sigma = estimate_sigma(image)
    denoised_image = nlmeans(image, sigma=sigma)
    method_noise = image - denoised_image
    
    plt.subplot(6, 3, subplot_index)
    plt.imshow(slice_image(image_name,image), cmap='jet')
    plt.title(f'{image_name} - Original')
    plt.axis('off')
    
    subplot_index = subplot_index + 1
    plt.subplot(6, 3, subplot_index)
    plt.imshow(slice_image(image_name,denoised_image), cmap='jet')
    plt.title(f'{image_name} - Denoised Image')
    plt.axis('off')

    subplot_index = subplot_index + 1
    plt.subplot(6, 3, subplot_index)
    plt.imshow(slice_image(image_name,method_noise), cmap='jet')
    plt.title(f'{image_name} - Method Noise')
    plt.axis('off')

    subplot_index = subplot_index + 1

plt.tight_layout()
plt.show()




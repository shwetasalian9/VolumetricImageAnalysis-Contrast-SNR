# -*- coding: utf-8 -*-
"""
Created on Sun May 21 21:11:27 2023

@author: shweta salian
"""


import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

# Load the NIfTI file
nii_data = nib.load('images/sub-01_ses-forrestgump_anat_sub-01_ses-forrestgump_acq-mag_veno.nii')

# Get the image data from the NIfTI file
image1 = nii_data.get_fdata()


# Define the range of slices to consider
start_slice = 50
end_slice = 100

# Select the desired slices from the 3D image data
selected_slices = image1[start_slice:end_slice]

# Calculate the maximum intensity projection
max_projection = np.max(selected_slices, axis=2)

# Plotting the maximum intensity projection
plt.imshow(max_projection, cmap='gray', vmax=300)
plt.title('Maximum Intensity Projection (Slices {}-{})'.format(start_slice, end_slice))
plt.show()



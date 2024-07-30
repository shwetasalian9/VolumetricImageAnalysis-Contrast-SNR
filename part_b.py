# -*- coding: utf-8 -*-
"""
Created on Sun May 21 21:54:00 2023

@author: scorp
"""

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
nii_data = nib.load('images/sub-01_ses-forrestgump_anat_sub-01_ses-forrestgump_angio.nii')

# Get the image data from the NIfTI file
image2 = nii_data.get_fdata()

# Calculate the maximum intensity projections
max_projection0 = np.max(image2, axis=0)
max_projection1 = np.max(image2, axis=1)
max_projection2 = np.max(image2, axis=2)

# Determine the common vmax values for better visual comparison
vmin = np.min(image2)
vmax = np.max(image2)

# Plotting the projections
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

axes[1].imshow(max_projection0, cmap='jet', vmax=vmax)
axes[1].set_title('axis0')

# axes[1].imshow(max_projection1, cmap='jet', vmin=vmin, vmax=vmax)
# axes[1].set_title('axis1')

# axes[1].imshow(max_projection2, cmap='jet', vmin=vmin, vmax=vmax)
# axes[1].set_title('axis2')

plt.show()
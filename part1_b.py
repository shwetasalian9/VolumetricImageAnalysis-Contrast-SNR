

import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib

# Load the nii image
nii_image_swi = nib.load('images/sub-01_ses-forrestgump_anat_sub-01_ses-forrestgump_acq-mag_veno.nii')
nii_image_tof = nib.load('images/sub-01_ses-forrestgump_anat_sub-01_ses-forrestgump_angio.nii')

# Get the image data from nii_image
swi_img = nii_image_swi.get_fdata()
tof_img = nii_image_tof.get_fdata()


# Plot the figure
fig = plt.figure(figsize=(10, 7))


# Plot maximum intensity projections for tof image
fig.add_subplot(2, 3, 1)
plt.imshow(np.rot90(np.max(tof_img, axis=0)), cmap='jet', vmax=300)
plt.title('tof (saggital view)')

fig.add_subplot(2, 3, 2)
plt.imshow(np.rot90(np.max(tof_img, axis=1)), cmap='jet', vmax=300)
plt.title('tof (coronal view)')

fig.add_subplot(2, 3, 3)
plt.imshow(np.rot90(np.max(tof_img, axis=2)), cmap='jet', vmax=300)
plt.title('tof (axial view)')


# Plot minimum intensity projections for swi image
fig.add_subplot(2, 3, 4)
plt.imshow(np.rot90(np.min(swi_img[200:300,:,:], axis=0)), cmap='jet')
plt.title('swi (saggital view, slices 200:300)')

fig.add_subplot(2, 3, 5)
plt.imshow(np.rot90(np.min(swi_img[:,200:300,:], axis=1)), cmap='jet')
plt.title('swi (coronal view, slices 200:300)')

fig.add_subplot(2, 3, 6)
plt.imshow(np.rot90(np.min(swi_img[:,:,225:300], axis=2)), cmap='jet')
plt.title('swi (axial view, slices 225:300)')


plt.tight_layout()
plt.show()

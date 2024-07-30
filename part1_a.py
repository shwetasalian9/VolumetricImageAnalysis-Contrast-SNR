
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np

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


# Slicing the middle z-slice for swi image and formatting it
swi_slice_image = np.rot90(swi_img[:, :, swi_img.shape[2]//2])

# Slicing the middle z-slice for tof image and formatting it
tof_slice_image =  np.fliplr(np.rot90(np.transpose(tof_img[:, :, tof_img.shape[2]//2]),2))

# Slicing the middle z-slice for t1 image and formatting it
t1_slice_image = np.fliplr(np.rot90(np.transpose(t1_img[:, :, t1_img.shape[2]//2]),2))

# Slicing the middle z-slice for t2 image and formatting it
t2_slice_image = np.fliplr(np.rot90(np.transpose(t2_img[:, :, t2_img.shape[2]//2]),2))

# Slicing the middle z-slice for dwi image and formatting it
dwi_slice_image = np.fliplr(np.rot90(np.transpose(dwi_img[:, :, dwi_img.shape[2]//2, dwi_img.shape[3]//2]),2))

# Slicing the middle z-slice for bold image and formatting it
bold_slice_image =  np.fliplr(np.rot90(np.transpose(bold_img[:, :, bold_img.shape[2]//2, bold_img.shape[3]//2]),2))

# Plot the figure
fig = plt.figure(figsize=(10, 7))

# Plotting swi
fig.add_subplot(2, 3, 1)
plt.imshow(swi_slice_image, cmap='jet')
plt.title('swi')

# Plotting tof
fig.add_subplot(2, 3, 2)
plt.imshow(tof_slice_image, cmap='jet')
plt.title('tof')

# Plotting t1
fig.add_subplot(2, 3, 3)
plt.imshow(t1_slice_image, cmap='jet')
plt.title('t1')

# Plotting t2
fig.add_subplot(2, 3, 4)
plt.imshow(t2_slice_image, cmap='jet')
plt.title('t2')

# Plotting dwi
fig.add_subplot(2, 3, 5)
plt.imshow(dwi_slice_image, cmap='jet')
plt.title('dwi')

# Plotting bold
fig.add_subplot(2, 3, 6)
plt.imshow(bold_slice_image, cmap='jet')
plt.title('bold')


plt.tight_layout()
plt.show()



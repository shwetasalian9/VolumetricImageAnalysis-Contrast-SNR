# -*- coding: utf-8 -*-
"""
Created on Mon May 29 20:40:33 2023

@author: scorp
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fftn, ifftn, fftshift, ifftshift
import nibabel as nib

# Load the nii image
nii_image_swi = nib.load('images/sub-01_ses-forrestgump_anat_sub-01_ses-forrestgump_acq-mag_veno.nii')
nii_image_t1 = nib.load('images/sub-01_ses-forrestgump_anat_sub-01_ses-forrestgump_T1w.nii')
nii_image_t2 = nib.load('images/sub-01_ses-forrestgump_anat_sub-01_ses-forrestgump_T2w.nii')
nii_image_dwi = nib.load('images/sub-01_ses-forrestgump_dwi_sub-01_ses-forrestgump_dwi.nii')
nii_image_bold = nib.load('images/sub-01_ses-forrestgump_func_sub-01_ses-forrestgump_task-forrestgump_acq-dico_run-01_bold.nii')


# Get the image data from nii_image
swi_data = nii_image_swi.get_fdata()
t1_data = nii_image_t1.get_fdata()
t2_data = nii_image_t2.get_fdata()
dwi_data = nii_image_dwi.get_fdata()
bold_img = nii_image_bold.get_fdata()


# Extract the first three dimensions for downsampling and filtering
bold_data = bold_img[..., 0]

# Define the desired downsampled resolution
target_resolution = (40, 40, 10)

# Compute the scale factors for each dimension
scale_factors = np.array(bold_data.shape) / np.array(target_resolution)

# Downsample the image using NumPy's indexing and interpolation
downsampled_data = bold_data[::int(scale_factors[0]), ::int(scale_factors[1]), ::int(scale_factors[2])]

# Define the 3D Gaussian kernel function
def gaussian_kernel(shape, sigma):
    kernel = np.zeros(shape)
    center_coords = [(coord - 1) / 2 for coord in shape]
    grid = np.meshgrid(*[np.arange(-coord, coord + 1) for coord in center_coords], indexing='ij')
    kernel = np.exp(-(grid[0] ** 2 + grid[1] ** 2 + grid[2] ** 2) / (2 * sigma ** 2))
    kernel /= np.sum(kernel)
    return kernel

# Apply linear filtering for different sigma values and show the middle z-slice of the filtered image
sigmas = [2, 4, 15]

fig, axes = plt.subplots(1, len(sigmas), figsize=(12, 4))

for i, sigma in enumerate(sigmas):
    # Generate the Gaussian kernel with the same shape as the downsampled image
    kernel_shape = downsampled_data.shape
    kernel = gaussian_kernel(kernel_shape, sigma)
    
    # Apply Fourier transform to the image and the kernel (with fftshift)
    image_fft = fftshift(fftn(downsampled_data))
    kernel_fft = fftshift(fftn(kernel, kernel_shape))
    
    # Apply linear filtering by multiplying the Fourier transformed image with the kernel
    filtered_image_fft = image_fft * kernel_fft
    
    # Perform inverse Fourier transform to obtain the filtered image (with ifftshift)
    filtered_image = np.real(ifftn(ifftshift(filtered_image_fft)))
    
    # Ensure the filtered image has the same shape as the input image
    filtered_image = filtered_image[:downsampled_data.shape[0], :downsampled_data.shape[1], :downsampled_data.shape[2]]
    
    # Get the middle z-slice index
    middle_slice = filtered_image.shape[2] // 2
    
    # Display the middle z-slice of the filtered image
    axes[i].imshow(filtered_image[..., middle_slice], cmap='gray', vmin=np.min(filtered_image), vmax=np.max(filtered_image))
    axes[i].set_title(f'Bold Image Sigma = {sigma}')
    axes[i].axis('off')

plt.tight_layout()
plt.show()


target_resolution = (40, 40, 9)
scale_factors = np.array(t1_data.shape) / np.array(target_resolution)
downsampled_data = t1_data[::int(scale_factors[0]), ::int(scale_factors[1]), ::int(scale_factors[2])]

def gaussian_kernel(shape, sigma):
    kernel = np.zeros(shape)
    center_coords = [(coord - 1) / 2 for coord in shape]
    grid = np.meshgrid(*[np.arange(-coord, coord + 1) for coord in center_coords], indexing='ij')
    kernel = np.exp(-(grid[0] ** 2 + grid[1] ** 2 + grid[2] ** 2) / (2 * sigma ** 2))
    kernel /= np.sum(kernel)
    return kernel

sigmas = [2, 4, 15]

fig, axes = plt.subplots(1, len(sigmas), figsize=(12, 4))

for i, sigma in enumerate(sigmas):
    kernel_shape = downsampled_data.shape
    kernel = gaussian_kernel(kernel_shape, sigma)
    
    image_fft = fftshift(fftn(downsampled_data))
    kernel_fft = fftshift(fftn(kernel, kernel_shape))
    
    filtered_image_fft = image_fft * kernel_fft
    
    filtered_image = np.real(ifftn(ifftshift(filtered_image_fft)))
    
    filtered_image = filtered_image[:downsampled_data.shape[0], :downsampled_data.shape[1], :downsampled_data.shape[2]]
    
    middle_slice = filtered_image.shape[2] // 2
    
    filtered_image = np.clip(filtered_image, 0, 255)
    
    axes[i].imshow(filtered_image[..., middle_slice], cmap='gray')
    axes[i].set_title(f'T1 Image Sigma = {sigma}')
    axes[i].axis('off')

plt.tight_layout()
plt.show()


# Define the desired downsampled resolution
target_resolution = (40, 40,10)

# Compute the scale factors for each dimension
scale_factors = np.array(t2_data.shape) / np.array(target_resolution)

# Downsample the image using NumPy's indexing and interpolation
downsampled_data = t2_data[::int(scale_factors[0]), ::int(scale_factors[1]), ::int(scale_factors[2])]

# Define the 3D Gaussian kernel function
def gaussian_kernel(shape, sigma):
    kernel = np.zeros(shape)
    center_coords = [(coord - 1) / 2 for coord in shape]
    grid = np.meshgrid(*[np.arange(-coord, coord + 1) for coord in center_coords], indexing='ij')
    kernel = np.exp(-(grid[0] ** 2 + grid[1] ** 2 + grid[2] ** 2) / (2 * sigma ** 2))
    kernel /= np.sum(kernel)
    return kernel

# Apply linear filtering for different sigma values and show the middle z-slice of the filtered image
sigmas = [2, 4, 15]

fig, axes = plt.subplots(1, len(sigmas), figsize=(12, 4))

for i, sigma in enumerate(sigmas):
    # Generate the Gaussian kernel
    kernel = gaussian_kernel(downsampled_data.shape, sigma)
    
    # Apply Fourier transform to the image and the kernel
    image_fft = fftn(downsampled_data)
    kernel_fft = fftn(kernel)
    
    # Apply linear filtering by multiplying the Fourier transformed image with the kernel
    filtered_image_fft = image_fft * kernel_fft

    # Perform inverse Fourier transform to obtain the filtered image
    filtered_image = np.real(ifftn(filtered_image_fft))
    
    # Ensure the filtered image has the same shape as the input image
    filtered_image = filtered_image[:downsampled_data.shape[0], :downsampled_data.shape[1], :downsampled_data.shape[2]]
    
    # Get the middle z-slice index
    middle_slice = filtered_image.shape[2] // 2
    
    # Display the middle z-slice of the filtered image
    axes[i].imshow(filtered_image[..., middle_slice], cmap='gray', vmin=np.min(filtered_image), vmax=np.max(filtered_image))
    axes[i].set_title(f'T2 Image Sigma = {sigma}')
    axes[i].axis('off')

plt.tight_layout()
plt.show()


# Load the image
image_path = 'images/sub-01_ses-forrestgump_anat_sub-01_ses-forrestgump_angio.nii'
image1 = nib.load(image_path)
tof_data = image1.get_fdata()


# Ensure the data has three dimensions
if tof_data.ndim == 4:
    tof_data = tof_data[..., 0]

# Define the desired downsampled resolution
target_resolution = (40, 40, 9)

# Compute the scale factors for each dimension
scale_factors = np.array(tof_data.shape[:3]) / np.array(target_resolution)

# Downsample the image using numpy's indexing and interpolation
downsampled_data = tof_data[::int(scale_factors[0]), ::int(scale_factors[1]), ::int(scale_factors[2])]

# Define the 3D Gaussian kernel function
def gaussian_kernel(shape, sigma):
    kernel = np.zeros(shape)
    center_coords = [(coord - 1) / 2 for coord in shape]
    grid = np.meshgrid(*[np.arange(-coord, coord + 1) for coord in center_coords], indexing='ij')
    kernel = np.exp(-(grid[0] ** 2 + grid[1] ** 2 + grid[2] ** 2) / (2 * sigma ** 2))
    kernel /= np.sum(kernel)
    return kernel

# Apply linear filtering for different sigma values and show the middle z-slice of the filtered image
sigmas = [2, 4, 15]

fig, axes = plt.subplots(1, len(sigmas), figsize=(12, 4))

for i, sigma in enumerate(sigmas):
    # Generate the Gaussian kernel
    kernel = gaussian_kernel(downsampled_data.shape, sigma)
    
    # Apply Fourier transform to the image and the kernel
    image_fft = fftn(downsampled_data)
    kernel_fft = fftn(kernel)
    
    # Apply fftshift to the Fourier transformed image and kernel
    image_fft_shifted = fftshift(image_fft)
    kernel_fft_shifted = fftshift(kernel_fft)
    
    # Apply linear filtering by multiplying the shifted Fourier transformed image with the shifted kernel
    filtered_image_fft_shifted = image_fft_shifted * kernel_fft_shifted

    # Perform inverse fftshift on the filtered image
    filtered_image_fft = ifftshift(filtered_image_fft_shifted)
    
    # Perform inverse Fourier transform to obtain the filtered image
    filtered_image = np.real(ifftn(filtered_image_fft))
    
    # Ensure the filtered image has the same shape as the input image
    filtered_image = filtered_image[:downsampled_data.shape[0], :downsampled_data.shape[1], :downsampled_data.shape[2]]
    
    # Get the middle z-slice index
    middle_slice = downsampled_data.shape[2] // 2
    
    # Display the middle z-slice of the filtered image
    axes[i].imshow(filtered_image[..., middle_slice], cmap='gray', vmin=np.min(filtered_image), vmax=np.max(filtered_image))
    axes[i].set_title(f'TOF Image Sigma = {sigma}')
    axes[i].axis('off')

plt.tight_layout()
plt.show()


# Define the desired downsampled resolution
target_resolution = (40, 40, 9)

# Compute the scale factors for each dimension
scale_factors = np.array(swi_data.shape[:3]) / np.array(target_resolution)

# Downsample the image using numpy's indexing and interpolation
downsampled_data = swi_data[::int(scale_factors[0]), ::int(scale_factors[1]), ::int(scale_factors[2])]

# Define the 3D Gaussian kernel function
def gaussian_kernel(shape, sigma):
    kernel = np.zeros(shape)
    center_coords = [(coord - 1) / 2 for coord in shape]
    grid = np.meshgrid(*[np.arange(-coord, coord + 1) for coord in center_coords], indexing='ij')
    kernel = np.exp(-(grid[0] ** 2 + grid[1] ** 2 + grid[2] ** 2) / (2 * sigma ** 2))
    kernel /= np.sum(kernel)
    return kernel

# Apply linear filtering for different sigma values and show the middle z-slice of the filtered image
sigmas = [2, 4, 15]

fig, axes = plt.subplots(1, len(sigmas), figsize=(12, 4))

for i, sigma in enumerate(sigmas):
    # Generate the Gaussian kernel
    kernel = gaussian_kernel(downsampled_data.shape, sigma)
    
    # Apply Fourier transform to the image and the kernel
    image_fft = fftn(downsampled_data)
    kernel_fft = fftn(kernel)
    
    # Apply linear filtering by multiplying the Fourier transformed image with the kernel
    filtered_image_fft = image_fft * kernel_fft

    # Perform inverse Fourier transform to obtain the filtered image
    filtered_image = np.real(ifftn(filtered_image_fft))
    
    # Ensure the filtered image has the same shape as the input image
    filtered_image = filtered_image[:downsampled_data.shape[0], :downsampled_data.shape[1], :downsampled_data.shape[2]]
    
    # Get the middle z-slice index
    middle_slice = downsampled_data.shape[2] // 2
    
    # Display the middle z-slice of the filtered image
    axes[i].imshow(filtered_image[..., middle_slice], cmap='gray', vmin=np.min(filtered_image), vmax=np.max(filtered_image))
    axes[i].set_title(f'SWI Image Sigma = {sigma}')
    axes[i].axis('off')

plt.tight_layout()
plt.show()


# Define the 3D Gaussian kernel function
def gaussian_kernel(shape, sigma):
    kernel = np.zeros(shape)
    center_coords = [(coord - 1) / 2 for coord in shape]
    grid = np.meshgrid(*[np.arange(-coord, coord + 1) for coord in center_coords], indexing='ij')
    kernel = np.exp(-(grid[0] ** 2 + grid[1] ** 2 + grid[2] ** 2) / (2 * sigma ** 2))
    kernel /= np.sum(kernel)
    return kernel

# Apply linear filtering for different sigma values and show the middle z-slice of the filtered image
sigmas = [2, 4, 15]

fig, axes = plt.subplots(1, len(sigmas), figsize=(12, 4))

for i, sigma in enumerate(sigmas):
    # Generate the Gaussian kernel
    kernel = gaussian_kernel(dwi_data.shape, sigma)
    
    # Apply Fourier transform to the image and the kernel
    image_fft = fftn(dwi_data)
    kernel_fft = fftn(kernel)
    
    # Apply linear filtering by multiplying the Fourier transformed image with the kernel
    filtered_image_fft = image_fft * kernel_fft

    # Perform inverse Fourier transform to obtain the filtered image
    filtered_image = np.real(ifftn(filtered_image_fft))
    
    # Ensure the filtered image has the same shape as the input image
    filtered_image = filtered_image[:dwi_data.shape[0], :dwi_data.shape[1], :dwi_data.shape[2], :dwi_data.shape[3]]
    
    # Get the middle z-slice index
    middle_slice = dwi_data.shape[2] // 2
    
    # Display the middle z-slice of the filtered image
    axes[i].imshow(filtered_image[..., middle_slice, 0], cmap='gray', vmin=np.min(filtered_image), vmax=np.max(filtered_image))
    axes[i].set_title(f'DWI Image Sigma = {sigma}')
    axes[i].axis('off')

plt.tight_layout()
plt.show()

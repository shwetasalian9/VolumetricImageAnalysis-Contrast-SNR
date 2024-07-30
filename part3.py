# -*- coding: utf-8 -*-
"""
Created on Mon May 29 17:59:47 2023

@author: scorp
"""

import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib

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

image_list = {"swi_img":swi_img, "tof_img":tof_img, "t1_img":t1_img, "t2_img":t2_img, "dwi_img":dwi_img, "bold_img":bold_img}


def plot_image(image,title):
    # Plotting the image
    plt.imshow(image)
    plt.title(title)
    
    
def plot_signal_patch(image,signal_patch,plt1,plt2, text_x, text_y):
    # Plotting the signal patch
    signal_patch_data = image[signal_patch]
    plt.imshow(signal_patch_data, cmap='viridis', alpha=0.5)
    plt.plot(plt1, plt2, 'r-', linewidth=2)  # Rectangle border
    plt.text(text_x, text_y, 'Signal', color='red', fontsize=12)
    return signal_patch_data
    
def plot_noise_patch(image,noise_patch,plt1,plt2, text_x, text_y):
    # Plotting the noise patch
    noise_patch_data = image[noise_patch]
    plt.imshow(noise_patch_data, cmap='cool', alpha=0.5)
    plt.plot(plt1, plt2, 'b-', linewidth=2)  # Rectangle border
    plt.text(text_x, text_y, 'Noise', color='blue', fontsize=12)
    return noise_patch_data
    
def calculate_SNR_formula(signal_patch_data, noise_patch_data):
    signal_mean = np.mean(signal_patch_data)
    noise_std = np.std(noise_patch_data)   
    snr = signal_mean / noise_std    
    return snr

def print_coordinates(image_name,signal_patch,noise_patch):
    # Printing the coordinates of signal and noise patches
    signal_coords = np.array(signal_patch)
    noise_coords = np.array(noise_patch)
    print(f'Signal Patch Coordinates for image {image_name} :')
    print('X:', signal_coords[0], 'Y:', signal_coords[1], 'Z:', signal_coords[2])
    print(f'Noise Patch Coordinates for image {image_name} :')
    print('X:', noise_coords[0], 'Y:', noise_coords[1], 'Z:', noise_coords[2])

def calculate_SNR(image_name,image):
    if image_name == "swi_img":       
        # Defining the signal and noise patches
        signal_patch = np.s_[190:240, 140:220, 250]
        noise_patch = np.s_[90:110, 340:360, 250] 
        print_coordinates(image_name,signal_patch,noise_patch)
        signal_patch_data = plot_signal_patch(swi_img,signal_patch, [140, 140, 219, 219, 140], [190, 239, 239, 190, 190], 140, 188)
        noise_patch_data = plot_noise_patch(swi_img, noise_patch, [340, 340, 359, 359, 340], [90, 109, 109, 90, 90], 340, 88)
        snr = round(calculate_SNR_formula(signal_patch_data,noise_patch_data),2)
        print(f'SNR for {image_name} is {snr}')
        plot_image(swi_img[:, :, 250],f'SW1 (acq-mag_veno) Image, snr = {snr}')
        
    elif image_name == "tof_img":       
        # Defining the signal and noise patches
        signal_patch = np.s_[110:145, 102:110, 80]
        noise_patch = np.s_[416:424, 532:550, 80]
        print_coordinates(image_name,signal_patch,noise_patch)
        signal_patch_data = plot_signal_patch(tof_img,signal_patch, [102, 102, 109, 109, 102], [110, 144, 144, 110, 110], 102, 108)
        noise_patch_data = plot_noise_patch(tof_img, noise_patch, [532, 532, 549, 549, 532], [416, 423, 423, 416, 416], 532, 414)
        snr = round(calculate_SNR_formula(signal_patch_data,noise_patch_data),2)
        print(f'SNR for {image_name} is {snr}')
        plot_image(tof_img[:, :, 80],f'TOF Image, snr = {snr}')
        
    elif image_name == "t1_img":       
        # Defining the signal and noise patches
        signal_patch = np.s_[70:95, 90:115, 250]
        noise_patch = np.s_[223:228, 261:267, 250] 
        print_coordinates(image_name,signal_patch,noise_patch)
        signal_patch_data = plot_signal_patch(t1_img,signal_patch, [90, 90, 115, 115, 90], [70, 95, 95, 70, 70], 90, 68)
        noise_patch_data = plot_noise_patch(t1_img, noise_patch, [261, 261, 267, 267, 261], [223, 228, 228, 223, 223], 261, 221)
        snr = round(calculate_SNR_formula(signal_patch_data,noise_patch_data),2)
        print(f'SNR for {image_name} is {snr}')
        plot_image(t1_img[:, :, 250],f'T1-weighted Image, snr = {snr}')
        
    elif image_name == "t2_img":       
        # Defining the signal and noise patches
        signal_patch = np.s_[85:115, 190:230, 192]
        noise_patch = np.s_[38:50, 264:278, 192]
        print_coordinates(image_name,signal_patch,noise_patch)
        signal_patch_data = plot_signal_patch(t2_img,signal_patch, [190, 190, 229, 229, 190], [85, 114, 114, 85, 85], 190, 83)
        noise_patch_data = plot_noise_patch(t2_img, noise_patch, [264, 264, 277, 277, 264], [38, 49, 49, 38, 38], 264, 36)
        snr = round(calculate_SNR_formula(signal_patch_data,noise_patch_data),2)
        print(f'SNR for {image_name} is {snr}')
        plot_image(t1_img[:, :, 250],f'T2-weighted Image, snr = {snr}')
        
    elif image_name == "dwi_img":       
        # Defining the signal and noise patches
        signal_patch = np.s_[48:54, 102:110, 35, 25]
        noise_patch = np.s_[103:106, 107:111, 35, 25]
        print_coordinates(image_name,signal_patch,noise_patch)
        signal_patch_data = plot_signal_patch(dwi_img,signal_patch, [102, 102, 109, 109, 102], [48, 53, 53, 48, 48], 102, 46)
        noise_patch_data = plot_noise_patch(dwi_img, noise_patch, [107, 107, 110, 110, 107], [103, 105, 105, 103, 103], 107, 101)
        snr = round(calculate_SNR_formula(signal_patch_data,noise_patch_data),2)
        print(f'SNR for {image_name} is {snr}')
        plot_image(dwi_img[:, :, 35, 25],f'DWI Image, snr = {snr}')
        
    elif image_name == "bold_img":       
        # Defining the signal and noise patches
        signal_patch = np.s_[70:78, 122:130, 18, 25]
        noise_patch = np.s_[111:116, 140:143, 18, 25]
        print_coordinates(image_name,signal_patch,noise_patch)
        signal_patch_data = plot_signal_patch(bold_img,signal_patch, [122, 122, 129, 129, 122], [70, 77, 77, 70, 70], 122, 68)
        noise_patch_data = plot_noise_patch(bold_img, noise_patch, [140, 140, 142, 142, 140], [111, 115, 115, 111, 111], 140, 109)
        snr = round(calculate_SNR_formula(signal_patch_data,noise_patch_data),2)
        print(f'SNR for {image_name} is {snr}')
        plot_image(bold_img[:, :, 18, 25],f'BOLD Image, snr = {snr}')
        

        
for image_name,image in image_list.items():
    calculate_SNR(image_name,image)  
    plt.tight_layout()
    plt.show()
    
    
def plot_noise_histogram(noise_patch_data,title):
    # Plot the histogram of the noise
    plt.hist(noise_patch_data.ravel(), bins=50, color='b', alpha=0.7)
    plt.title(title)
    plt.xlabel('Intensity')
    plt.ylabel('Frequency')
    plt.show()
    
for image_name,image in image_list.items():
    if image_name == "swi_img":    
        # Define the noise patch
        noise_patch = np.s_[90:110, 340:360, 250]
        # Extract the noise data
        noise_patch_data = swi_img[noise_patch]
        plot_noise_histogram(noise_patch_data,'Histogram of Noise SW1')
    elif image_name == "tof_img":
        # Define the noise patch
        noise_patch = np.s_[416:424, 532:550, 80]
        # Extract the noise data
        noise_patch_data = tof_img[noise_patch]
        plot_noise_histogram(noise_patch_data,'Histogram of Noise TOF')
    elif image_name == "t1_img":
        # Define the noise patch
        noise_patch = np.s_[223:228, 261:267, 250]
        # Extract the noise data
        noise_patch_data = t1_img[noise_patch]
        plot_noise_histogram(noise_patch_data,'Histogram of Noise T1')
    elif image_name == "t2_img":
        # Define the noise patch
        noise_patch = np.s_[38:50, 264:278, 192]
        # Extract the noise data
        noise_patch_data = t2_img[noise_patch]
        plot_noise_histogram(noise_patch_data,'Histogram of Noise T2')
    elif image_name == "dwi_img":
        # Define the noise patch
        noise_patch = np.s_[103:106, 107:111, 35, 25]
        # Extract the noise data
        noise_patch_data = dwi_img[noise_patch]
        plot_noise_histogram(noise_patch_data,'Histogram of Noise DWI')
    elif image_name == "bold_img":
        # Define the noise patch
        noise_patch = np.s_[111:116, 140:143, 18, 25]
        # Extract the noise data
        noise_patch_data = bold_img[noise_patch]
        plot_noise_histogram(noise_patch_data,'Histogram of Noise BOLD')
        
        
    
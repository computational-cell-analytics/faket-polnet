import pandas as pd
import json
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm
import sys
import numpy as np
import shutil
import random
import time
import matplotlib.pyplot as plt
#import napari
import mrcfile
from skimage.metrics import structural_similarity as ssim, mean_squared_error

def transform_directory_structure(source_dir, target_dir_faket, target_dir_basic, copy_flag = True):
    """
    THIS FUNCTION IS DEPRECATED.

    Move tomogram files into the correct directory structure, preserving parent dir names.

    Parameters:
        source_dir (str): Path to the source directory containing reconstructed tomograms.
        target_dir_faket (str): Path to the target directory for faket tomograms.
        target_dir_basic (str): Path to the target directory for basic tomograms.
    """
    
    for folder in sorted(os.listdir(source_dir),key=lambda x: (int(x.split('_')[1]), int(x.split('_')[2]))):
        folder_path = os.path.join(source_dir, folder)
        if not os.path.isdir(folder_path):
            continue

        # Use the original folder name (e.g., tomogram_1_5)
        for file in os.listdir(folder_path):
            source_file_path = os.path.join(folder_path, file)

            if file.endswith("_faket.mrc"):
                # Preserve parent dir name
                new_folder_path = os.path.join(target_dir_faket, folder)
                os.makedirs(new_folder_path, exist_ok=True)
                new_file_path = os.path.join(new_folder_path, file)
                if copy_flag:
                    shutil.copy(source_file_path, new_file_path)
                else:
                    shutil.move(source_file_path, new_file_path)
                print(f"Moved (faket): {source_file_path} → {new_file_path}")

            elif file.endswith(".mrc") and not file.endswith("_faket.mrc"):
                new_folder_path = os.path.join(target_dir_basic, folder)
                os.makedirs(new_folder_path, exist_ok=True)
                new_file_path = os.path.join(new_folder_path, file)
                if copy_flag:
                    shutil.copy(source_file_path, new_file_path)
                else:
                    shutil.move(source_file_path, new_file_path)
                print(f"Moved (basic): {source_file_path} → {new_file_path}")

def collect_results_to_train_dir(source_dir, target_dir_faket, target_dir_basic, simulation_index, copy_flag = True):
    """
    Updated version of `transform_directory_structure`.
    Move tomogram files into the correct directory structure, preserving parent dir names.

    Parameters:
        source_dir (str): Path to the source directory containing reconstructed tomograms.
        target_dir_faket (str): Path to the target directory for faket tomograms.
        target_dir_basic (str): Path to the target directory for basic tomograms.
    """
    
    for folder in sorted(os.listdir(source_dir),key=lambda x: (int(x.split('_')[1]), int(x.split('_')[2]))):
        folder_path = os.path.join(source_dir, folder)
        if not os.path.isdir(folder_path):
            continue

        # Use the original folder name (e.g., tomogram_1_5)
        for file in os.listdir(folder_path):
            source_file_path = os.path.join(folder_path, file)

            if file.endswith("_faket.mrc"):
                # Preserve parent dir name
                new_folder_path = os.path.join(target_dir_faket, folder)
                os.makedirs(new_folder_path, exist_ok=True)
                new_file_path = os.path.join(new_folder_path, file)
                if copy_flag:
                    shutil.copy(source_file_path, new_file_path)
                else:
                    shutil.move(source_file_path, new_file_path)
                print(f"Moved (faket): {source_file_path} → {new_file_path}")

            elif file.endswith(".mrc") and not file.endswith("_faket.mrc"):
                new_folder_path = os.path.join(target_dir_basic, folder)
                os.makedirs(new_folder_path, exist_ok=True)
                new_file_path = os.path.join(new_folder_path, file)
                if copy_flag:
                    shutil.copy(source_file_path, new_file_path)
                else:
                    shutil.move(source_file_path, new_file_path)
                print(f"Moved (basic): {source_file_path} → {new_file_path}")

def get_absolute_paths(parent_dir):
    """
    Get absolute paths of all directories inside a given directory.
    
    Parameters:
        parent_dir (str): Path to the parent directory.
    
    Returns:
        list: A list of absolute paths of subdirectories.
    """
    return [os.path.abspath(os.path.join(parent_dir, d)) for d in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, d))]


def visualize_tomograms(tomogram_paths):
    """
    Visualize reconstructed tomograms using Napari.

    Parameters:
        tomogram_paths (list): List of paths to the tomogram files (.mrc) to visualize.
    """
    viewer = napari.Viewer()

    for tomo_path in tomogram_paths:
        if not os.path.exists(tomo_path):
            print(f"File not found: {tomo_path}")
            continue

        # Load the tomogram
        with mrcfile.open(tomo_path, permissive=True) as mrc:
            tomo_data = np.copy(mrc.data)

        # Add the tomogram to the Napari viewer
        viewer.add_image(
            tomo_data,
            name=os.path.basename(tomo_path),
            colormap="gray",
            contrast_limits=(tomo_data.min(), tomo_data.max()),
        )
        print(f"Loaded tomogram: {tomo_path}")

    # Start the Napari event loop
    napari.run()

def load_mrc(fname, mmap=False, no_saxes=True):
    """
    Load an input MRC tomogram as ndarray

    :param fname: the input MRC
    :param mmap: if True (default False) the data are read as a memory map
    :param no_saxes: if True (default) then X and Y axes are swaped to cancel the swaping made by mrcfile package
    :return: a ndarray (or memmap is mmap=True)
    """
    if mmap:
        mrc = mrcfile.mmap(fname, permissive=True, mode='r+')
    else:
        mrc = mrcfile.open(fname, permissive=True, mode='r+')
    if no_saxes:
        return np.swapaxes(mrc.data, 0, 2)
    return mrc.data


def center_crop(arr, target_shape):
    z, y, x = arr.shape
    tz, ty, tx = target_shape
    startz = (z - tz) // 2
    starty = (y - ty) // 2
    startx = (x - tx) // 2
    return arr[startz:startz+tz, starty:starty+ty, startx:startx+tx]

def compare_tomograms(tomo1_path, tomo2_path,MSE= True,SSIM = True):
    """
    Compare two tomograms and calculate MSE and SSIM.

    Parameters:
        tomo1_path (str): Path to the first tomogram file (e.g., faket tomogram).
        tomo2_path (str): Path to the second tomogram file (e.g., basic tomogram).

    Returns:
        dict: A dictionary containing MSE, SSIM, and the difference array.
    """
    # Load the tomograms
    with mrcfile.open(tomo1_path, permissive=True) as mrc:
        tomo1 = np.copy(mrc.data)
    with mrcfile.open(tomo2_path, permissive=True) as mrc:
        tomo2 = np.copy(mrc.data)

    # Ensure the tomograms have the same shape
    if tomo1.shape != tomo2.shape:
        print(f"Shape mismatch: {tomo1.shape} vs {tomo2.shape}, cropping larger to match smaller.")
        if np.prod(tomo1.shape) > np.prod(tomo2.shape):
            tomo1 = center_crop(tomo1, tomo2.shape)
        else:
            tomo2 = center_crop(tomo2, tomo1.shape)

    # Calculate MSE
    if MSE:
        mse_val = mean_squared_error(tomo1, tomo2)
        print(f"MSE: {mse_val:.4f}")
    # Calculate SSIM
    if SSIM:
        ssim_val = ssim(tomo1, tomo2, data_range=tomo2.max() - tomo2.min())
        print(f"SSIM: {ssim_val:.4f}")
    # Calculate the difference
    difference = tomo1 - tomo2

    # Display the results
    print(f"tomo1_path is {tomo1_path}")
    print(f"tomo2_path is {tomo2_path}")
    tomo_id_1 = tomo1_path.split("/")[-1].split(".")[0]
    tomo_id_2 = tomo2_path.split("/")[-1].split(".")[0]
    print(f"Tomogram IDs: {tomo_id_1} vs {tomo_id_2}")
    print(f"Tomogram shapes: {tomo1.shape} vs {tomo2.shape}")
    # Visualize the difference
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    plt.title(f"Tomogram_{tomo_id_1}")
    plt.imshow(tomo1[tomo1.shape[0] // 2], cmap="gray")
    plt.colorbar()

    plt.subplot(1, 3, 2)
    plt.title(f"Tomogram_{tomo_id_2}")
    plt.imshow(tomo2[tomo2.shape[0] // 2], cmap="gray")
    plt.colorbar()

    plt.subplot(1, 3, 3)
    plt.title("Difference (Tomo1 - Tomo2)")
    plt.imshow(difference[tomo1.shape[0] // 2], cmap="seismic", vmin=-np.max(np.abs(difference)), vmax=np.max(np.abs(difference)))
    plt.colorbar()

    plt.tight_layout()
    plt.show()

def copy_style_micrographs(source_dir, destination_dir,copy_flag = False):
    """
    Traverse the source directory, find style micrograph files, and copy them to the destination directory.

    Parameters:
        source_dir (str): The base directory containing the style micrographs.
        destination_dir (str): The directory where the selected files will be copied.
    """
    # Ensure the destination directory exists
    os.makedirs(destination_dir, exist_ok=True)

    # Traverse the source directory
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            # Check if the file matches the pattern *_style_mics.mrc
            if file.endswith("_style_mics.mrc"):
                source_file_path = os.path.join(root, file)
                destination_file_path = os.path.join(destination_dir, file)
                if copy_flag:
                    # Copy the file to the destination directory
                    shutil.copy2(source_file_path, destination_file_path)
                    print(f"Copied: {source_file_path} → {destination_file_path}")
                else:
                    # Copy the file to the destination directory
                    shutil.move(source_file_path, destination_file_path)
                    print(f"Moved: {source_file_path} → {destination_file_path}")

def check_mrc_files(directory,file_threshold = 5):
    """Scans a directory for MRC files and prints their shape and size."""
    mrc_files = [f for f in os.listdir(directory) if f.endswith(".mrc")]
    
    if not mrc_files:
        print("No MRC files found in the directory.")
        return

    print(f"Found {len(mrc_files)} MRC files in {directory}:\n")
    file_count = 0
    for mrc_file in mrc_files:
        mrc_path = os.path.join(directory, mrc_file)
        file_size = os.path.getsize(mrc_path) / (1024 * 1024)  # Convert bytes to MB

        try:
            with mrcfile.open(mrc_path, permissive=True) as mrc:
                shape = mrc.data.shape  # (Z, Y, X)
                dtype = mrc.data.dtype
                voxel_size = mrc.voxel_size  # Gives voxel spacing in Ångströms
                print(f"Voxel spacing: {voxel_size} Å/voxel")
                print(f"File: {mrc_file}")
                print(f"  Shape: {shape} (Z, Y, X)")
                print(f"  Data type: {dtype}")
                print(f"  File size: {file_size:.2f} MB\n")
        except Exception as e:
            print(f"Error reading {mrc_file}: {e}\n")
        file_count += 1
        if file_count > file_threshold:
            break

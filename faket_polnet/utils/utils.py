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
    THIS FUNCTION IS DEPRECATED. Instead use `collect_results_to_train_dir`. 

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

def collect_results_to_train_dir(source_dir, target_dir_faket, copy_flag=False,
                                 czii_dir_structure=False, collect_faket=True):
    """
    Updated version of `transform_directory_structure`.
    Collect reconstructed tomogram files into the target directory.
    If `czii_dir_structure` is True, match the directory structure to CZII challenge data;
    otherwise create a flat structure in the target directory.

    Parameters:
        source_dir (str): Path to the source directory containing reconstructed tomograms.
        target_dir_faket (str): Path to the target directory for collected tomograms.
        copy_flag (bool): If True, copy data to target directory; if False, move the data.
        czii_dir_structure (bool): If True, match directory structure to CZII challenge data.
        collect_faket (bool): If True, collect only `_faket.mrc` files (default).
            If False, collect all `.mrc` files that are not `_faket.mrc` (basic tomograms).
    """
    os.makedirs(target_dir_faket, exist_ok=True)

    folders = sorted(os.listdir(source_dir),
                     key=lambda x: (int(x.split('_')[1]), int(x.split('_')[2])))

    for folder in folders:
        folder_path = os.path.join(source_dir, folder)
        if not os.path.isdir(folder_path):
            continue

        for file in os.listdir(folder_path):
            if collect_faket:
                if not file.endswith("_faket.mrc"):
                    continue
            else:
                if not file.endswith(".mrc") or file.endswith("_faket.mrc"):
                    continue
            source_file_path = os.path.join(folder_path, file)

            # Use the original folder name (e.g., tomogram_1_5) if `czii_dir_structure` is True
            target_file_path = (
                os.path.join(target_dir_faket, folder, file) if czii_dir_structure
                else os.path.join(target_dir_faket, file)
            )
            os.makedirs(os.path.dirname(target_file_path), exist_ok=True)

            if copy_flag:
                shutil.copy(source_file_path, target_file_path)
            else:
                label = "faket" if collect_faket else "basic"
                shutil.move(source_file_path, target_file_path)
                print(f"Moved ({label}): {source_file_path} → {target_file_path}")
        

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

def copy_style_micrographs(source_dir, destination_dir, copy_flag=False):
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

def check_mrc_files(directory, file_threshold=5):
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

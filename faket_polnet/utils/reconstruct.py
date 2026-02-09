import pandas as pd
import json
import numpy as np
import os
from tqdm import tqdm
import sys
import numpy as np
from .import lio
from .import tem
import shutil
import random
import time



def project_content_micrographs(out_base_dir, simulation_dirs, tilt_range=(-60, 60, 3), detector_snr=None, simulation_index=None,
                                 micrograph_threshold=100, reconstruct_3d=False, add_misalignment = False, simulation_threshold=1, 
                                 ax="Y", cluster_run = False):
    """
    Project micrographs from 3D densities and save all TEM-related files in the output directory.
    # TODO pipeline.py is currently setup to processs one simulation directory at a time

    Parameters:
        out_base_dir (str): Directory to save the generated micrographs and TEM files.
        simulation_dirs (list): List of simulation directories containing 3D densities.
        tilt_range (tuple): Range of tilt angles (start, stop, step).
        detector_snr (list or float, optional): Signal-to-noise ratio for adding noise to micrographs.
        micrograph_threshold (int): Maximum number of micrographs to process.
        reconstruct_3d (bool): Whether to perform 3D reconstruction.
    """
    MALIGN_MN = 1
    MALIGN_MX = 1.5
    MALIGN_SG = 0.2
    
    if simulation_index is None:
        simulation_index = 0
    micrograph_index = 0
    snr_list = []

    for sim_dir in sorted(simulation_dirs):
        if not os.path.exists(sim_dir):
            print(f"Simulation directory {sim_dir} does not exist. Skipping.")
            continue

        if not os.listdir(sim_dir):
            print(f"Simulation directory {sim_dir} is empty. Skipping.")
            continue
        
        print(f"Processing simulation directory: {sim_dir}")
        tom_dir = os.path.join(sim_dir, "tomos")
        if not os.path.exists(tom_dir):
            raise FileNotFoundError("Tomogram directory not found.")

        tomogram_files = [f for f in os.listdir(tom_dir) if f.startswith("tomo_den_") and f.endswith(".mrc")]
        n_tomos = len(tomogram_files)
        print(f"Found {n_tomos} tomograms to process in {tom_dir}.")

        for tomod_id in range(n_tomos):
            print("PROJECTING MICROGRAPHS FOR TOMOGRAM NUMBER:", tomod_id)
            hold_time = time.time()

            tomo_den_out = os.path.join(tom_dir, f"tomo_den_{tomod_id}.mrc")
            if not os.path.exists(tomo_den_out):
                raise FileNotFoundError(f"3D density file {tomo_den_out} is missing.")

            # Create a unique TEM directory for this tomogram
            tem_output_dir = os.path.join(out_base_dir, f"TEM/tomogram_{simulation_index}_{tomod_id}")
            os.makedirs(tem_output_dir, exist_ok=True)

            # Create output directory for micrographs
            tomo_output_dir = os.path.join(out_base_dir, f"Micrographs/tomogram_{simulation_index}_{tomod_id}")
            os.makedirs(tomo_output_dir, exist_ok=True)

            # Create a new TEM object for this tomogram
            temic = tem.TEM(tem_output_dir)
            vol = lio.load_mrc(tomo_den_out)
            print(f"Loaded tomogram from {tomo_den_out} with shape {vol.shape}")
            if vol is None:
                raise ValueError(f"Failed to load 3D density file {tomo_den_out}.")
            if np.count_nonzero(vol) == 0:
                print(f"3D density file {tomo_den_out} is empty. Skipping.")
                continue
            # Load and print shape of saved micrograph
            if cluster_run:
                temic.gen_tilt_series_imod_0(vol, np.arange(*tilt_range), ax=ax)
            else:
                temic.gen_tilt_series_imod(vol, np.arange(*tilt_range), ax=ax)

            clean_mics_path = os.path.join(tomo_output_dir, f"tomo_mics_clean_{tomod_id}.mrc")
            shutil.copyfile(temic._TEM__micgraphs_file, clean_mics_path)
            print(f"Saved clean noiseless projections to {clean_mics_path}")
            
            if add_misalignment:
                temic.add_mics_misalignment(MALIGN_MN, MALIGN_MX, MALIGN_SG)

            if detector_snr:
                snr = round(random.uniform(detector_snr[0], detector_snr[1]), 2) if isinstance(detector_snr, list) else detector_snr
                temic.add_detector_noise(snr)
                snr_list.append(snr)
                print(f"Added detector noise with SNR: {snr}")
            else:
                print("No detector noise added.")

            temic.invert_mics_den()
            out_mics = os.path.join(tomo_output_dir, f"tomo_mics_{tomod_id}_{snr}.mrc")
            shutil.copyfile(os.path.join(tem_output_dir, "out_micrographs.mrc"), out_mics)
            saved_micrograph = lio.load_mrc(out_mics)
            if saved_micrograph is not None:
                print(f"Saved micrograph shape: {saved_micrograph.shape}")
            else:
                print(f"Failed to load saved micrograph from {out_mics}")
            if reconstruct_3d:
                temic.recon3D_imod()
                temic.set_header(data="rec3d", p_size=(10, 10, 10), origin=(0, 0, 0))
            print(f"Micrographs for tomogram {tomod_id} projected in {time.time() - hold_time:.2f} seconds.")
            micrograph_index += 1
            print(f"Micrograph {micrograph_index} saved to {out_mics}.")
            if micrograph_index >= micrograph_threshold:
                print(f"Micrograph index {micrograph_index} exceeds threshold. Stopping further processing.")
                break
        simulation_index += 1
        print(f"Simulation {simulation_index} processed.")
        if simulation_index >= simulation_threshold:
            print("Simulation threshold had been surpassed Terminating...")
            break

    print("Successfully projected micrographs for all simulations.")
    return snr_list

def reconstruct_micrographs_only_recon3D(TEM_paths,faket_paths, out_base_dir,snr_list, custom_mic=False,
                                         micrograph_threshold=100, cluster_run=False):
    os.makedirs(out_base_dir, exist_ok=True)
    
    micrograph_index = 0

    for TEM_path,snr,faket_path in zip(TEM_paths,snr_list,faket_paths):
        if not os.path.exists(TEM_path):
            print(f"Micrograph file {TEM_path} does not exist. Skipping.")
            continue

        print(f"Processing micrograph: {TEM_path}")
        hold_time = time.time()

        tomogram_id = TEM_path.split("/")[-1]

        tomo_output_dir = os.path.join(out_base_dir, f"{tomogram_id}")
        os.makedirs(tomo_output_dir, exist_ok=True)

        temic = tem.TEM(TEM_path)
        temic.set_header(data="mics", p_size=(10, 10, 10))
        if custom_mic:
            custom_mic_file = faket_path
            if not os.path.exists(custom_mic_file):
                raise FileNotFoundError(f"Micrographs file {custom_mic_file} does not exist.")
            temic._TEM__micgraphs_file = custom_mic_file
        

        print("Micrographs file:", temic._TEM__micgraphs_file)
        print("Tilt file:", temic._TEM__tangs_file)
        print("Output file:", temic._TEM__rec3d_file)

        try:
            if cluster_run:
                temic.recon3D_imod_0()
            else:
                temic.recon3D_imod()
        except Exception as e:
            print(f"Error during reconstruction: {e}")
            continue
        if custom_mic:
            out_tomo_rec = os.path.join(tomo_output_dir, f"{tomogram_id}_faket.mrc")
        else:
            out_tomo_rec = os.path.join(tomo_output_dir, f"{tomogram_id}_{snr}.mrc")
        if not os.path.exists(temic._TEM__rec3d_file):
            print(f"Reconstruction failed. Output file {temic._TEM__rec3d_file} not found.")
            continue
 
        shutil.move(temic._TEM__rec3d_file, out_tomo_rec)
        print(f"Reconstructed tomogram saved to {out_tomo_rec}")
        print(f"Micrograph {micrograph_index} processed in {time.time() - hold_time:.2f} seconds.")
        micrograph_index += 1
        if micrograph_index >= micrograph_threshold:
            # If the simulation index exceeds the threshold, stop further processing
            # and break out of the loop
            print(f"Micrograph index {micrograph_index} exceeds threshold. Stopping further processing.")
            break
    print("Reconstruction completed for all micrographs.")

def project_style_micrographs(style_tomo_dir, out_base_dir, tilt_range=(-60, 60, 3), ax = "Y", 
                              cluster_run = False, invert_density=False,projection_threshold = 1):
    """
    Project style micrographs from reconstructed tomograms.

    Parameters:
        style_tomo_dir (str): Directory containing reconstructed tomograms (.mrc files).
        out_base_dir (str): Output base directory for the style micrographs.
        tilt_range (tuple): Range of tilt angles (start, stop, step).
    """
    os.makedirs(out_base_dir, exist_ok=True)
    style_tomo_files = [f for f in os.listdir(style_tomo_dir) if f.endswith('.mrc')]
    projection_count = 0
    for i, filename in enumerate(sorted(style_tomo_files)):
        tomo_path = os.path.join(style_tomo_dir, filename)
        tomo_id = os.path.splitext(filename)[0]
        print(f"Processing style tomogram: {tomo_path}")

        # Load the reconstructed volume
        vol = lio.load_mrc(tomo_path)
        if vol is None or np.count_nonzero(vol) == 0:
            print(f"Skipping {filename}, empty or unreadable.")
            continue

        # Create output folder for projections
        tomo_output_dir = os.path.join(out_base_dir, f"StyleMicrographs_{tomo_id}")
        os.makedirs(tomo_output_dir, exist_ok=True)

        # Use TEM object to simulate tilt series (no noise, no misalignment)
        temic = tem.TEM(tomo_output_dir)
        if cluster_run:
            temic.gen_tilt_series_imod_0(vol, np.arange(*tilt_range), ax=ax)
        else:
            temic.gen_tilt_series_imod(vol, np.arange(*tilt_range), ax=ax)
        if invert_density:
            temic.invert_mics_den()
        # Save the style micrographs (no noise)
        style_mics_out = os.path.join(tomo_output_dir, f"{tomo_id}_style_mics.mrc")
        shutil.copyfile(temic._TEM__micgraphs_file, style_mics_out)
        print(f"Saved style projections to {style_mics_out}")
        projection_count += 1
        if projection_count > projection_threshold:
            print("Projection threshold has been reached ")
            break

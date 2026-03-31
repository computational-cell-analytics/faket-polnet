import pandas as pd
import json
import numpy as np
import os
from tqdm import tqdm
import sys
import os
from pathlib import Path
import numpy as np
import shutil
import random
import time


def get_absolute_paths(parent_dir):
    """
    Get absolute paths of all directories inside a given directory.
    
    Parameters:
        parent_dir (str): Path to the parent directory.
    
    Returns:
        list: A list of absolute paths of subdirectories.
    """
    return [os.path.abspath(os.path.join(parent_dir, d)) for d in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, d))]

def quaternion_to_matrix(q1, q2, q3, q4):
    """Convert quaternion to a 4x4 transformation matrix."""
    # Normalize quaternion
    norm = np.sqrt(q1**2 + q2**2 + q3**2 + q4**2)
    q1, q2, q3, q4 = q1 / norm, q2 / norm, q3 / norm, q4 / norm

    # Create rotation matrix
    rotation_matrix = np.array([
        [1 - 2*(q2**2 + q3**2), 2*(q1*q2 - q3*q4), 2*(q1*q3 + q2*q4), 0],
        [2*(q1*q2 + q3*q4), 1 - 2*(q1**2 + q3**2), 2*(q2*q3 - q1*q4), 0],
        [2*(q1*q3 - q2*q4), 2*(q2*q3 + q1*q4), 1 - 2*(q1**2 + q2**2), 0],
        [0, 0, 0, 1]
    ])
    return rotation_matrix.tolist()

def csv_to_json(csv_file, json_directory, labels_table,mapping=None):
    # Read CSV file with pandas (using tab separator)
    df = pd.read_csv(csv_file, sep='\t')

    # Read labels table to create a mapping from Code to protein names
    labels_df = pd.read_csv(labels_table, sep='\t')
    code_to_protein = dict(zip(labels_df['LABEL'], labels_df['MODEL']))
    get_name = lambda x: x.split("/")[1].split(".")[0]
    code_to_protein = {k: get_name(v) for k, v in code_to_protein.items()}  # Convert keys to strings
    # Filter rows where Type is either 'SAWCL' or 'Mb-SAWLC'
    df_filtered = df[df['Type'].isin(['Helix', 'SAWLC', 'Mb-SAWLC'])]

    # Group by the 'Code' column to create separate JSON files for each protein
    grouped = df_filtered.groupby('Label')

    # Create the JSON directory if it doesn't exist
    os.makedirs(json_directory, exist_ok=True)

    # Iterate over each group (protein code)
    for label, group in grouped:
        # Get the protein name from the mapping
        protein_name = code_to_protein.get(label, str(label))  # Use code as fallback if not found
        print(f"Processing protein: {protein_name}")
        if mapping is not None:
            protein_name = mapping.get(protein_name, protein_name)  # Use mapping if provided
            #print(f"Mapped protein name: {protein_name}")
        # Check if the protein name is a valid string
        # Check if the protein name is valid
        if not isinstance(protein_name, str) or not protein_name:
            print(f"Invalid protein name for label {label}. Skipping...")
            continue
        # Initialize JSON structure for this protein
        json_data = {
            "pickable_object_name": protein_name,  # Use the protein name
            "user_id": "curation",
            "session_id": "0",
            "run_name": "TS_5_4",
            "voxel_spacing": None,
            "unit": "angstrom",
            "points": [],
            "trust_orientation": True
        }

        # Iterate over rows in the group
        for _, row in group.iterrows():
            # Extract relevant fields
            x = float(row['X'])
            y = float(row['Y'])
            z = float(row['Z'])
            instance_id = int(row['Polymer'])
            q1 = float(row['Q1'])
            q2 = float(row['Q2'])
            q3 = float(row['Q3'])
            q4 = float(row['Q4'])

            # Convert quaternion to transformation matrix
            transformation = quaternion_to_matrix(q1, q2, q3, q4)

            # Add point to JSON
            json_data['points'].append({
                "location": {"x": x, "y": y, "z": z},
                "transformation_": transformation,
                "instance_id": instance_id
            })

        # Define the output JSON file path
        json_file = os.path.join(json_directory, f"{protein_name}.json")

        # Write JSON to file
        with open(json_file, mode='w') as file:
            json.dump(json_data, file, indent=4)

        #print(f"Created JSON file for protein '{protein_name}' at {json_file}")

def get_tomos_motif_list_paths(master_dir):
    """
    Get the absolute paths of tomo_motif_list_*.csv files from the simulation directory.
ther
    Parameters:
        master_dir (str): Path to the simulation directory containing `motif_lists` directory.

    Returns:
        list: Sorted list of absolute paths to tomo_motif_list_*.csv files.
    """
    csv_dir = Path(master_dir) / "motif_lists"
    if not csv_dir.exists():
        raise FileNotFoundError(f"`motif_lists` directory not found in {master_dir}.")

    return sorted(csv_dir.glob("tomo_motif_list_*.csv"))


def find_labels_table(simulation_dirs, filename="labels_table.csv"):
    """
    Search for the labels_table.csv file in the given simulation directories.

    Parameters:
        simulation_dirs (list): List of simulation directories to search in.
        filename (str): Name of the file to search for (default is 'labels_table.csv').

    Returns:
        str: Path to the labels_table.csv file if found, otherwise raises an error.
    """
    for sim_dir in simulation_dirs:
        print(f"Searching in: {sim_dir}")
        potential_path = os.path.join(sim_dir, filename)
        if os.path.exists(potential_path):
            print(f"Found {filename} at: {potential_path}")
            return potential_path
    raise FileNotFoundError(f"{filename} not found in any of the simulation directories.")


def label_transform(in_csv_list, out_dir, labels_table, simulation_index, mapping_flag=True):
    """
    Convert tomo_motif_list_*.csv files directly to JSON annotation files.

    Parameters:
        in_csv_list (list): List of paths to the input tomo_motif_list_*.csv files.
        out_dir (str): Path to the output directory where JSON files will be saved.
        labels_table (str): Path to the labels table CSV file.
        simulation_index (int): Simulation index used to name output tomogram directories.
        mapping_flag (bool): If True, apply protein name mapping.
    """
    if mapping_flag:
        mapping = {
            "1fa2_10A": "beta-amylase",
            "6drv_10A": "beta-galactosidase",
            "6n4v_10A": "virus-like-particle",
            "6qzp_10A": "ribosome",
            "7n4y_10A": "thyroglobulin",
            "8cpv_10A": "apo-ferritin",
            "8vaf_10A": "albumin"
                }
    else:
        mapping = None

    os.makedirs(out_dir, exist_ok=True)

    for in_csv in in_csv_list:
        tomo_index = os.path.splitext(os.path.basename(in_csv))[0].split('_')[-1]
        json_output_dir = os.path.join(out_dir, f"tomogram_{simulation_index}_{tomo_index}")
        csv_to_json(str(in_csv), json_output_dir, labels_table, mapping=mapping)
        print(f"Label transform complete for tomogram_{simulation_index}_{tomo_index}.")

    print(f"Performed label transform for simulation {simulation_index}.")
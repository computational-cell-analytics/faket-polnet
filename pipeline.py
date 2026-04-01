import os
import json
import subprocess
import numpy as np
import shutil
import ssl
import mrcfile
import configargparse
from pathlib import Path
from faket_polnet.utils.reconstruct import project_content_micrographs, reconstruct_micrographs_only_recon3D, project_style_micrographs
from faket_polnet.utils.utils import collect_results_to_train_dir, copy_style_micrographs
from faket_polnet.utils.label_transform import label_transform, find_labels_table, get_tomos_motif_list_paths

from faket_polnet.utils.faket_wrapper import style_transfer_wrapper

ssl._create_default_https_context = ssl._create_unverified_context

def parse_args():
    parser = configargparse.ArgParser(
        config_file_parser_class=configargparse.TomlConfigParser(["tool.faket"])
    )
    # Input configuration file (TOML)
    parser.add_argument("--config", type=str, default=None, is_config_file_arg=True,
                        help="Path to TOML configuration file.")
    
    # Required argument
    parser.add_argument('--base_dir', type=str, required=True,
                        help='Base directory containing simulation and style directories')
    
    # Index parameters
    parser.add_argument('--micrograph_index', type=int, default=0,
                        help='Micrograph index. Handles intermediate directories, should match `train_dir_index`.')
    parser.add_argument('--style_index', type=int, default=0,
                        help='Style index. Corresponds to the style tomograms.')
    parser.add_argument('--simulation_index', type=int, default=0,
                        help='Simulation index. Corresponds to the PolNet simulation.')
    parser.add_argument('--train_dir_index', type=int, default=0,
                        help='Train directory index')
    parser.add_argument('--czii_dir_structure', action='store_true', default=False, 
                        help='Match training directory structure to CZII challenge format.')
    
    # Tilt range parameters
    parser.add_argument('--tilt_start', type=int, default=-60, help='Tilt series start angle')
    parser.add_argument('--tilt_stop', type=int, default=60, help='Tilt series stop angle')
    parser.add_argument('--tilt_step', type=int, default=3, help='Tilt series step size')
    
    # Other parameters
    parser.add_argument('--detector_snr', type=float, nargs=2, default=[0.15, 0.20], help='Detector SNR range')
    parser.add_argument('--denoised', action='store_true', help='Use denoised style micrographs')
    parser.add_argument('--random_faket', action='store_true', default=True, help='Use random faket style transfer')
    
    # Faket parameters
    parser.add_argument('--faket_gpu', type=int, default=0, help='GPU device ID for faket')
    parser.add_argument('--faket_iterations', type=int, default=500, help='Number of iterations for faket style transfer')
    parser.add_argument('--faket_step_size', type=float, default=0.02, help='Step size for faket')
    parser.add_argument('--faket_min_scale', type=int, default=128, help='Minimum scale for faket')
    parser.add_argument('--faket_end_scale', type=int, default=None, help='End scale for faket')
    
    return parser.parse_args()

def validate_directories(base_dir, simulation_index, style_index):
    """Validate that required directories exist"""
    simulation_dir = base_dir / f"simulation_dir_{simulation_index}"
    style_tomo_dir = base_dir / f"style_tomograms_{style_index}"
    style_dir = base_dir / f"style_micrographs_{style_index}"
    if not simulation_dir.exists():
        raise ValueError(
            f"ERROR Simulation directory not found: {simulation_dir}"
            )
    
    # Check if either style_tomo_dir or style_dir exists
    style_tomo_exists = style_tomo_dir.exists()
    style_dir_exists = style_dir.exists()
    
    if not style_tomo_exists and not style_dir_exists:
        raise ValueError(
            f"ERROR: Neither style tomogram directory {style_tomo_dir.relative_to(base_dir)} nor style directory {style_dir.relative_to(base_dir)} found in {base_dir}. At least one must exist."
            )
    print(f"Parent directory: {base_dir}")
    print(f"Found simulation directory: {simulation_dir}")
    if style_tomo_exists:
        print(f"Found style tomogram directory: {style_tomo_dir}")
    if style_dir_exists:
        print(f"Found style directory: {style_dir}")
    
    return simulation_dir, style_tomo_dir, style_dir, style_tomo_exists, style_dir_exists

def main():
    args = parse_args()

    print(f"\n====== Simulation {args.simulation_index} ======\n")

    # Set base directory from arguments
    base_dir = Path(args.base_dir)
    
    # Validate that required directories exist
    simulation_dir, style_tomo_dir, style_dir, style_tomo_exists, style_dir_exists = validate_directories(base_dir, args.simulation_index, args.style_index)
    
    # Set parameters from arguments
    micrograph_index = args.micrograph_index
    style_index = args.style_index
    simulation_index = args.simulation_index
    train_dir_index = args.train_dir_index
    czii_dir_structure = args.czii_dir_structure
    tilt_range = (args.tilt_start, args.tilt_stop + 1, args.tilt_step)
    detector_snr = args.detector_snr
    denoised = args.denoised
    random_faket = args.random_faket

    # Faket parameters
    faket_gpu = args.faket_gpu
    faket_iterations = args.faket_iterations
    faket_step_size = args.faket_step_size
    faket_min_scale = args.faket_min_scale
    faket_end_scale = args.faket_end_scale
    
    # Calculate seq_end
    seq_end = len(np.arange(*tilt_range))
    print(f"tilt_range: {tilt_range}")
    print(f"seq_end: {seq_end}")

    # Simulation parameters
    simulation_base_dir = base_dir / f"simulation_dir_{simulation_index}"
    simulation_dirs = [simulation_base_dir]
    labels_table = find_labels_table(simulation_dirs)

    in_csv_list = get_tomos_motif_list_paths(simulation_base_dir)
    out_dir = base_dir / f"train_dir_{train_dir_index}/overlay"

    # Auto-set faket_end_scale from simulation tomogram dims before style projection
    if faket_end_scale is None:
        tomo_den = simulation_dir / "tomos" / "tomo_den_0.mrc"
        if tomo_den.exists():
            with mrcfile.open(str(tomo_den), permissive=True) as mrc:
                Z, H, W = mrc.data.shape
                faket_end_scale = int(max(H, W))
                print(f"Automatically set faket_end_scale to {faket_end_scale}.")

    # Handle style directory logic
    print("\n=== Style Micrograph Projection ===\n")
    micrographs_base_dir = base_dir / f"micrograph_dir_{micrograph_index}"
    style_mics_out_dir = micrographs_base_dir / f"style_micrographs_{style_index}"

    if style_dir_exists:
        print("Style directory already exists, skipping projection and copy.")
    elif style_tomo_exists:
        # Only run projection if style_tomo_dir exists and style_dir doesn't exist
        print("Style tomogram directory found but style directory doesn't exist. Running projection...")
        style_mics_out_dir.mkdir(parents=True, exist_ok=True)
        project_style_micrographs(style_tomo_dir, style_mics_out_dir, tilt_range=tilt_range, ax="Y", cluster_run=False, projection_threshold=100, target_size=faket_end_scale)
        copy_style_micrographs(style_mics_out_dir, style_dir, copy_flag=False)
        print(f"Style projection completed and copied to: {style_dir}")
    else:
        # This should not happen due to validation, but just in case
        raise ValueError("No style directory available")

    # Label transformation
    if not out_dir.exists():
        if czii_dir_structure:
            label_transform(in_csv_list, out_dir, labels_table, simulation_index, mapping_flag=True)
    
    # Project content micrographs
    print("\n=== Simulation Micrograph Projection ===\n")
    content_mics_out_dir = micrographs_base_dir / f"content_micrographs_{simulation_index}"
    
    if not content_mics_out_dir.exists():
        project_content_micrographs(
            content_mics_out_dir, simulation_dir, tilt_range, detector_snr, simulation_index,
            reconstruct_3d=False, add_misalignment=True
        )

    print("\n=== Style Transfer ===\n")    
    STYLE_DIR = style_dir  # Use the validated style directory

    # Define directories
    CLEAN_DIR = content_mics_out_dir / "Micrographs"
    OUTPUT_DIR = micrographs_base_dir / f"faket_micrographs_{simulation_index}"


    # Create OUTPUT_DIR if it doesn't exist
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Find all clean tomograms
    clean_tomograms = subprocess.run(f"find {CLEAN_DIR} -name 'tomo_mics_clean_*.mrc'", 
                                    shell=True, capture_output=True, text=True).stdout.splitlines()
    
    if not clean_tomograms:
        print(f"No clean tomograms found in {CLEAN_DIR}")
        return
    
    # build dict to store selected style and snr
    json_dict = {}
    for CLEAN_TOMOGRAM in clean_tomograms:
        TOMOGRAM_DIR = CLEAN_TOMOGRAM.split('/')[-2]
        CLEAN_ID = CLEAN_TOMOGRAM.split('_')[-1].replace('.mrc', '')
        print(f"Clean ID: {CLEAN_ID}")
        print(f"Looking for: tomo_mics_{CLEAN_ID}_*.mrc")
        NOISY_TOMOGRAM_CMD = f"find {CLEAN_DIR}/{TOMOGRAM_DIR} -name 'tomo_mics_{CLEAN_ID}_*.mrc' | head -n 1"
        NOISY_TOMOGRAM = subprocess.run(NOISY_TOMOGRAM_CMD, shell=True, capture_output=True, text=True).stdout.strip()

        if not NOISY_TOMOGRAM:
            print(f"Noisy tomogram not found for {TOMOGRAM_DIR}. Skipping...")
            continue
        
        if random_faket:
            STYLE_TOMOGRAM_CMD = f"find {STYLE_DIR} -name '*.mrc' | shuf -n 1"
            STYLE_TOMOGRAM = subprocess.run(STYLE_TOMOGRAM_CMD, shell=True, capture_output=True, text=True).stdout.strip()
        else:
            # If not random, you might want to implement specific style selection logic here
            STYLE_TOMOGRAM_CMD = f"find {STYLE_DIR} -name '*.mrc' | head -n 1"
            STYLE_TOMOGRAM = subprocess.run(STYLE_TOMOGRAM_CMD, shell=True, capture_output=True, text=True).stdout.strip()

        json_dict[TOMOGRAM_DIR] = {
            "snr": float(Path(NOISY_TOMOGRAM).stem.rsplit('_', 1)[-1]),
            "style": Path(STYLE_TOMOGRAM).name.replace('_style_mics.mrc', '')
        }

        OUTPUT_TOMOGRAM = f"{OUTPUT_DIR}/tomo_style_transfer_{TOMOGRAM_DIR}.mrc"
        print(f"Output Tomogram: {OUTPUT_TOMOGRAM}")

        if os.path.exists(OUTPUT_TOMOGRAM):
            print(f"Output tomogram already exists, skipping: {OUTPUT_TOMOGRAM}")
            continue
            
        print(f"Processing:"
              f"\n  Clean: {CLEAN_TOMOGRAM}"
              f"\n  Noisy: {NOISY_TOMOGRAM}"
              f"\n  Style: {STYLE_TOMOGRAM}")
        # build extra_args dict
        extra_args = {
            "init": NOISY_TOMOGRAM,
            "seq_start": 0,
            "seq_end": seq_end,
            "content-weight": 0.015,
            "tv-weight": 2,
            "initial-iterations": 1000, 
            "step-size": 0.02, 
            "avg-decay": 0.99,
            "style-scale-fac": 1.0,
            "pooling": "max",
            "content_layers": 8,
            "content_layers_weights": 100,
            "model_weights": "pretrained",
            }

        style_transfer_wrapper(
            content_path=CLEAN_TOMOGRAM,
            style_paths=[STYLE_TOMOGRAM],
            output_path=OUTPUT_TOMOGRAM,
            devices=[f"cuda:{faket_gpu}"],
            iterations=faket_iterations,
            save_every=100,
            min_scale=faket_min_scale,
            end_scale=faket_end_scale,
            random_seed=0,
            style_weights=[1.0],
            extra_args=extra_args,
        )

    print("Style transfer completed for one index!")

    base_dir_Micrographs = content_mics_out_dir / "Micrographs"
    base_dir_TEM = content_mics_out_dir / "TEM"
    base_dir_faket = micrographs_base_dir / f"faket_micrographs_{simulation_index}"

    # save styles and snr for documentation
    snr_list_dir = base_dir / f"train_dir_{train_dir_index}/snr_list_dir"
    snr_list_dir.mkdir(parents=True, exist_ok=True)

    with (snr_list_dir / f"snr_list_{simulation_index}.json").open("w") as file:
        json.dump(json_dict, file, indent=4)

    # Check if directories exist before processing
    if not base_dir_TEM.exists() or not base_dir_Micrographs.exists():
        print(f"Required directories not found: {base_dir_TEM} or {base_dir_Micrographs}")
        return
    
    tomograms_sorted = sorted(
        base_dir_TEM.iterdir(), 
        key=lambda x: (int(x.name.split('_')[1]), int(x.name.split('_')[2]))
        )
    TEM_paths = [str(p) for p in tomograms_sorted]


    print(f"TEM Paths: {[TEM_paths]}")

    faket_paths = list(base_dir_faket.glob("*.mrc"))

    print("\n=== Tomogram Reconstruction ===\n")
    if faket_paths:
        faket_paths = sorted(faket_paths, key=lambda x: (int(x.name.split('_')[4]), int(x.name.split('_')[5].split('.')[0])))
        faket_paths = [str(p) for p in faket_paths]

        source_dir = f"{base_dir}/reconstructed_tomograms_{train_dir_index}"
        target_dir_faket = f"{base_dir}/train_dir_{train_dir_index}/faket_tomograms"
        
        for TEM_path, faket_path in zip(TEM_paths, faket_paths):
            reconstruct_micrographs_only_recon3D(TEM_path, faket_path, source_dir, custom_mic=True)

        print("\n=== Cleanup ===\n")

        if czii_dir_structure:
            collect_results_to_train_dir(source_dir, target_dir_faket, czii_dir_structure = True)
        else:
            collect_results_to_train_dir(source_dir, target_dir_faket)

        dirs_to_remove = [source_dir, micrographs_base_dir]

        for dir in dirs_to_remove:
            try:
                shutil.rmtree(dir)
                print(f"Successfully deleted the directory: {dir}\n")
            except Exception as e:
                print(f"Error deleting directory {dir}: {e}\n")

if __name__ == "__main__":
    main()
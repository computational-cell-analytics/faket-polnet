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
from faket_polnet.utils.utils import get_absolute_paths, collect_results_to_train_dir, copy_style_micrographs
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
    parser.add_argument('--stage', type=int, required=True, choices=[1, 2, 3],
                        help='Pipeline stage to run: 1. setup, 2. style transfer, 3. cleanup.')
    
    # Index parameters
    parser.add_argument('--micrograph_index', type=int, default=0, 
                        help='Micrograph index. Handles intermediate directories,'
                        'should match `train_dir_index`.')
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

def run_setup(args, base_dir, style_dir, style_tomo_dir, style_tomo_exists, style_dir_exists):
    """
    Stage 1: Setup (shared)
    - Project style micrographs
    - Label transform (required for CZII challenge)
    """
    tilt_range = (args.tilt_start, args.tilt_stop + 1, args.tilt_step)
    simulation_base_dir = base_dir / f"simulation_dir_{args.simulation_index}"
    micrographs_base_dir = base_dir / f"micrograph_dir_{args.micrograph_index}"
    train_dir_index = args.train_dir_index
    style_mics_out_dir = micrographs_base_dir / f"style_micrographs_{args.style_index}"

    print("\n=== Style Micrograph Projection ===\n")
    if style_dir_exists:
        print("Style directory already exists, skipping projection and copy.")
    elif style_tomo_exists:
        print("Style tomogram directory found but style directory doesn't exist. Running projection...")
        style_mics_out_dir.mkdir(parents=True, exist_ok=True)
        project_style_micrographs(style_tomo_dir, style_mics_out_dir, tilt_range=tilt_range,
                                   ax="Y", cluster_run=False, projection_threshold=100)
        copy_style_micrographs(style_mics_out_dir, style_dir, copy_flag=False)
        print(f"Style projection completed and copied to: {style_dir}")
    else:
        raise ValueError("No style directory available")

    print("\n=== Label Transformation ===\n")
    out_dir = base_dir / f"train_dir_{train_dir_index}/overlay"
    if not out_dir.exists():
        if args.czii_dir_structure:
            labels_table = find_labels_table([simulation_base_dir])
            in_csv_list = sorted(get_tomos_motif_list_paths(simulation_base_dir))
            csv_dir_list = [Path(d) / "csv" for d in get_absolute_paths(simulation_base_dir)]
            label_transform(in_csv_list, out_dir, csv_dir_list, labels_table,
                            args.simulation_index, mapping_flag=True)
    else:
        print("Overlay directory already exists, skipping label transformation.")

    print("\nSetup complete.")


def run_style_transfer(tomo_index, args, base_dir, style_dir, faket_end_scale):
    """
    Stage 2: Style transfer (per-tomogram)
    - Projects content micrographs for the current tomogram.
    - Applies style transfer.
    - Reconstructs the style-transferred tomogram.
    - Saves per-tomogram JSON.

    Parameters:
        tomo_index (int): Tomogram index, equal to SLURM_ARRAY_TASK_ID.
        args: Parsed arguments from parse_args().
        base_dir (Path): Base directory for the pipeline.
        style_dir (Path): Directory containing style micrographs.
        faket_end_scale (int): End scale for faket style transfer.
    """
    simulation_index = args.simulation_index
    train_dir_index = args.train_dir_index
    micrograph_index = args.micrograph_index
    tilt_range = (args.tilt_start, args.tilt_stop + 1, args.tilt_step)
    seq_end = len(np.arange(*tilt_range))
    simulation_base_dir = base_dir / f"simulation_dir_{simulation_index}"
    micrographs_base_dir = base_dir / f"micrograph_dir_{micrograph_index}"
    content_mics_out_dir = micrographs_base_dir / f"content_micrographs_{simulation_index}"
    CLEAN_DIR = content_mics_out_dir / "Micrographs"
    TOMOGRAM_DIR = f"tomogram_{simulation_index}_{tomo_index}"
    OUTPUT_DIR = micrographs_base_dir / f"faket_micrographs_{simulation_index}"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    snr_list_dir = base_dir / f"train_dir_{train_dir_index}/snr_list_dir"
    
    print(f"\n====== Processing Tomogram {tomo_index} ======\n")

    print("\n=== Content Micrograph Projection ===\n")
    CLEAN_TOMOGRAM = CLEAN_DIR / TOMOGRAM_DIR / f"tomo_mics_clean_{tomo_index}.mrc"
    if not CLEAN_TOMOGRAM.exists():
        snr = project_content_micrographs(
            content_mics_out_dir, simulation_base_dir,
            tilt_range, args.detector_snr, simulation_index,
            micrograph_threshold=100, reconstruct_3d=False, add_misalignment=True,
            tomo_index=tomo_index,
        )
    else:
        print(f"Content micrographs already exist, skipping: {CLEAN_TOMOGRAM}")
        snr = None

    print("\n=== Style Transfer ===\n")

    if not CLEAN_TOMOGRAM.exists():
        print(f"Clean tomogram not found: {CLEAN_TOMOGRAM}. Skipping.")
        return

    NOISY_TOMOGRAM_CMD = (f"find {CLEAN_DIR}/{TOMOGRAM_DIR} "
                          f"-name 'tomo_mics_{tomo_index}_*.mrc' | head -n 1")
    NOISY_TOMOGRAM = subprocess.run(
        NOISY_TOMOGRAM_CMD, shell=True, capture_output=True, text=True
    ).stdout.strip()
    if not NOISY_TOMOGRAM:
        print(f"Noisy tomogram not found for {TOMOGRAM_DIR}. Skipping.")
        return

    if snr is None:
        snr = float(Path(NOISY_TOMOGRAM).stem.rsplit('_', 1)[-1])

    if args.random_faket:
        STYLE_TOMOGRAM_CMD = f"find {style_dir} -name '*.mrc' | shuf -n 1"
    else:
        STYLE_TOMOGRAM_CMD = f"find {style_dir} -name '*.mrc' | head -n 1"
    STYLE_TOMOGRAM = subprocess.run(
        STYLE_TOMOGRAM_CMD, shell=True, capture_output=True, text=True
    ).stdout.strip()

    style_name = Path(STYLE_TOMOGRAM).name.replace('_style_mics.mrc', '')

    OUTPUT_TOMOGRAM = str(OUTPUT_DIR / f"tomo_style_transfer_{TOMOGRAM_DIR}.mrc")
    print(f"Output Tomogram: {OUTPUT_TOMOGRAM}")

    if os.path.exists(OUTPUT_TOMOGRAM):
        print(f"Output tomogram already exists, skipping: {OUTPUT_TOMOGRAM}")
    else:
        print(f"Processing:"
              f"\n  Clean: {CLEAN_TOMOGRAM}"
              f"\n  Noisy: {NOISY_TOMOGRAM}"
              f"\n  Style: {STYLE_TOMOGRAM}")
        extra_args = {
            "init": NOISY_TOMOGRAM,
            "seq_start": 0,
            "seq_end": seq_end,
            "content-weight": 0.015,
            "tv-weight": 2,
            "initial-iterations": 1000,
            "step-size": args.faket_step_size,
            "avg-decay": 0.99,
            "style-scale-fac": 1.0,
            "pooling": "max",
            "content_layers": 8,
            "content_layers_weights": 100,
            "model_weights": "pretrained",
        }
        style_transfer_wrapper(
            content_path=str(CLEAN_TOMOGRAM),
            style_paths=[STYLE_TOMOGRAM],
            output_path=OUTPUT_TOMOGRAM,
            devices=[f"cuda:{args.faket_gpu}"],
            iterations=args.faket_iterations,
            save_every=100,
            min_scale=args.faket_min_scale,
            end_scale=faket_end_scale,
            random_seed=0,
            style_weights=[1.0],
            extra_args=extra_args,
        )

    json_dict = {TOMOGRAM_DIR: {"snr": snr, "style": style_name}}

    snr_list_dir.mkdir(parents=True, exist_ok=True)
    with (snr_list_dir / f"snr_list_{simulation_index}_{tomo_index}.json").open("w") as f:
        json.dump(json_dict, f, indent=4)

    print("\n=== Tomogram Reconstruction ===\n")
    TEM_path = str(content_mics_out_dir / "TEM" / TOMOGRAM_DIR)
    source_dir = base_dir / f"reconstructed_tomograms_{train_dir_index}"
    source_dir.mkdir(parents=True, exist_ok=True)
    reconstruct_micrographs_only_recon3D(
        TEM_path, OUTPUT_TOMOGRAM, str(source_dir), snr,
        custom_mic=True,
    )

    print(f"Style transfer and reconstruction complete for Tomogram {tomo_index}.")


def run_cleanup(args, base_dir):
    """
    Stage 3: Cleanup (shared)
    - Collects per-tomogram metadata into one JSON file
    - Collects reconstructed tomograms to directory for training
    - Removes intermediate directories
    """
    simulation_index = args.simulation_index
    train_dir_index = args.train_dir_index
    micrograph_index = args.micrograph_index
    style_index = args.style_index 

    snr_list_dir = base_dir / f"train_dir_{train_dir_index}/snr_list_dir"

    print("\n=== Collecting per-tomogram metadata to JSON ===\n")
    json_files = sorted(snr_list_dir.glob(f"snr_list_{simulation_index}_*.json"))
    if not json_files:
        print(f"No JSON files found in {snr_list_dir}")
    else:
        merged = {}
        for f in json_files:
            with f.open() as fh:
                merged.update(json.load(fh))

        out_path = snr_list_dir / f"snr_list_{simulation_index}.json"
        with out_path.open("w") as fh:
            json.dump(merged, fh, indent=4)
        print(f"Merged {len(json_files)} files into {out_path}")

    print("\n=== Collecting reconstructed tomograms ===\n")
    source_dir = str(base_dir / f"reconstructed_tomograms_{train_dir_index}")
    target_dir_faket = str(base_dir / f"train_dir_{train_dir_index}/faket_tomograms")
    collect_results_to_train_dir(source_dir, target_dir_faket,
                                 czii_dir_structure=args.czii_dir_structure)

    print("\n=== Removing intermediate directories ===\n")
    dirs_to_remove = [
        base_dir / f"micrograph_dir_{micrograph_index}",
        base_dir / f"style_micrographs_{style_index}",
        base_dir / f"reconstructed_tomograms_{train_dir_index}",
    ]
    for dir in dirs_to_remove:
        try:
            shutil.rmtree(dir)
            print(f"Removed: {dir}")
        except FileNotFoundError:
            print(f"Not found, skipping: {dir}")
        except Exception as e:
            print(f"Error removing {dir}: {e}")


def main():
    args = parse_args()

    print(f"\n====== Simulation {args.simulation_index} | Stage {args.stage} ======\n")

    base_dir = Path(args.base_dir)

    simulation_dir, style_tomo_dir, style_dir, style_tomo_exists, style_dir_exists = validate_directories(
        base_dir, args.simulation_index, args.style_index
    )

    if args.stage == 1:
        run_setup(args, base_dir, style_dir, style_tomo_dir, style_tomo_exists, style_dir_exists)

    elif args.stage == 2:
        tomo_index = int(os.environ.get("SLURM_ARRAY_TASK_ID"))
        faket_end_scale = args.faket_end_scale

        if faket_end_scale is None:
            tomo_den = simulation_dir / "tomos" / f"tomo_den_{tomo_index}.mrc"
            with mrcfile.open(str(tomo_den), permissive=True) as mrc:
                _, H, W = mrc.data.shape
                faket_end_scale = int(max(H, W))
                print(f"Automatically set faket_end_scale to {faket_end_scale} to match spatial dimensions of input {tomo_den}.")

        run_style_transfer(tomo_index, args, base_dir, style_dir, faket_end_scale)

    elif args.stage == 3:
        run_cleanup(args, base_dir)


if __name__ == "__main__":
    main()
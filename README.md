# Style Transfer Pipeline for Synthetic Cryo-ET Data

A comprehensive pipeline for applying neural style transfer to synthetic cryo-electron tomography (cryo-ET) data using [faket](https://github.com/paloha/faket.git), with support for micrograph projection, reconstruction, and training data preparation. The motivation for this package is to combine the [faket-polnet](https://github.com/ybo-source/fakET_polnet) pipeline by ybo-source and FakET into a single, easy-to-install package, simplifying setup and usage on both local and HPC systems.

## Overview

This pipeline processes cryo-ET simulation data and applies neural style transfer to generate augmented training datasets. It handles:

- Style micrograph projection from tomograms
- Clean and noisy micrograph generation
- Neural style transfer using faket
- Tomogram reconstruction from style-transferred micrographs
- Training data organization for downstream tasks

## Requirements
- [IMOD](https://bio3d.colorado.edu/imod/doc/guide.html) must be installed on the system since the pipeline calls to some of its standalone commands.

### Optional Dependencies
- CUDA-enabled GPU (recommended for faster style transfer)
- MPI (for parallel processing)

## Installation
This package was built ontop of [faket](https://github.com/paloha/faket.git). All Python dependencies can be installed using the `environment-gpu.yaml`.

```bash
# clone this repository
git clone https://github.com/stmartineau99/faket-polnet.git
cd faket-polnet 

conda create -n faket-polnet -f environment-gpu.yaml --channel-priority flexible

# activate environment and install faket-polnet
conda activate faket-polnet 
pip install .

# or install in development mode:
pip install -e .
```
## Directory Structure

Before running the pipeline, set up your directory structure as follows:

```
base_directory/
├── simulation_dir_<simulation_index>/     # Simulation data (Required, from polnet)
├── style_tomograms_<style_index>/         # Style tomograms for projection (Required)
├── style_micrographs_<style_index>/       # Projected style micrographs (auto-created)
├── micrograph_dir_<simulation_index>/     # Output directories (auto-created)
├── train_dir_<train_dir_index>/           # Training data (auto-created)
```

## Usage

### Basic Usage

**1. Download pretrained weights.**

Faket uses a pretrained VGG19 model for neural style transfer.

On most HPC systems, compute nodes do not have internet access, therefore it is recommended to first download the model weights using the login node before running the pipeline. Run the following command inside your environment:

```bash
python - <<'EOF'
from torchvision.models import vgg19, VGG19_Weights
vgg19(weights=VGG19_Weights.DEFAULT)
EOF
```
The weights will be cached locally, and SLURM will automatically locate the cached file.

**2. Running the pipeline.**

The faket-polnet pipeline requires IMOD to be available on your `PATH`.
 - For local systems, this means installing IMOD and ensuring its binaries are discoverable. 
 - For HPC systems, IMOD is typically provided via an environment module and can be loaded inside the job script.

Running the pipeline using the following:
```bash
python pipeline.py /path/to/your/base_directory
```

### Running the Pipeline with Configuration Files
The pipeline.py has been setup to work with TOML configuration files. An example config file can be found at configs/czii.toml. 

An example SLURM submission script can be found at slurm_scripts/sbatch_polnet_faket.sh. 

```bash
python pipeline.py /path/to/config.toml
```

### Parameters

#### Required Arguments
- `base_dir`: Base directory containing simulation and style directories

#### Index Parameters
- `--style_index`: Style index (default: 0)
- `--simulation_index`: Simulation index (default: 0)
- `--train_dir_index`: Train directory index (default: 0)

#### Tilt Series Parameters
- `--tilt_start`: Tilt series start angle (default: -60)
- `--tilt_stop`: Tilt series stop angle (default: 60)
- `--tilt_step`: Tilt series step size (default: 3)

#### Simulation Parameters
- `--detector_snr`: Detector SNR range (default: [0.15, 0.20])

#### Style Transfer Parameters
- `--faket_gpu`: GPU device ID for faket (default: 0)
- `--faket_iterations`: Number of iterations for faket style transfer (default: 5)
- `--faket_step_size`: Step size for faket (default: 0.15)
- `--faket_min_scale`: Minimum scale for faket (default: 630)
- `--faket_end_scale`: End scale for faket (default: 630)
- `--random_faket`: Use random faket style transfer (default: True)
- `--denoised`: Use denoised style micrographs (default: False)

## Pipeline Steps

1. **Directory Validation**: Checks for required simulation and style directories
2. **Style Micrograph Projection**: Projects style tomograms to micrographs (if needed)
3. **Label Transformation**: Processes simulation labels for training
4. **Content Micrograph Projection**: Generates clean and noisy micrographs from simulations
5. **Style Transfer**: Applies neural style transfer using faket
6. **Reconstruction**: Reconstructs tomograms from style-transferred micrographs
7. **Data Organization**: Prepares final training dataset structure

## Output Structure

After successful execution, the pipeline creates:

```
base_directory/
├── micrograph_dir_{index}/                 # Temporary (removed during cleanup)
│   ├── content_micrographs_{index}/
│   │   ├── Micrographs/                    # Projected micrographs
│   │   └── TEM/                            # TEM simulations
│   └── faket_micrographs_{index}/          # Style-transferred tomograms
├── style_micrographs_{index}/              # Final style micrographs
└── train_dir_{index}/
    ├── faket_tomograms/                    # Style-transferred tomograms
    ├── overlay/                            # Labels
    └── snr_list_dir/                       # SNR metadata

```

## Customization

### Adding New Style Sources

1. Place new style tomograms in `base_directory/style_tomograms_{new_index}/`
2. Run the pipeline with `--style_index {new_index}`

### Modifying Style Transfer Parameters

Adjust faket parameters for different style transfer effects:

```bash
python pipeline.py /path/to/base_directory \
    --faket_iterations 10 \
    --faket_step_size 0.1 \
    --faket_min_scale 500 \
    --faket_end_scale 800
```

### Using Pre-projected Style Micrographs

If you already have style micrographs, place them in:
`base_directory/style_micrographs_{index}/`
The pipeline will skip projection and use them directly.

## Troubleshooting

### Common Issues

1. **Missing Dependencies**: Ensure faket and svnet utilities are properly installed
2. **Directory Not Found**: Verify the base directory contains required subdirectories
3. **CUDA Errors**: Check that CUDA is properly configured and the specified GPU is available
4. **Memory Issues**: Reduce batch size or use fewer iterations for style transfer

### Logging

The pipeline provides detailed logging. Key information includes:
- Directory validation status
- Style transfer progress
- Reconstruction steps
- Output file locations


## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## Support

For questions and support, please open an issue on GitHub.

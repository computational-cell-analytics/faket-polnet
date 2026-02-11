#!/bin/bash
CONFIG_DIR=/mnt/lustre-grete/projects/nim00020/sage/data/simulation/deepict_dataset_1/configs

for CONFIG in $CONFIG_DIR/*.toml; do
    JOB_NAME=$(basename "$CONFIG" .toml)
    sbatch --job-name="$JOB_NAME" slurm_scripts/sbatch_polnet_faket.sh "$CONFIG"
done
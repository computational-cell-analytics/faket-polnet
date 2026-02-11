#!/bin/bash
################################################################################
# Batch submit integrated polnet-faket pipeline jobs for multiple configuration files.
#
# Description:
#   This script iterates over all TOML configuration files in a directory and
#   submits a separate Slurm job for each one using the `sbatch_polnet_faket.sh`
#   script.
#
# Usage:
#   1) Set CONFIG_DIR to the folder containing your .toml configuration files.
#   2) Make sure this script is executable, then run with:
#        ./submit_polnet_faket.sh
# 
# Requirements:
#   - polnet-synaptic environment
#   - polnet-faket environment
################################################################################

CONFIG_DIR=/mnt/lustre-grete/projects/nim00020/sage/data/simulation/deepict_dataset_1/configs

for CONFIG in $CONFIG_DIR/*.toml; do
    JOB_NAME=$(basename "$CONFIG" .toml)
    sbatch --job-name="$JOB_NAME" slurm_scripts/sbatch_polnet_faket.sh "$CONFIG"
done

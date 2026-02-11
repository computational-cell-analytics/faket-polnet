#!/bin/bash
################################################################################
# Description: Integrated polnet-faket pipeline for cryo-ET data simulation. 
#              Resource request selected for generating 20 tomograms with CZII shape (630, 630, 184).
# Author: Sage Martineau
# Date: 10-02-2026
#
# Steps:
#   1) polnet: generate tomograms with specified features
#   2) faket: add noise using style transfer
#
# Usage:
#   sbatch sbatch_polnet_faket_config.sh <config.toml>
#
# Resources requested:
#   Partition: grete:interactive
#   Walltime: 8:00:00 if membranes enabled
#             1:30:00 if membranes disabled
#   Nodes: 1
#   CPUs per task: 8
#   Memory: 40G
#   GPU: 1g.20gb
################################################################################

#SBATCH -p grete:interactive
#SBATCH --job-name=JOB_NAME
#SBATCH -o data/simulation/slurm_logs/slurm-%j_%x.out
#SBATCH -t 8:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH -G 1g.20gb

CONFIG=$1

# Step 1 - Polnet
source ~/.bashrc
micromamba activate -p /mnt/lustre-grete/projects/nim00020/sage/envs/polnet-synaptic

SCRIPT_DIR=/mnt/lustre-grete/projects/nim00020/sage/source/polnet-synaptic/scripts/data_gen
cd $SCRIPT_DIR

python all_features_argument.py --config "$CONFIG"

# Step 1 - Faket
module purge
module load gcc/13.2.0
module load imod/5.1.0
export IMOD_DIR=/sw/rev/25.04/rome_mofed_cuda80_rocky8/linux-rocky8-zen2/gcc-13.2.0/imod-5.1.0-ucflk2pud47w7jj27xr5zzitis7kredg
source $IMOD_DIR/IMOD-linux.sh

micromamba activate -p /mnt/lustre-grete/projects/nim00020/sage/envs/faket-polnet

SCRIPT_DIR=/mnt/lustre-grete/projects/nim00020/sage/source/faket-polnet
cd $SCRIPT_DIR

python pipeline.py --config "$CONFIG"

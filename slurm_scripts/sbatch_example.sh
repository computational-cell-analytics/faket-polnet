#!/bin/bash
 
#SBATCH -p grete:shared
#SBATCH --job-name=test_faket
#SBATCH -o data/simulation/faket/slurm_logs/slurm-%j.out
#SBATCH -t 1:00:00
#SBATCH --nodes=1
#SBATCH -G A100:1
 
source ~/.bashrc

# load imod module and source the setup script
module purge
module load gcc/13.2.0
module load imod/5.1.0
export IMOD_DIR=/sw/rev/25.04/rome_mofed_cuda80_rocky8/linux-rocky8-zen2/gcc-13.2.0/imod-5.1.0-ucflk2pud47w7jj27xr5zzitis7kredg
source $IMOD_DIR/IMOD-linux.sh

micromamba activate -p /mnt/lustre-grete/projects/nim00020/sage/envs/faket-polnet

# check that imod commands are accessible
# which xyzproj
# which tilt 
# which alterheader

DATA_DIR='/mnt/lustre-grete/projects/nim00020/sage/data/simulation/faket/deepict'

SCRIPT_DIR=/mnt/lustre-grete/projects/nim00020/sage/source/faket-polnet
cd $SCRIPT_DIR

python pipeline.py $DATA_DIR \
    --micrograph_index 0 \
    --style_index 0 \
    --simulation_index 0 \
    --faket_index 0 \
    --train_dir_index 0 \
    --static_index 0 \
    --tilt_start -60 \
    --tilt_end 61 \
    --tilt_step 2 \
    --detector_snr 0.15 0.20 \
    --simulation_name "deepict_run1" \
    --faket_iterations 5 \
    --faket_step_size 0.15 \
    --random_faket
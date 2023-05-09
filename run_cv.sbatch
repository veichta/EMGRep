#!/bin/bash
#SBATCH --job-name="emgrep-cv"
#SBATCH --mail-type=ALL

#SBATCH --nodes 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem-per-cpu=5000
#SBATCH --gpus=1

#SBATCH --time=05:00:00

#SBATCH --array=1-5

#SBATCH --output="logs/emgrep_cv-%A_%a.out"
#SBATCH --error="logs/emgrep_cv-%A_%a.err"

# Load modules
module load gcc/8.2.0 python_gpu/3.10.4 eth_proxy
pip install -q -r requirements.txt

# Run script
python main.py \
    --data /cluster/scratch/${USER}/nina_db/data/01_raw \
    --log_dir /cluster/scratch/${USER}/nina_db/logs \
    --debug \
    --wandb \
    --device cuda \
    --lr_cpc 2e-4 \
    --encoder_dim 512 \
    --ar_layers 5 \
    --ar_dim 512 \
    --positive_mode none \
    --split_mode subject \
    --val_idx ${SLURM_ARRAY_TASK_ID} \
    --test_idx $((SLURM_ARRAY_TASK_ID % 5 + 1)) 


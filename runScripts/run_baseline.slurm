#!/usr/bin/bash -l

#SBATCH --job-name=run_baseline
#SBATCH --time=0-12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --output=logs/%x_%j.out

module load apptainer

cd $HOME/SwissCrop25
export OMP_NUM_THREADS=1

apptainer exec --nv conda.sif python src/train.py \
    --res_dir storage/baseline/weight_decay05 \
    --num_workers $((SLURM_CPUS_PER_TASK - 1)) \
    --use_temp_calendar --use_temp_subsampling \
    --use_class_balance_loss --beta_class_balance 0.9999 \
    --epochs 15 --batch_size 512 \
    --weight_decay 0.05

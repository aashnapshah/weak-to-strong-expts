#!/bin/bash
#SBATCH --mem 64G
#SBATCH -t 100:00:00  # Requesting 10 days of runtime
#SBATCH -p gpu_quad
#SBATCH --gres=gpu:rtx8000:2

#SBATCH --output=logs/%j.log     # Standard output and error log
#SBATCH -e logs/%j.log
#SBATCH --mail-type=ALL               # Type of email notification- BEGIN, END,FAIL,ALL
#SBATCH --mail-user=aashnashah@g.harvard.edu

module load python/3.10.11
module load gcc/9.2.0
module load cuda/11.7

python train_weak_to_strong.py
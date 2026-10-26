#!/bin/bash
#SBATCH --account=def-kjerbi
#SBATCH --job-name=ema2
#SBATCH --gpus-per-node=nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/alouis/logs_pokair/%x-%j.out
#SBATCH --mail-user=alexandre.louis@umontreal.ca
#SBATCH --mail-type=END,FAIL
module load StdEnv/2023 python/3.11.5
source "$HOME/poker_env/bin/activate"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
cd /scratch/alouis/pokair
python train_spin.py --iters 600 --tables 512 --steps 16384 --minibatch 2048 --pretrain 0 --epochs 2 --clip 0.1 --ent 0.04 --ema 0.99 --p-self 0.5 --save-every 200 --pool-every 25 --pool-size 4 --eval-every 100 --eval-games 1000 --out /scratch/alouis/pokair/ckpt_ema2

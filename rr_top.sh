#!/bin/bash
#SBATCH --account=rrg-kjerbi
#SBATCH --job-name=rrtop
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/alouis/logs_pokair/%x-%j.out
module load StdEnv/2023 python/3.11.5
source "$HOME/poker_env/bin/activate"
cd /scratch/alouis/pokair
python train_spin.py --roundrobin ckpt_rew0b/net_1200.pt ckpt_rew0b/net_2000.pt ckpt_rew0b/net_3600.pt --versus-games 4000

#!/bin/bash
#SBATCH --account=rrg-kjerbi
#SBATCH --job-name=ladder
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/alouis/logs_pokair/%x-%j.out
module load StdEnv/2023 python/3.11.5
source "$HOME/poker_env/bin/activate"
cd /scratch/alouis/pokair
for N in 400 800 1200 1600 2000 2400 2800 3200 3600 4000; do
  echo "== net_$N ($((N + 1200)) cumulees)"
  python train_spin.py --versus ckpt_rew0b/net_$N.pt ckpt_rew0/ref_v8_1200.pt --versus-games 3000 | grep tournois
done

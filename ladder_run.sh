#!/bin/bash
#SBATCH --account=rrg-kjerbi
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/alouis/logs_pokair/%x-%j.out
#SBATCH --mail-user=alexandre.louis@umontreal.ca
#SBATCH --mail-type=END,FAIL
module load StdEnv/2023 python/3.11.5
source "$HOME/poker_env/bin/activate"
cd /scratch/alouis/pokair
RUN=$1
for N in 400 800 1200 1600 2000 2400 2800 3200 3600 4000; do
  r=$(python train_spin.py --versus ckpt_$RUN/net_$N.pt ckpt_rew0/ref_v8_1200.pt --versus-games 3000 | awk '/tournois/{print $5}')
  echo "net_$N $r"
done | tee /tmp/lad_$RUN.txt
awk 'NF==2{s+=$2;q+=$2*$2;n++} END{m=s/n; printf "moyenne %.4f ecart-type %.4f n=%d\n", m, sqrt((q-n*m*m)/(n-1)), n}' /tmp/lad_$RUN.txt

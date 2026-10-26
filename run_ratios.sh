#!/bin/bash
cd /scratch/alouis/pokair
for PS in 0.1 0.5 0.9; do
  TAG="ps${PS/./}"
  mkdir -p "ckpt_$TAG"
  cat > "run_$TAG.sh" <<SBATCH
#!/bin/bash
#SBATCH --account=def-kjerbi
#SBATCH --job-name=$TAG
#SBATCH --gpus-per-node=nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:30:00
#SBATCH --output=/scratch/alouis/logs_pokair/%x-%j.out
#SBATCH --mail-user=alexandre.louis@umontreal.ca
#SBATCH --mail-type=END,FAIL
module load StdEnv/2023 python/3.11.5
source "\$HOME/poker_env/bin/activate"
export OMP_NUM_THREADS=\$SLURM_CPUS_PER_TASK
cd /scratch/alouis/pokair
python train_spin.py --iters 801 --tables 512 --steps 16384 --minibatch 2048 --epochs 2 --clip 0.1 --ent 0.04 --p-self $PS --p-frozen 0.3 --save-every 400 --pool-every 25 --pool-size 4 --eval-every 200 --eval-games 1000 --out /scratch/alouis/pokair/ckpt_$TAG
SBATCH
  bash -n "run_$TAG.sh" || exit 1
  sbatch "run_$TAG.sh"
done

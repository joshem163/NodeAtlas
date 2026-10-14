#!/bin/bash
# One baseline model on ogbn-arxiv under OGB's own time-based split.
#   sbatch baselines_ogbsplit_gpu.slurm GCN
# The split is fixed, so the 10 runs differ only in model initialisation --
# the protocol the OGB leaderboard uses.
#SBATCH --job-name=OGBSPLIT
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=4-00:00:00
#SBATCH --output=logs/ogbsplit_%x_%j.out
#SBATCH --error=logs/ogbsplit_%x_%j.err
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
M=$1
echo "host=$(hostname)  dataset=ogbn-arxiv  split=official  model=$M  start=$(date '+%F %T')"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
mkdir -p results
time python run_baselines_ogbsplit.py --datasets ogbn-arxiv --runs 10 --device cuda \
     --models-only --models $M --out results/arxiv_${M}_ogb10.csv
echo "end=$(date '+%F %T')"

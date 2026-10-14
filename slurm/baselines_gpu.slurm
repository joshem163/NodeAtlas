#!/bin/bash
# One baseline model, 10 runs, stratified 60/20/20 (Tables 3 and 10).
#   sbatch baselines_gpu.slurm cora GCN
# H2GCN and the graph transformers need a large-memory GPU: H2GCN OOMs on a
# 24 GB card for the larger benchmarks, and GPS materialises a dense
# [heads, N, N] attention tensor. Request an 80 GB+ device for those.
#SBATCH --job-name=BASE
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=4-00:00:00
#SBATCH --output=logs/base_%x_%j.out
#SBATCH --error=logs/base_%x_%j.err
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
DS=$1; M=$2
echo "host=$(hostname)  dataset=$DS  model=$M  start=$(date '+%F %T')"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
mkdir -p results
time python run_baselines.py --datasets $DS --runs 10 --device cuda \
     --models-only --models $M --out results/${DS}_${M}_10.csv
echo "end=$(date '+%F %T')"

#!/bin/bash
# NSA scores for one benchmark. NSA is pure numpy/sklearn and uses no GPU;
# every runtime reported in Appendix Table 8 was measured with this script.
#   sbatch nsa_cpu.slurm ogbn-arxiv --split official
#   sbatch nsa_cpu.slurm ogbn-arxiv --split official --probe 10000
#SBATCH --job-name=NSA
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/nsa_%x_%j.out
#SBATCH --error=logs/nsa_%x_%j.err
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
echo "host=$(hostname)  start=$(date '+%F %T')"
time python run_nsa.py "$@"
echo "end=$(date '+%F %T')"

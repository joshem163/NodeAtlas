#!/bin/bash
# Probe-node subsampling study on ogbn-arxiv (Appendix Table 9).
# All sample sizes run back-to-back on one node so the timings are comparable.
# The graph is never modified -- only the number of diagnostic target nodes.
#SBATCH --job-name=NSASUB
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/nsasub_%j.out
#SBATCH --error=logs/nsasub_%j.err
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
echo "host=$(hostname)  start=$(date '+%F %T')"
for n in 10000 40000 60000; do
  python run_nsa.py ogbn-arxiv --split official --probe $n
done
python run_nsa.py ogbn-arxiv --split official
echo "end=$(date '+%F %T')"

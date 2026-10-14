#!/bin/bash
# NSA scores + runtimes for every benchmark, sequentially in one job so the
# times in Appendix Table 8 come from an identical machine state.
#   sbatch nsa_all_cpu.slurm                 # all registered datasets
#   sbatch nsa_all_cpu.slurm cora citeseer   # a subset
#SBATCH --job-name=NSAALL
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/nsaall_%j.out
#SBATCH --error=logs/nsaall_%j.err
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
echo "host=$(hostname)  start=$(date '+%F %T')"
time python run_all_benchmarks.py "$@"
echo "end=$(date '+%F %T')"

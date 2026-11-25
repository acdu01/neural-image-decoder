#!/bin/bash
#SBATCH --job-name=neural-decode
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G
#SBATCH --time=00:45:00
#SBATCH --output=logs/run-%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
.venv/bin/python neural_decoder.py run
.venv/bin/python -m pytest -q

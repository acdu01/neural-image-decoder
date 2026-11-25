#!/bin/bash
#SBATCH --job-name=neural-viewer
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=04:00:00
#SBATCH --output=logs/viewer-%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
printf 'Compute host: %s\n' "$(hostname -f)"
exec .venv/bin/python neural_decoder.py serve --port 8502

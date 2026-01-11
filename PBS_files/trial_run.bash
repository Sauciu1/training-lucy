#!/bin/bash
#PBS -N train_walking_0
#PBS -l select=1:ncpus=50:mem=128gb
#PBS -l walltime=00:20:00
#PBS -j oe

cd "$HOME/projects/training-lucy" || exit 1
export PATH="$HOME/.cargo/bin:$PATH"

uv run -m src.cluster_scripts.cluster_log_port
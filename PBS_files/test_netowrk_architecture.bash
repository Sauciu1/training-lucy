#!/bin/bash
#PBS -N train_walking_0
#PBS -l select=1:ncpus=50:mem=250gb
#PBS -l walltime=00:40:00
#PBS -j oe

cd "$HOME/projects/training-lucy" || exit 1
export PATH="$HOME/.cargo/bin:$PATH"

uv run -m src.cluster_scripts.test_network_architecture
#!/bin/bash
#PBS -N train_walking_0
#PBS -l select=1:ncpus=16:mem=120gb
#PBS -l walltime=03:40:00
#PBS -j oe
# PBS -N test_clip

cd "$HOME/projects/training-lucy" || exit 1
export PATH="$HOME/.cargo/bin:$PATH"

uv run -m src.cluster_scripts.standing_clip_test
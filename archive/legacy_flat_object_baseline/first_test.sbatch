#!/bin/bash
#SBATCH --job-name=aria_gtn_test
#SBATCH --partition=day
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=aria_gtn_test.%J.out
#SBATCH --error=aria_gtn_test.%J.err

echo "This job belongs to the archived flat/object baseline and is no longer runnable" >&2
echo "with the current hierarchical model. Use Training/jobs/train_transformer_v1.sbatch." >&2
exit 2

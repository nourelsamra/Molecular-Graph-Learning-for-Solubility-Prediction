#!/bin/bash

#SBATCH --job-name=esol_gnn
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

#SBATCH --array=0-2

#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00


# ============================================================
# Molecular GNN — SLURM Job Array
#
# Job 0 = GCN
# Job 1 = GAT
# Job 2 = GINE
#
# NOTE:
# Cluster-specific module / environment commands below
# must be adapted to the target HPC system.
# ============================================================


mkdir -p logs
mkdir -p results/slurm


# ------------------------------------------------------------
# OPTIONAL CLUSTER MODULES
# Uncomment / adapt based on the cluster
# ------------------------------------------------------------

# module purge
# module load cuda


# ------------------------------------------------------------
# PYTHON ENVIRONMENT
# Replace this with the actual virtual environment
# on the target cluster.
# ------------------------------------------------------------

# source ~/venvs/molecular-gnn/bin/activate


MODELS=(
    "gcn"
    "gat"
    "gine"
)


MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}


echo "========================================"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Array Task: $SLURM_ARRAY_TASK_ID"
echo "Model: $MODEL"
echo "========================================"


nvidia-smi


python src/train.py \
    --model "$MODEL" \
    --epochs 300 \
    --patience 40 \
    --batch-size 32 \
    --lr 0.001 \
    --seed 42 \
    --output-dir results/slurm


echo "Finished model: $MODEL"

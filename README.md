# Molecular Graph Learning for Solubility Prediction

A molecular machine learning project for predicting aqueous solubility (**logS**) from molecular graphs using Graph Neural Networks.

The project compares **GCN**, **GAT**, and a **bond-aware GINE** model on the ESOL dataset, with emphasis on molecular generalization through **Bemis–Murcko scaffold splitting**, GPU-accelerated training, and reproducible evaluation.

## Research Question

How well do different graph neural network architectures generalize to structurally distinct molecules, and can bond-aware message passing improve molecular property prediction?

## Models

- **GCN** — graph convolutional baseline
- **GAT** — attention-based graph neural network
- **Bond-Aware GINE** — learned atom and bond embeddings with edge-aware message passing

## Dataset

**ESOL (Delaney Solubility Dataset)**

- 1,128 molecules
- Target: aqueous solubility (**logS**)
- Molecular structures represented from SMILES using RDKit and PyTorch Geometric
- Atoms represented as graph nodes
- Chemical bonds represented as graph edges

## Evaluation

### Random Split

| Model | Test RMSE | Test MAE | Test R² |
|---|---:|---:|---:|
| GCN | 0.8744 | 0.6602 | 0.8178 |
| GAT | 0.9065 | 0.6796 | 0.8042 |

### Bemis–Murcko Scaffold Split

The scaffold split prevents molecular scaffolds from overlapping across training, validation, and test sets, providing a more challenging evaluation of structural generalization.

| Model | Validation RMSE | Test RMSE | Test MAE | Test R² |
|---|---:|---:|---:|---:|
| GCN | 1.3672 | 1.2512 | 0.9500 | 0.6103 |
| GAT | 1.1908 | 1.2349 | 0.9745 | 0.6204 |
| **Bond-Aware GINE** | **1.1646** | **1.1362** | **0.8917** | **0.6786** |

The bond-aware GINE model achieved the strongest overall performance on the held-out scaffold test set.

## Key Findings

- Scaffold-based evaluation was more challenging than random splitting, highlighting the difficulty of generalizing to structurally distinct molecules.
- GAT achieved stronger scaffold-validation performance than GCN.
- Bond-aware GINE achieved the best scaffold-test performance with **RMSE = 1.1362** and **R² = 0.6786**.
- Molecular-level error analysis showed that GCN and GAT perform differently across individual molecules.
- No strong linear relationship was observed between relative GCN/GAT error and common molecular descriptors such as molecular weight, LogP, TPSA, ring count, or rotatable bonds.

## Training Pipeline

The training pipeline includes:

- CUDA/GPU acceleration
- Adam optimization
- Early stopping
- Best-validation checkpointing
- Reproducible random seeds
- RMSE, MAE, and R² evaluation

## HPC / SLURM Support

The project includes a standalone training pipeline and a **SLURM job-array configuration** for reproducible execution on HPC systems.

The SLURM configuration is provided as an HPC-ready template for deployment on a compatible cluster.

Example:

```bash
sbatch slurm/train_models.slurm
```

The job array is configured for:

- GCN
- GAT
- Bond-Aware GINE

## Running the Project

Install dependencies:

```bash
pip install -r requirements.txt
```

Train a model:

```bash
python src/train.py --model gcn
python src/train.py --model gat
python src/train.py --model gine
```

Optional parameters include:

```bash
--epochs 300
--patience 40
--batch-size 32
--lr 0.001
--seed 42
```

## Project Structure

```text
Molecular_GNN_ESOL_Project/
│
├── src/
│   └── train.py
│
├── models/
│   ├── gcn_scaffold.pt
│   ├── gat_scaffold.pt
│   └── gine_scaffold.pt
│
├── results/
│   ├── scaffold_results.csv
│   ├── random_vs_scaffold.csv
│   ├── error_descriptor_analysis.csv
│   └── config.json
│
├── figures/
│   ├── scaffold_predictions.png
│   └── validation_curves.png
│
├── slurm/
│   └── train_models.slurm
│
├── requirements.txt
├── .gitignore
└── README.md
```

## Technologies

Python · PyTorch · PyTorch Geometric · RDKit · CUDA · scikit-learn · Pandas · NumPy · Matplotlib · SLURM

## Future Work

- Evaluation on additional molecular datasets
- Multiple-seed experiments for uncertainty estimation
- Hyperparameter optimization
- Deeper molecular GNN architectures
- Large-scale execution on an HPC cluster 

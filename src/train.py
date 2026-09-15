
import os
import json
import math
import copy
import random
import argparse
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from torch import nn
from torch.utils.data import Subset

from torch_geometric.datasets import MoleculeNet
from torch_geometric.loader import DataLoader
from torch_geometric.nn import (
    GCNConv,
    GATConv,
    GINEConv,
    global_mean_pool
)

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# SCAFFOLD SPLIT
# ============================================================

def get_scaffold(smiles):

    mol = Chem.MolFromSmiles(smiles)

    return MurckoScaffold.MurckoScaffoldSmiles(
        mol=mol,
        includeChirality=False
    )


def scaffold_split(dataset):

    scaffold_to_indices = defaultdict(list)

    for idx, data in enumerate(dataset):

        scaffold = get_scaffold(
            data.smiles
        )

        scaffold_to_indices[
            scaffold
        ].append(idx)


    groups = sorted(
        scaffold_to_indices.values(),
        key=lambda x: (-len(x), x[0])
    )


    n = len(dataset)

    train_cutoff = int(0.80 * n)
    val_cutoff = int(0.90 * n)


    train_indices = []
    val_indices = []
    test_indices = []


    for group in groups:

        if (
            len(train_indices)
            + len(group)
            <= train_cutoff
        ):

            train_indices.extend(group)


        elif (
            len(train_indices)
            + len(val_indices)
            + len(group)
            <= val_cutoff
        ):

            val_indices.extend(group)


        else:

            test_indices.extend(group)


    return (
        Subset(dataset, train_indices),
        Subset(dataset, val_indices),
        Subset(dataset, test_indices)
    )


# ============================================================
# GCN
# ============================================================

class GCN(nn.Module):

    def __init__(
        self,
        num_node_features,
        hidden_channels=64
    ):

        super().__init__()

        self.conv1 = GCNConv(
            num_node_features,
            hidden_channels
        )

        self.conv2 = GCNConv(
            hidden_channels,
            hidden_channels
        )

        self.lin1 = nn.Linear(
            hidden_channels,
            32
        )

        self.lin2 = nn.Linear(
            32,
            1
        )


    def forward(
        self,
        x,
        edge_index,
        batch,
        edge_attr=None
    ):

        x = self.conv1(
            x.float(),
            edge_index
        )

        x = F.relu(x)

        x = self.conv2(
            x,
            edge_index
        )

        x = F.relu(x)

        x = global_mean_pool(
            x,
            batch
        )

        x = F.relu(
            self.lin1(x)
        )

        return self.lin2(x)


# ============================================================
# GAT
# ============================================================

class GAT(nn.Module):

    def __init__(
        self,
        num_node_features,
        hidden_channels=64,
        heads=4
    ):

        super().__init__()

        self.gat1 = GATConv(
            num_node_features,
            hidden_channels,
            heads=heads,
            concat=True
        )

        self.gat2 = GATConv(
            hidden_channels * heads,
            hidden_channels,
            heads=1,
            concat=False
        )

        self.lin1 = nn.Linear(
            hidden_channels,
            32
        )

        self.lin2 = nn.Linear(
            32,
            1
        )


    def forward(
        self,
        x,
        edge_index,
        batch,
        edge_attr=None
    ):

        x = self.gat1(
            x.float(),
            edge_index
        )

        x = F.elu(x)

        x = self.gat2(
            x,
            edge_index
        )

        x = F.elu(x)

        x = global_mean_pool(
            x,
            batch
        )

        x = F.relu(
            self.lin1(x)
        )

        return self.lin2(x)


# ============================================================
# CATEGORICAL ENCODER FOR GINE
# ============================================================

class CategoricalEncoder(nn.Module):

    def __init__(
        self,
        cardinalities,
        embedding_dim
    ):

        super().__init__()

        self.embeddings = nn.ModuleList([

            nn.Embedding(
                cardinality,
                embedding_dim
            )

            for cardinality
            in cardinalities
        ])


        for embedding in self.embeddings:

            nn.init.xavier_uniform_(
                embedding.weight
            )


    def forward(
        self,
        features
    ):

        encoded = None

        for i, embedding in enumerate(
            self.embeddings
        ):

            current = embedding(
                features[:, i].long()
            )

            if encoded is None:
                encoded = current

            else:
                encoded = (
                    encoded + current
                )

        return encoded


# ============================================================
# BOND-AWARE GINE
# ============================================================

class BondAwareGINE(nn.Module):

    def __init__(
        self,
        node_cardinalities,
        edge_cardinalities,
        hidden_channels=64,
        dropout=0.10
    ):

        super().__init__()

        self.dropout = dropout


        self.atom_encoder = CategoricalEncoder(
            node_cardinalities,
            hidden_channels
        )


        self.bond_encoder = CategoricalEncoder(
            edge_cardinalities,
            hidden_channels
        )


        mlp1 = nn.Sequential(

            nn.Linear(
                hidden_channels,
                hidden_channels
            ),

            nn.ReLU(),

            nn.Linear(
                hidden_channels,
                hidden_channels
            )
        )


        mlp2 = nn.Sequential(

            nn.Linear(
                hidden_channels,
                hidden_channels
            ),

            nn.ReLU(),

            nn.Linear(
                hidden_channels,
                hidden_channels
            )
        )


        self.conv1 = GINEConv(
            mlp1
        )

        self.conv2 = GINEConv(
            mlp2
        )


        self.lin1 = nn.Linear(
            hidden_channels,
            32
        )

        self.lin2 = nn.Linear(
            32,
            1
        )


    def forward(
        self,
        x,
        edge_index,
        batch,
        edge_attr=None
    ):

        x = self.atom_encoder(
            x
        )

        edge_attr = self.bond_encoder(
            edge_attr
        )


        x = self.conv1(
            x,
            edge_index,
            edge_attr
        )

        x = F.relu(x)

        x = F.dropout(
            x,
            p=self.dropout,
            training=self.training
        )


        x = self.conv2(
            x,
            edge_index,
            edge_attr
        )

        x = F.relu(x)


        x = global_mean_pool(
            x,
            batch
        )


        x = F.relu(
            self.lin1(x)
        )


        return self.lin2(x)


# ============================================================
# FEATURE CARDINALITIES
# ============================================================

def get_cardinalities(dataset):

    node_cardinalities = []

    edge_cardinalities = []


    for feature_idx in range(
        dataset.num_node_features
    ):

        maximum = max(

            int(
                data.x[
                    :,
                    feature_idx
                ].max()
            )

            for data in dataset
        )

        node_cardinalities.append(
            maximum + 1
        )


    for feature_idx in range(
        dataset.num_edge_features
    ):

        maximum = max(

            int(
                data.edge_attr[
                    :,
                    feature_idx
                ].max()
            )

            for data in dataset

            if data.edge_attr.numel() > 0
        )

        edge_cardinalities.append(
            maximum + 1
        )


    return (
        node_cardinalities,
        edge_cardinalities
    )


# ============================================================
# EVALUATION
# ============================================================

@torch.no_grad()
def predict(
    model,
    loader,
    device
):

    model.eval()

    predictions = []
    targets = []


    for batch in loader:

        batch = batch.to(device)


        output = model(

            batch.x,

            batch.edge_index,

            batch.batch,

            batch.edge_attr
        )


        predictions.extend(

            output
            .cpu()
            .numpy()
            .flatten()
        )


        targets.extend(

            batch.y
            .cpu()
            .numpy()
            .flatten()
        )


    return (
        np.array(targets),
        np.array(predictions)
    )


def metrics(
    y_true,
    y_pred
):

    rmse = math.sqrt(

        mean_squared_error(
            y_true,
            y_pred
        )
    )


    mae = mean_absolute_error(
        y_true,
        y_pred
    )


    r2 = r2_score(
        y_true,
        y_pred
    )


    return {
        "rmse": float(rmse),
        "mae": float(mae),
        "r2": float(r2)
    }


# ============================================================
# TRAINING
# ============================================================

def train_model(
    model,
    train_loader,
    val_loader,
    device,
    lr,
    max_epochs,
    patience
):

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr
    )


    criterion = nn.MSELoss()


    best_val_rmse = float("inf")

    best_epoch = 0

    best_state = None

    no_improvement = 0


    history = []


    for epoch in range(
        1,
        max_epochs + 1
    ):

        model.train()

        total_loss = 0


        for batch in train_loader:

            batch = batch.to(device)

            optimizer.zero_grad()


            predictions = model(

                batch.x,

                batch.edge_index,

                batch.batch,

                batch.edge_attr
            )


            loss = criterion(
                predictions,
                batch.y.float()
            )


            loss.backward()

            optimizer.step()


            total_loss += (
                loss.item()
                * batch.num_graphs
            )


        train_mse = (
            total_loss
            / len(train_loader.dataset)
        )


        val_true, val_pred = predict(
            model,
            val_loader,
            device
        )


        val_rmse = math.sqrt(

            mean_squared_error(
                val_true,
                val_pred
            )
        )


        history.append({

            "epoch": epoch,

            "train_mse": train_mse,

            "val_rmse": val_rmse
        })


        if val_rmse < best_val_rmse:

            best_val_rmse = val_rmse

            best_epoch = epoch

            best_state = copy.deepcopy(
                model.state_dict()
            )

            no_improvement = 0


        else:

            no_improvement += 1


        if (
            epoch == 1
            or epoch % 10 == 0
        ):

            print(

                f"Epoch {epoch:03d} | "

                f"Train MSE: "
                f"{train_mse:.4f} | "

                f"Validation RMSE: "
                f"{val_rmse:.4f}"
            )


        if no_improvement >= patience:

            print(
                f"Early stopping "
                f"at epoch {epoch}"
            )

            break


    model.load_state_dict(
        best_state
    )


    return (
        model,
        best_epoch,
        best_val_rmse,
        history
    )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()


    parser.add_argument(
        "--model",
        choices=[
            "gcn",
            "gat",
            "gine"
        ],
        required=True
    )


    parser.add_argument(
        "--epochs",
        type=int,
        default=300
    )


    parser.add_argument(
        "--patience",
        type=int,
        default=40
    )


    parser.add_argument(
        "--batch-size",
        type=int,
        default=32
    )


    parser.add_argument(
        "--lr",
        type=float,
        default=0.001
    )


    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )


    parser.add_argument(
        "--data-root",
        default="data/ESOL"
    )


    parser.add_argument(
        "--output-dir",
        default="results/run"
    )


    args = parser.parse_args()


    os.makedirs(
        args.output_dir,
        exist_ok=True
    )


    set_seed(
        args.seed
    )


    device = torch.device(

        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


    print(
        "Device:",
        device
    )


    if torch.cuda.is_available():

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )


    dataset = MoleculeNet(
        root=args.data_root,
        name="ESOL"
    )


    train_dataset, \
    val_dataset, \
    test_dataset = scaffold_split(
        dataset
    )


    generator = torch.Generator()

    generator.manual_seed(
        args.seed
    )


    train_loader = DataLoader(

        train_dataset,

        batch_size=args.batch_size,

        shuffle=True,

        generator=generator
    )


    val_loader = DataLoader(

        val_dataset,

        batch_size=args.batch_size,

        shuffle=False
    )


    test_loader = DataLoader(

        test_dataset,

        batch_size=args.batch_size,

        shuffle=False
    )


    node_cardinalities, \
    edge_cardinalities = get_cardinalities(
        dataset
    )


    if args.model == "gcn":

        model = GCN(
            dataset.num_node_features
        )


    elif args.model == "gat":

        model = GAT(
            dataset.num_node_features
        )


    else:

        model = BondAwareGINE(

            node_cardinalities,

            edge_cardinalities
        )


    model = model.to(
        device
    )


    print(
        "\nModel:",
        args.model.upper()
    )


    print(
        "Trainable parameters:",
        sum(
            p.numel()
            for p in model.parameters()
            if p.requires_grad
        )
    )


    model, \
    best_epoch, \
    best_val_rmse, \
    history = train_model(

        model,

        train_loader,

        val_loader,

        device,

        args.lr,

        args.epochs,

        args.patience
    )


    test_true, \
    test_pred = predict(

        model,

        test_loader,

        device
    )


    test_metrics = metrics(
        test_true,
        test_pred
    )


    results = {

        "model": args.model,

        "device": str(device),

        "seed": args.seed,

        "best_epoch": best_epoch,

        "best_validation_rmse":
            float(best_val_rmse),

        "test_rmse":
            test_metrics["rmse"],

        "test_mae":
            test_metrics["mae"],

        "test_r2":
            test_metrics["r2"]
    }


    print(
        "\n========== FINAL RESULTS =========="
    )


    print(
        json.dumps(
            results,
            indent=4
        )
    )


    torch.save(

        model.state_dict(),

        os.path.join(
            args.output_dir,
            f"{args.model}_model.pt"
        )
    )


    with open(

        os.path.join(
            args.output_dir,
            f"{args.model}_results.json"
        ),

        "w"

    ) as f:

        json.dump(
            results,
            f,
            indent=4
        )


    pd.DataFrame(
        history
    ).to_csv(

        os.path.join(
            args.output_dir,
            f"{args.model}_history.csv"
        ),

        index=False
    )


    pd.DataFrame({

        "actual_logS":
            test_true,

        "predicted_logS":
            test_pred

    }).to_csv(

        os.path.join(
            args.output_dir,
            f"{args.model}_predictions.csv"
        ),

        index=False
    )


if __name__ == "__main__":

    main()

from typing import Dict, List, Optional, Tuple

import logging
import warnings
from dataclasses import dataclass

import dask
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from dask.distributed import Client, LocalCluster
from numpy.typing import NDArray
from pytorch_lightning.utilities.warnings import PossibleUserWarning

from .data_partitioning import create_ensemble_splits
from .rff_classifier import RFFClassifier, RFFDataModule


@dataclass
class DataSplit:
    """Container for a train/test split"""

    X_train: NDArray
    y_train: NDArray
    X_test: NDArray
    y_test: NDArray
    name: str
    split_id: int


@dataclass
class ScanResult:
    """Container for scan results"""

    length_scale: float
    dataset_name: str
    model_id: int
    train_loss: float
    val_loss: float
    val_acc: float
    model_state: dict
    input_dim: int
    n_classes: int
    n_features: int


def train_single_model(
    data_split: DataSplit,
    length_scale: float,
    model_id: int,
    n_features: int = 1024,
    max_epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    accelerator: str = "cpu",
    num_workers: int = 0,
) -> ScanResult:
    """
    Train a single RFF model with given parameters.

    Args:
        data_split: Train/test data
        length_scale: RBF kernel length scale
        model_id: Identifier for this model instance
        n_features: Number of random features
        max_epochs: Maximum training epochs
        batch_size: Training batch size
        learning_rate: Learning rate
        accelerator: 'cpu' or 'gpu' for training
        num_workers: Number of workers for data loading

    Returns:
        ScanResult: Training results
    """
    input_dim = data_split.X_train.shape[1]
    n_classes = len(np.unique(data_split.y_train))

    model = RFFClassifier(
        input_dim=input_dim,
        n_classes=n_classes,
        n_features=n_features,
        length_scale=length_scale,
        learning_rate=learning_rate,
    )

    # Create data module with num_workers
    data = RFFDataModule(
        X_train=data_split.X_train,
        y_train=data_split.y_train,
        X_val=data_split.X_test,
        y_val=data_split.y_test,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    # Create trainer with specified accelerator
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator=accelerator,
        devices=1,
        enable_progress_bar=False,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
    )

    # Train model
    trainer.fit(model, data)

    # Get final metrics
    metrics = trainer.callback_metrics
    train_loss = float(metrics.get("train_loss", 0))
    val_loss = float(metrics.get("val_loss", 0))
    val_acc = float(metrics.get("val_acc", 0))

    return ScanResult(
        length_scale=length_scale,
        dataset_name=data_split.name,
        model_id=model_id,
        train_loss=train_loss,
        val_loss=val_loss,
        val_acc=val_acc,
        model_state=model.state_dict(),
        input_dim=input_dim,
        n_classes=n_classes,
        n_features=n_features,
    )


def parameter_scan(
    data_splits: list[DataSplit],
    length_scales: list[float],
    n_models: int = 5,
    n_features: int = 1024,
    max_epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    client: Optional[Client] = None,
    accelerator: str = "cpu",
    num_workers: int = 0,
) -> list[ScanResult]:
    """Perform distributed parameter scan."""
    # Create tasks for all combinations
    tasks = []
    for split in data_splits:
        for length_scale in length_scales:
            for model_id in range(n_models):
                task = dask.delayed(train_single_model)(
                    data_split=split,
                    length_scale=length_scale,
                    model_id=model_id,
                    n_features=n_features,
                    max_epochs=max_epochs,
                    batch_size=batch_size,
                    learning_rate=learning_rate,
                    accelerator=accelerator,
                    num_workers=num_workers,
                )
                tasks.append(task)

    # Compute results
    results = client.compute(tasks)
    results = client.gather(results)

    return list(results)


def analyze_results(results: list[ScanResult], n_sigma: float = 3.0) -> dict:
    """
    Analyze scan results and find models within n_sigma of mean best length scale.

    Args:
        results: List of scan results
        n_sigma: Number of standard deviations for threshold (default: 3.0)

    Returns:
        Dict: Analysis of results including best length scales and filtered model indices
    """
    # Convert to numpy arrays for easier analysis
    length_scales = np.array([r.length_scale for r in results])
    datasets = np.array([r.dataset_name for r in results])
    val_accs = np.array([r.val_acc for r in results])

    # Get unique values
    unique_scales = np.unique(length_scales)
    unique_datasets = np.unique(datasets)

    # Store best length scales for each dataset
    best_length_scales = []

    # Compute statistics
    analysis = {}
    for dataset in unique_datasets:
        analysis[dataset] = {}
        dataset_mask = datasets == dataset
        dataset_scales = length_scales[dataset_mask]
        dataset_accs = val_accs[dataset_mask]

        # Find best performing length scale for this dataset
        best_idx = np.argmax(dataset_accs)
        best_length_scales.append(dataset_scales[best_idx])

        for scale in unique_scales:
            mask = (datasets == dataset) & (length_scales == scale)
            accs = val_accs[mask]

            analysis[dataset][scale] = {
                "mean_acc": np.mean(accs),
                "std_acc": np.std(accs),
                "min_acc": np.min(accs),
                "max_acc": np.max(accs),
                "n_models": len(accs),
            }

    # Compute mean and std of best length scales
    mean_best_scale = np.median(best_length_scales)
    std_best_scale = np.std(best_length_scales)

    # Find models within n_sigma of mean best length scale
    lower_bound = mean_best_scale - n_sigma * std_best_scale
    upper_bound = mean_best_scale + n_sigma * std_best_scale

    # Get indices of models within bounds
    filtered_indices = [
        i
        for i, result in enumerate(results)
        if lower_bound <= result.length_scale <= upper_bound
    ]

    # Add length scale statistics and filtered indices to analysis
    analysis["length_scale_stats"] = {
        "mean_best_scale": mean_best_scale,
        "std_best_scale": std_best_scale,
        "best_scales": best_length_scales,
        "sigma_threshold": n_sigma,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "filtered_model_indices": filtered_indices,
    }

    return analysis


def create_scan_splits(
    X: NDArray,
    y: NDArray,
    n_splits: int = 5,
    cal_size: float = 0.2,
    test_size: float = 0.2,
    min_samples_per_class: int = 5,
    random_state: Optional[int] = None,
) -> tuple[list[DataSplit], dict]:
    """
    Create data splits for parameter scanning.

    Args:
        X: Input features
        y: Target labels
        n_splits: Number of ensemble splits
        cal_size: Proportion of data for calibration
        test_size: Proportion of working data for testing
        min_samples_per_class: Minimum samples per class
        random_state: Random seed

    Returns:
        Tuple[List[DataSplit], Dict]: List of splits for scanning and calibration data
    """
    # Create ensemble splits
    split_data = create_ensemble_splits(
        X=X,
        y=y,
        n_splits=n_splits,
        cal_size=cal_size,
        test_size=test_size,
        min_samples_per_class=min_samples_per_class,
        random_state=random_state,
    )

    # Extract calibration set
    calibration_data = split_data["calibration"]

    # Create DataSplit objects for each ensemble split
    scan_splits = []
    for i, split in enumerate(split_data["ensemble"]):
        split_obj = DataSplit(
            X_train=split["train"]["X"],
            y_train=split["train"]["y"],
            X_test=split["test"]["X"],
            y_test=split["test"]["y"],
            name=f"split_{i+1}",
            split_id=i,
        )
        scan_splits.append(split_obj)

    return scan_splits, calibration_data


def run_parameter_scan(
    X: NDArray,
    y: NDArray,
    length_scales: list[float],
    n_splits: int = 5,
    n_models_per_split: int = 3,
    cal_size: float = 0.2,
    test_size: float = 0.2,
    min_samples_per_class: int = 5,
    n_features: int = 1024,
    max_epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    random_state: Optional[int] = None,
    client: Optional[Client] = None,
    n_workers: int = 4,
    accelerator: str = "cpu",
    memory_limit: str = "4GB",
    dataloader_workers: int = 0,
    verbose: bool = False,
) -> tuple[list[ScanResult], dict]:
    """Run complete parameter scan with proper data splitting."""

    # Suppress warnings and logging if not verbose
    if not verbose:
        warnings.filterwarnings("ignore", category=PossibleUserWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
        logging.getLogger("distributed").setLevel(logging.ERROR)
        logging.getLogger("distributed.worker").setLevel(logging.ERROR)

    # Create data splits
    scan_splits, calibration_data = create_scan_splits(
        X=X,
        y=y,
        n_splits=n_splits,
        cal_size=cal_size,
        test_size=test_size,
        min_samples_per_class=min_samples_per_class,
        random_state=random_state,
    )

    # Run parameter scan
    if client is None:
        cluster = LocalCluster(
            n_workers=n_workers,
            threads_per_worker=1,
            processes=True,
            scheduler_port=0,
            silence_logs=False,
            memory_limit=memory_limit,
        )

        with Client(cluster) as client:
            results = parameter_scan(
                data_splits=scan_splits,
                length_scales=length_scales,
                n_models=n_models_per_split,
                n_features=n_features,
                max_epochs=max_epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                client=client,
                accelerator=accelerator,
                num_workers=dataloader_workers,
            )

        cluster.close()
    else:
        results = parameter_scan(
            data_splits=scan_splits,
            length_scales=length_scales,
            n_models=n_models_per_split,
            n_features=n_features,
            max_epochs=max_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            client=client,
            accelerator=accelerator,
            num_workers=dataloader_workers,
        )

    return results, calibration_data


def reconstruct_model(result: ScanResult) -> RFFClassifier:
    """Reconstruct a model from scan results"""
    model = RFFClassifier(
        input_dim=result.input_dim,
        n_classes=result.n_classes,
        n_features=result.n_features,
        length_scale=result.length_scale,
    )
    model.load_state_dict(result.model_state)
    model.eval()  # Set to evaluation mode
    return model


def evaluate_models_at_coordinates(
    coordinates: NDArray, scan_results: list[ScanResult]
) -> pd.DataFrame:
    """
    Evaluate multiple models at given coordinates and return averaged predictions.

    Args:
        coordinates: Array of shape (n_points, 2) containing XY coordinates
        scan_results: List of ScanResult objects containing trained models

    Returns:
        pd.DataFrame: DataFrame containing coordinates and probability values for each class
    """
    # Convert input to tensor
    X = torch.FloatTensor(coordinates)

    # Storage for all model predictions
    all_predictions = []

    # Reconstruct and evaluate each model
    for result in scan_results:
        model = reconstruct_model(result)
        with torch.no_grad():
            logits = model(X)
            probs = torch.softmax(logits, dim=1)
            all_predictions.append(probs.numpy())

    # Average predictions across models
    avg_predictions = np.mean(all_predictions, axis=0)

    # Renormalize to ensure probabilities sum to 1
    row_sums = avg_predictions.sum(axis=1, keepdims=True)
    avg_predictions = avg_predictions / row_sums

    # Create DataFrame
    df = pd.DataFrame(coordinates, columns=["x", "y"])

    # Add probability columns for each class
    for i in range(avg_predictions.shape[1]):
        df[f"prob_class_{i}"] = avg_predictions[:, i]

    return df

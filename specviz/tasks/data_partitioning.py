from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray


def stratified_split(
    X: NDArray,
    y: NDArray,
    test_size: float = 0.2,
    min_samples_per_class: int = 5,
    random_state: Optional[int] = None,
) -> tuple[NDArray, NDArray]:
    """
    Create a stratified split with minimum samples per class using sampling with replacement if needed.

    Args:
        X: Input features
        y: Target labels
        test_size: Proportion of data to use for calibration
        min_samples_per_class: Minimum number of samples required for each class in both sets
        random_state: Random seed for reproducibility

    Returns:
        tuple: (train_indices, calibration_indices)
    """
    unique_classes = np.unique(y)
    train_idx = []
    cal_idx = []

    rng = np.random.RandomState(random_state)

    for class_label in unique_classes:
        # Get indices for this class
        class_indices = np.where(y == class_label)[0]
        n_samples = len(class_indices)

        # Calculate desired number of calibration samples
        n_cal_desired = max(min_samples_per_class, int(n_samples * test_size))
        n_train_desired = max(min_samples_per_class, n_samples - n_cal_desired)

        # If we don't have enough samples, use sampling with replacement
        if n_samples < (n_cal_desired + n_train_desired):
            print(
                f"Warning: Class {class_label} has only {n_samples} samples. "
                f"Using sampling with replacement to get {n_cal_desired + n_train_desired} samples."
            )

            # Sample with replacement for both sets
            cal_samples = rng.choice(class_indices, size=n_cal_desired, replace=True)
            train_samples = rng.choice(
                class_indices, size=n_train_desired, replace=True
            )

            cal_idx.extend(cal_samples)
            train_idx.extend(train_samples)
        else:
            # Regular stratified split if we have enough samples
            shuffled_indices = class_indices[rng.permutation(len(class_indices))]
            cal_idx.extend(shuffled_indices[:n_cal_desired])
            train_idx.extend(shuffled_indices[n_cal_desired:])

    # Convert to arrays
    train_idx = np.array(train_idx)
    cal_idx = np.array(cal_idx)

    # Print split statistics
    train_dist = {c: np.sum(y[train_idx] == c) for c in unique_classes}
    cal_dist = {c: np.sum(y[cal_idx] == c) for c in unique_classes}

    print("\nSplit distribution:")
    print(f"Training set: {train_dist}")
    print(f"Calibration set: {cal_dist}")
    print(
        f"Using sampling with replacement: {len(set(train_idx)) + len(set(cal_idx)) < len(train_idx) + len(cal_idx)}"
    )

    return train_idx, cal_idx


def create_ensemble_splits(
    X: NDArray,
    y: NDArray,
    n_splits: int = 5,
    cal_size: float = 0.2,
    test_size: float = 0.2,
    min_samples_per_class: int = 5,
    random_state: Optional[int] = None,
) -> dict[str, dict]:
    """
    Create ensemble splits for conformal prediction with selection arrays.
    """
    rng = np.random.RandomState(random_state)

    # First, create the calibration split
    working_idx, cal_idx = stratified_split(
        X,
        y,
        test_size=cal_size,
        min_samples_per_class=min_samples_per_class,
        random_state=rng.randint(1e6),
    )
    print(cal_idx.shape)
    print(working_idx.shape)
    # Create calibration set with indices
    calibration_set = {"X": X[cal_idx], "y": y[cal_idx], "indices": cal_idx}

    # Get class-specific indices for working set
    unique_classes = np.unique(y)
    working_class_indices = {}

    for c in unique_classes:
        # Find indices in working set for this class
        class_indices = np.where(y[working_idx] == c)[0]
        # Map back to original indices
        working_class_indices[c] = working_idx[class_indices]

    # Create ensemble splits from working set
    ensemble_splits = []

    for split in range(n_splits):
        train_idx = []
        test_idx = []

        # Split each class separately to maintain proportions
        for class_label in unique_classes:
            class_indices = working_class_indices[class_label]
            n_samples = len(class_indices)
            n_test = int(np.ceil(n_samples * test_size))

            # Sample test indices
            test_samples = rng.choice(class_indices, size=n_test, replace=True)

            # Sample train indices
            train_samples = rng.choice(
                class_indices, size=n_samples - n_test, replace=True
            )

            train_idx.extend(train_samples)
            test_idx.extend(test_samples)

        # Shuffle both sets
        train_idx = np.array(train_idx)
        test_idx = np.array(test_idx)
        rng.shuffle(train_idx)
        rng.shuffle(test_idx)

        # Store split data
        split_data = {
            "train": {"X": X[train_idx], "y": y[train_idx], "indices": train_idx},
            "test": {"X": X[test_idx], "y": y[test_idx], "indices": test_idx},
        }
        ensemble_splits.append(split_data)

        # Print distributions
        train_dist = {c: np.sum(y[train_idx] == c) for c in unique_classes}
        test_dist = {c: np.sum(y[test_idx] == c) for c in unique_classes}
        print(f"\nSplit {split + 1} distribution:")
        print(f"Training set: {train_dist}")
        print(f"Test set: {test_dist}")

        # Print unique samples info
        n_unique_train = len(np.unique(train_idx))
        n_unique_test = len(np.unique(test_idx))
        print(
            f"Unique training samples: {n_unique_train}/{len(train_idx)} "
            f"({100*n_unique_train/len(train_idx):.1f}%)"
        )
        print(
            f"Unique test samples: {n_unique_test}/{len(test_idx)} "
            f"({100*n_unique_test/len(test_idx):.1f}%)"
        )

    return {"calibration": calibration_set, "ensemble": ensemble_splits}

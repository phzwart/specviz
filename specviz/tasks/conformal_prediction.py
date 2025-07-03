from typing import Dict, List, Set, Tuple

import numpy as np
from numpy.typing import NDArray


class OneClassConformalPredictor:
    """
    One-class-at-a-time conformal predictor that guarantees
    class-conditional coverage by treating each class independently.
    """

    def __init__(self, confidence_level: float = 0.95):
        """
        Initialize conformal predictor.

        Args:
            confidence_level: Desired confidence level per class (e.g., 0.95 for 95%)
        """
        if not 0 < confidence_level < 1:
            raise ValueError("Confidence level must be between 0 and 1")

        self.confidence_level = confidence_level
        self.thresholds: dict[int, float] = {}
        self.is_calibrated = False

    def calibrate(
        self, cal_probabilities: NDArray, cal_labels: NDArray
    ) -> "OneClassConformalPredictor":
        """
        Calibrate each class independently using class probabilities.

        Args:
            cal_probabilities: Calibration set probabilities (n_samples, n_classes)
            cal_labels: True labels for calibration set
        """
        # Ensure arrays are numpy arrays
        cal_probabilities = np.asarray(cal_probabilities)
        cal_labels = np.asarray(cal_labels)

        # Get unique classes from both labels and probability matrix shape
        n_classes = cal_probabilities.shape[1]
        unique_classes = np.unique(cal_labels)

        # Calculate threshold for each class independently
        for class_idx in range(n_classes):
            # Get samples for this class
            class_mask = cal_labels == class_idx

            if not np.any(class_mask):
                continue

            # Get probabilities for this class using boolean indexing
            class_probs = cal_probabilities[class_mask][:, class_idx]

            # Calculate threshold using class probabilities directly
            # For conformal prediction, we want the (1-confidence_level) percentile
            # Apply finite sample correction: adjust alpha by n/(n+1)
            n_samples = len(class_probs)
            alpha = 1 - self.confidence_level
            adjusted_alpha = alpha * (n_samples / (n_samples + 1))
            threshold_idx = int(np.ceil(n_samples * adjusted_alpha)) - 1
            threshold_idx = min(threshold_idx, n_samples - 1)

            # Sort probabilities and get threshold
            sorted_probs = np.sort(class_probs)
            self.thresholds[class_idx] = sorted_probs[threshold_idx]

        self.is_calibrated = True
        return self

    def __call__(self, probabilities: NDArray) -> tuple[list[set[int]], NDArray]:
        """
        Get prediction sets by comparing each class probability to its threshold.

        Args:
            probabilities: Class probabilities (n_samples, n_classes)

        Returns:
            Tuple[List[Set[int]], NDArray]: Prediction sets and input probabilities
        """
        if not self.is_calibrated:
            raise RuntimeError("Predictor must be calibrated before prediction")

        probabilities = np.asarray(probabilities)
        n_samples = probabilities.shape[0]
        prediction_sets = []

        for i in range(n_samples):
            pred_set = set()

            # Check each class independently
            for class_idx, threshold in self.thresholds.items():
                if probabilities[i, class_idx] >= threshold:
                    pred_set.add(class_idx)

            # If empty prediction set, add highest probability class
            if not pred_set:
                pred_set.add(np.argmax(probabilities[i]))

            prediction_sets.append(pred_set)

        return prediction_sets, probabilities

    def get_thresholds(self) -> dict[int, float]:
        """Get calibrated thresholds for each class"""
        if not self.is_calibrated:
            raise RuntimeError("Predictor not calibrated")
        return self.thresholds.copy()

    def evaluate_class_coverage(
        self, probabilities: NDArray, true_labels: NDArray
    ) -> dict[int, float]:
        """
        Evaluate coverage for each class independently.

        Args:
            probabilities: Test set probabilities
            true_labels: True labels

        Returns:
            Dict[int, float]: Coverage per class
        """
        prediction_sets, _ = self(probabilities)
        coverage = {}

        for class_idx in self.thresholds.keys():
            class_mask = true_labels == class_idx
            if not np.any(class_mask):
                continue

            class_indices = np.where(class_mask)[0]
            correct = sum(class_idx in prediction_sets[i] for i in class_indices)
            coverage[class_idx] = correct / len(class_indices)

        return coverage


def get_points_with_class(prediction_sets: list[set[int]], class_label: int) -> NDArray:
    """
    Get indices of points whose prediction sets contain the specified class.

    Args:
        prediction_sets: List of prediction sets
        class_label: Class label to search for

    Returns:
        NDArray: Array of indices where class_label is in prediction set
    """
    return np.array(
        [i for i, pred_set in enumerate(prediction_sets) if class_label in pred_set]
    )


def get_set_sizes(prediction_sets: list[set[int]]) -> NDArray:
    """
    Get size of prediction set for each point.

    Args:
        prediction_sets: List of prediction sets

    Returns:
        NDArray: Array of prediction set sizes
    """
    return np.array([len(pred_set) for pred_set in prediction_sets])

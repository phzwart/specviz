from typing import Optional, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy.typing import NDArray


class RFFClassifier(pl.LightningModule):
    """Random Fourier Features classifier with PyTorch Lightning"""

    def __init__(
        self,
        input_dim: int,
        n_classes: int,
        n_features: int = 1024,
        length_scale: float = 1.0,
        learning_rate: float = 1e-3,
    ):
        """
        Initialize RFF classifier.

        Args:
            input_dim: Dimension of input features
            n_classes: Number of output classes
            n_features: Number of random features
            length_scale: RBF kernel length scale
            learning_rate: Learning rate for optimization
        """
        super().__init__()
        self.save_hyperparameters()

        # Initialize random frequencies
        self.register_buffer(
            "frequencies", torch.randn(input_dim, n_features) / length_scale
        )

        # Initialize random offsets
        self.register_buffer("offsets", 2 * np.pi * torch.rand(n_features))

        # Linear layer for classification
        self.classifier = nn.Linear(2 * n_features, n_classes)

        # Store parameters
        self.input_dim = input_dim
        self.n_classes = n_classes
        self.n_features = n_features
        self.length_scale = length_scale
        self.learning_rate = learning_rate

    def rff_transform(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply random Fourier feature transform.

        Args:
            x: Input tensor (batch_size, input_dim)

        Returns:
            torch.Tensor: Transformed features (batch_size, 2*n_features)
        """
        # Project input
        projection = x @ self.frequencies

        # Add random offsets
        projection += self.offsets

        # Apply sine and cosine
        return torch.cat(
            [torch.sin(projection), torch.cos(projection)], dim=1
        ) / np.sqrt(self.n_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning log probabilities.

        Args:
            x: Input tensor (batch_size, input_dim)

        Returns:
            torch.Tensor: Log probabilities (batch_size, n_classes)
        """
        # Transform input using RFF
        features = self.rff_transform(x)

        # Get logits
        logits = self.classifier(features)

        # Return log probabilities
        return F.log_softmax(logits, dim=1)

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Training step with detailed logging"""
        x, y = batch
        log_probs = self(x)
        loss = F.nll_loss(log_probs, y)

        # Log batch metrics
        self.log(
            "train_loss",
            loss,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            batch_size=len(x),
        )

        return loss

    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        """Validation step with detailed logging"""
        x, y = batch
        log_probs = self(x)
        loss = F.nll_loss(log_probs, y)

        # Log metrics
        self.log(
            "val_loss",
            loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=len(x),
        )

        # Calculate and log accuracy
        preds = torch.argmax(log_probs, dim=1)
        acc = (preds == y).float().mean()
        self.log(
            "val_acc",
            acc,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=len(x),
        )

    def configure_optimizers(self):
        """Configure optimizer"""
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def predict_proba(self, X: NDArray) -> NDArray:
        """
        Predict class probabilities for numpy array.

        Args:
            X: Input features (n_samples, input_dim)

        Returns:
            NDArray: Class probabilities (n_samples, n_classes)
        """
        self.eval()
        with torch.no_grad():
            x = torch.FloatTensor(X).to(self.device)
            log_probs = self(x)
            return torch.exp(log_probs).cpu().numpy()

    def predict(self, X: NDArray) -> NDArray:
        """
        Predict class labels for numpy array.

        Args:
            X: Input features (n_samples, input_dim)

        Returns:
            NDArray: Predicted labels (n_samples,)
        """
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)


# Data module for training
class RFFDataModule(pl.LightningDataModule):
    """Data module for RFF classifier"""

    def __init__(
        self,
        X_train: NDArray,
        y_train: NDArray,
        X_val: Optional[NDArray] = None,
        y_val: Optional[NDArray] = None,
        batch_size: int = 32,
        num_workers: int = 0,
    ):
        """
        Initialize data module.

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            batch_size: Batch size
            num_workers: Number of worker threads
        """
        super().__init__()
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage: Optional[str] = None):
        """Setup data"""
        # Convert to tensors
        self.train_data = torch.utils.data.TensorDataset(
            torch.FloatTensor(self.X_train), torch.LongTensor(self.y_train)
        )

        if self.X_val is not None and self.y_val is not None:
            self.val_data = torch.utils.data.TensorDataset(
                torch.FloatTensor(self.X_val), torch.LongTensor(self.y_val)
            )

    def train_dataloader(self):
        """Get train dataloader"""
        return torch.utils.data.DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers > 0 else False,
            pin_memory=True if torch.cuda.is_available() else False,
        )

    def val_dataloader(self):
        """Get validation dataloader"""
        if hasattr(self, "val_data"):
            return torch.utils.data.DataLoader(
                self.val_data,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                persistent_workers=True if self.num_workers > 0 else False,
                pin_memory=True if torch.cuda.is_available() else False,
            )


class MetricsCallback(pl.Callback):
    """Custom callback for detailed metrics logging"""

    def __init__(self):
        super().__init__()
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []

    def _to_scalar(self, value):
        """Convert value to scalar, handling both tensors and numbers"""
        if torch.is_tensor(value):
            return value.item()
        return float(value)

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Print detailed metrics at end of training epoch"""
        train_loss = trainer.callback_metrics.get("train_loss_epoch", 0)
        val_loss = trainer.callback_metrics.get("val_loss", 0)
        val_acc = trainer.callback_metrics.get("val_acc", 0)

        # Store metrics
        self.train_losses.append(self._to_scalar(train_loss))
        self.val_losses.append(self._to_scalar(val_loss))
        self.val_accuracies.append(self._to_scalar(val_acc))

        # Print metrics
        print(f"\nEpoch {trainer.current_epoch}:")
        print(f"  Train Loss: {self.train_losses[-1]:.4f}")
        if val_loss:
            print(f"  Val Loss:   {self.val_losses[-1]:.4f}")
        if val_acc:
            print(f"  Val Acc:    {self.val_accuracies[-1]:.4f}")

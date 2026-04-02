from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def split_by_lesion_id(
    lesion_ids: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.1,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split tabular feature matrix by lesion_id while preserving lesion groups."""
    if lesion_ids.shape[0] != x.shape[0] or lesion_ids.shape[0] != y.shape[0]:
        raise ValueError("lesion_ids, x, y must have the same length")

    unique_lesions = np.unique(lesion_ids)
    train_lesions, test_lesions = train_test_split(
        unique_lesions,
        test_size=test_size,
        random_state=random_state,
        stratify=None,
    )

    train_mask = np.isin(lesion_ids, train_lesions)
    test_mask = np.isin(lesion_ids, test_lesions)

    x_train, y_train = x[train_mask], y[train_mask]
    x_test, y_test = x[test_mask], y[test_mask]

    return x_train, y_train, x_test, y_test, lesion_ids[train_mask], lesion_ids[test_mask]


@dataclass
class PreprocessingPipeline:
    """Standard preprocessing for tabular and HAM10000-like metadata."""

    config: Dict[str, Any]
    scaler: Optional[StandardScaler] = None

    def __post_init__(self):
        self.scaler = StandardScaler()

    def fit_transform(self, x_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
        """Fit preprocessing on training data and transform."""
        if x_train.ndim != 2:
            raise ValueError("x_train must be 2D for tabular preprocessing")

        return self.scaler.fit_transform(x_train)

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Transform validation/test data with already-fit preprocessing."""
        if self.scaler is None:
            raise RuntimeError("Pipeline has not been fit yet")
        if x.ndim != 2:
            raise ValueError("x must be 2D for tabular preprocessing")

        return self.scaler.transform(x)

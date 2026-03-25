from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from .contracts import HospitalDataBundle
from .hospital_env import HospitalSplits, VirtualHospital

SPLIT_NAMES = ("train", "val", "test")
CANCER_TYPES = ("BCC", "SCC", "MELANOMA", "AKIEC")


@dataclass(frozen=True)
class LocalHospitalData:
    """Hospital-local dataset container with standardized splits and cancer labels."""

    bundle: HospitalDataBundle
    test_ids: np.ndarray
    cancer_train: np.ndarray
    cancer_val: np.ndarray
    cancer_test: np.ndarray

    def one_vs_rest_labels(self, cancer_type: str, split: Literal["train", "val", "test"]) -> np.ndarray:
        normalized = _normalize_cancer_type(cancer_type)
        split_labels = self._split_cancer_labels(split)
        return (split_labels == normalized).astype(np.int64)

    def filter_for_cancer(
        self,
        cancer_type: str,
        split: Literal["train", "val", "test"],
        positive_only: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return split features with one-vs-rest labels or only positive-class rows."""
        normalized = _normalize_cancer_type(cancer_type)
        x_split, _ = self._split_xy(split)
        split_labels = self._split_cancer_labels(split)

        if positive_only:
            mask = split_labels == normalized
            return x_split[mask], np.ones(mask.sum(), dtype=np.int64)

        y_binary = (split_labels == normalized).astype(np.int64)
        return x_split, y_binary

    def _split_xy(self, split: Literal["train", "val", "test"]) -> tuple[np.ndarray, np.ndarray]:
        if split == "train":
            return self.bundle.x_train, self.bundle.y_train
        if split == "val":
            return self.bundle.x_val, self.bundle.y_val
        if split == "test":
            return self.bundle.x_test, self.bundle.y_test
        raise ValueError(f"Unsupported split: {split}. Supported: {', '.join(SPLIT_NAMES)}")

    def _split_cancer_labels(self, split: Literal["train", "val", "test"]) -> np.ndarray:
        if split == "train":
            return self.cancer_train
        if split == "val":
            return self.cancer_val
        if split == "test":
            return self.cancer_test
        raise ValueError(f"Unsupported split: {split}. Supported: {', '.join(SPLIT_NAMES)}")


class LocalDataPipeline:
    """Builds standardized local splits and exposes per-cancer filtering hooks."""

    def __init__(self, dataset_handler: VirtualHospital | None = None) -> None:
        self.dataset_handler = dataset_handler or VirtualHospital(random_state=42)

    def load(
        self,
        ham_metadata_csv: str | Path,
        isic_labels_csv: str | Path,
    ) -> LocalHospitalData:
        splits = self.dataset_handler.load(ham_metadata_csv=ham_metadata_csv, isic_labels_csv=isic_labels_csv)
        return self._to_local_data(splits)

    @staticmethod
    def _to_local_data(splits: HospitalSplits) -> LocalHospitalData:
        bundle = HospitalDataBundle(
            x_train=splits.x_train,
            y_train=splits.y_train,
            x_val=splits.x_val,
            y_val=splits.y_val,
            x_test=splits.x_test,
            y_test=splits.y_test,
        )
        return LocalHospitalData(
            bundle=bundle,
            test_ids=splits.test_ids,
            cancer_train=splits.cancer_train,
            cancer_val=splits.cancer_val,
            cancer_test=splits.cancer_test,
        )


def _normalize_cancer_type(cancer_type: str) -> str:
    normalized = cancer_type.strip().upper()
    if normalized not in CANCER_TYPES:
        raise ValueError(f"Unsupported cancer type: {cancer_type}. Supported: {', '.join(CANCER_TYPES)}")
    return normalized

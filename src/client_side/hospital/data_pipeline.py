from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from .contracts import HospitalDataBundle
from .config_helpers import get_cancer_types
from .hospital_env import HospitalSplits, VirtualHospital
from .augmentations import augment_dataset
from ..pre_processing.pipeline import PreprocessingPipeline

SPLIT_NAMES = ("train", "val", "test")


@dataclass(frozen=True)
class LocalHospitalData:
    """Hospital-local dataset container with standardized splits and cancer labels."""

    bundle: HospitalDataBundle
    test_ids: np.ndarray
    cancer_train: np.ndarray
    cancer_val: np.ndarray
    cancer_test: np.ndarray
    cancer_types: tuple[str, ...]

    def filter_for_cancer(
        self,
        cancer_type: str,
        split: Literal["train", "val", "test"],
        positive_only: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return split features with one-vs-rest labels or only positive-class rows."""
        normalized = _normalize_cancer_type(cancer_type, self.cancer_types)
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

    def __init__(self, dataset_handler: VirtualHospital | None = None, hospital_id: str = None, config: dict = None, hospital_ids: list = None) -> None:
        self.config = config
        self.hospital_id = hospital_id
        self.hospital_ids = hospital_ids
        self._cancer_types = get_cancer_types(config)
        # Always pass config to VirtualHospital
        self.dataset_handler = dataset_handler or VirtualHospital(config=config)

    def load(
        self,
        ham_metadata_csv: str | Path = None,
        isic_labels_csv: str | Path = None,
    ) -> LocalHospitalData:
        kwargs = {}
        if ham_metadata_csv is not None:
            kwargs["ham_metadata_csv"] = ham_metadata_csv
        if isic_labels_csv is not None:
            kwargs["isic_labels_csv"] = isic_labels_csv
        # Pass hospital_id and hospital_ids to activate per-hospital chunking
        kwargs["hospital_id"] = self.hospital_id
        kwargs["hospital_ids"] = self.hospital_ids
        splits = self.dataset_handler.load(**kwargs)

        if self.config is not None and self.config.get("augmentation", {}).get("enabled", False):
            mode = self.config.get("data_split", {}).get("mode", "tabular")
            if mode in ("tabular", "combined"):
                x_train_aug, y_train_aug = augment_dataset(splits.x_train, splits.y_train, self.config)
                splits = HospitalSplits(
                    x_train=x_train_aug,
                    y_train=y_train_aug,
                    x_val=splits.x_val,
                    y_val=splits.y_val,
                    x_test=splits.x_test,
                    y_test=splits.y_test,
                    test_ids=splits.test_ids,
                    cancer_train=splits.cancer_train,
                    cancer_val=splits.cancer_val,
                    cancer_test=splits.cancer_test,
                )

        if self.config is not None and self.config.get("preprocessing", {}).get("enabled", False):
            preprocess_cfg = self.config.get("preprocessing", {})
            pipeline = PreprocessingPipeline(preprocess_cfg)
            # Scale train/val/test sets with same parameters
            splits = HospitalSplits(
                x_train=pipeline.fit_transform(splits.x_train, splits.y_train),
                y_train=splits.y_train,
                x_val=pipeline.transform(splits.x_val),
                y_val=splits.y_val,
                x_test=pipeline.transform(splits.x_test),
                y_test=splits.y_test,
                test_ids=splits.test_ids,
                cancer_train=splits.cancer_train,
                cancer_val=splits.cancer_val,
                cancer_test=splits.cancer_test,
            )

        return self._to_local_data(splits)

    @staticmethod
    def _to_local_data(self, splits: HospitalSplits) -> LocalHospitalData:
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
            cancer_types=self._cancer_types,
        )


def _normalize_cancer_type(cancer_type: str, cancer_types: tuple[str, ...]) -> str:
    normalized = cancer_type.strip().upper()
    if normalized not in cancer_types:
        raise ValueError(f"Unsupported cancer type: {cancer_type}. Supported: {', '.join(cancer_types)}")
    return normalized

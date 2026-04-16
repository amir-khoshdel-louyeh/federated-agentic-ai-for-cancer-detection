from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

from ..pre_processing.normalization import (
    load_or_build_ham10000_metadata,
    normalize_isic2019_metadata,
)


@dataclass
class HospitalSplits:
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    test_ids: np.ndarray
    cancer_train: np.ndarray
    cancer_val: np.ndarray
    cancer_test: np.ndarray


class VirtualHospital:
    """Loads and prepares local hospital data from HAM10000 and ISIC 2019 metadata."""

    def __init__(self, random_state: int = None, config: dict = None) -> None:
        # Prefer config['sampling']['random_seed'] if available
        self.config = config
        if random_state is not None:
            self.random_state = random_state
        elif config is not None:
            self.random_state = config.get("sampling", {}).get("random_seed", 42)
        else:
            self.random_state = 42

    def _resolve_stratify_labels(
        self,
        y: np.ndarray,
        cancer_types: np.ndarray,
        stratify: bool,
    ) -> np.ndarray | None:
        """Prefer stratifying by cancer type, fall back to malignancy when needed."""
        if not stratify:
            return None

        cancer_labels = np.asarray(cancer_types, dtype=str)
        if cancer_labels.size > 0 and len(np.unique(cancer_labels)) > 1:
            return cancer_labels

        if y is not None and len(np.unique(y)) > 1:
            return y

        return None

    def _split_with_stratification(
        self,
        *arrays,
        test_size: float,
        stratify: np.ndarray | None,
        random_state: int,
    ):
        if stratify is None:
            return train_test_split(*arrays, test_size=test_size, random_state=random_state)

        try:
            return train_test_split(
                *arrays,
                test_size=test_size,
                stratify=stratify,
                random_state=random_state,
            )
        except ValueError:
            # Fall back to malignancy stratification or random split if cancer type stratification fails.
            fallback = self._resolve_stratify_labels(np.asarray(arrays[1], dtype=np.int64), np.asarray(arrays[3], dtype=str), False)
            if fallback is not None:
                return train_test_split(
                    *arrays,
                    test_size=test_size,
                    stratify=fallback,
                    random_state=random_state,
                )
            return train_test_split(*arrays, test_size=test_size, random_state=random_state)

    def _select_iid_total_samples_for_hospital(
        self,
        hospital_id: str,
        hospital_ids: list[str],
        total_samples: int,
        x: np.ndarray,
        y: np.ndarray,
        ids: np.ndarray,
    ) -> np.ndarray:
        """Select a stratified, deterministic subset for each hospital.

        This method chooses total_samples across the full dataset while preserving
        both malignant/benign proportions and assigning a fixed IID-like chunk to
        each hospital.
        """
        num_hospitals = len(hospital_ids)
        n_samples = total_samples // num_hospitals

        if total_samples > len(y):
            raise ValueError(f"Requested total_samples ({total_samples}) exceeds available samples ({len(y)}).")

        labels, counts = np.unique(y, return_counts=True)
        label_counts = {int(label): int(count) for label, count in zip(labels, counts)}
        malignant_count = label_counts.get(1, 0)
        benign_count = label_counts.get(0, 0)

        if malignant_count + benign_count != len(y):
            raise ValueError("Only binary target labels 0/1 are supported for total_samples selection.")

        malignant_selected = int(round(total_samples * malignant_count / len(y)))
        benign_selected = total_samples - malignant_selected

        if malignant_count > 0 and malignant_selected == 0:
            malignant_selected = 1
            benign_selected = total_samples - 1
        if benign_count > 0 and benign_selected == 0:
            benign_selected = 1
            malignant_selected = total_samples - 1

        if malignant_selected > malignant_count or benign_selected > benign_count:
            raise ValueError(
                "Not enough samples available to satisfy stratified total_samples selection."
            )

        sorted_ids = np.asarray(ids, dtype=str)
        order = np.argsort(sorted_ids, kind="stable")
        malignant_indices = order[y[order] == 1]
        benign_indices = order[y[order] == 0]

        # Choose a stratified subset of the requested size in a deterministic way.
        malignant_indices = malignant_indices[:malignant_selected]
        benign_indices = benign_indices[:benign_selected]

        # Compute per-hospital quotas based on the overall class ratio.
        class_ratio = malignant_selected / total_samples if total_samples > 0 else 0.0
        base_malignant = int(np.floor(n_samples * class_ratio))
        remainder = malignant_selected - base_malignant * num_hospitals
        malignant_quota = [base_malignant + (1 if i < remainder else 0) for i in range(num_hospitals)]
        selected_indices: list[int] = []
        malignant_cursor = 0
        benign_cursor = 0
        for i, hid in enumerate(hospital_ids):
            quota_malignant = malignant_quota[i]
            quota_benign = n_samples - quota_malignant
            if quota_benign < 0:
                raise ValueError("Invalid hospital quota computed for total_samples selection.")
            if malignant_cursor + quota_malignant > len(malignant_indices):
                raise ValueError("Not enough malignant samples for hospital selection.")
            if benign_cursor + quota_benign > len(benign_indices):
                raise ValueError("Not enough benign samples for hospital selection.")
            if hid == hospital_id:
                selected_indices = list(malignant_indices[malignant_cursor:malignant_cursor + quota_malignant])
                selected_indices += list(benign_indices[benign_cursor:benign_cursor + quota_benign])
                break
            malignant_cursor += quota_malignant
            benign_cursor += quota_benign

        return np.array(sorted(selected_indices), dtype=int)

    def load(
        self,
        ham_metadata_csv: str | Path = None,
        isic_labels_csv: str | Path = None,
        isic_metadata_csv: str | Path = None,
        hospital_id: str = None,
        hospital_ids: list = None,
    ) -> HospitalSplits:
        # Load data
        dfs = []
        if ham_metadata_csv is not None:
            dfs.append(self._load_ham10000(ham_metadata_csv))
        if isic_labels_csv is not None:
            dfs.append(self._load_isic2019(isic_labels_csv, isic_metadata_csv))
        if not dfs:
            raise ValueError("No dataset enabled: at least one of ham_metadata_csv or isic_labels_csv must be provided.")
        data = pd.concat(dfs, axis=0, ignore_index=True)

        x_data = data.drop(columns=["target", "image_id", "cancer_type"], errors="ignore")
        if "dx" in x_data.columns:
            x_data = x_data.drop(columns=["dx"])
        x = x_data.select_dtypes(include=[np.number]).to_numpy(dtype=np.float32)
        y = data["target"].to_numpy(dtype=np.int64)
        ids = data["image_id"].to_numpy()
        cancer_types = data["cancer_type"].to_numpy(dtype=str)

        # Shuffle whole combined dataset first (reproducible via random_state)
        rng = np.random.default_rng(self.random_state)
        perm = rng.permutation(len(x))
        x = x[perm]
        y = y[perm]
        ids = ids[perm]
        cancer_types = cancer_types[perm]

        # Subsample and split among hospitals if requested
        total_samples = self.config.get("sampling", {}).get("total_samples", None) if self.config else None
        hospital_ids = hospital_ids or (self.config.get("hospital_ids", []) if self.config else [])
        if isinstance(hospital_ids, str):
            hospital_ids = [h.strip() for h in hospital_ids.split(",")]
        hospital_id = hospital_id or (self.config.get("hospital_id") if self.config else None)
        if total_samples is not None and hospital_id and hospital_ids:
            num_hospitals = len(hospital_ids)
            if total_samples % num_hospitals != 0:
                raise ValueError(f"Total samples ({total_samples}) must be divisible by number of hospitals ({num_hospitals}).")
            if total_samples > len(x):
                raise ValueError(f"Requested total_samples ({total_samples}) exceeds available samples ({len(x)}).")

            selected = self._select_iid_total_samples_for_hospital(
                hospital_id=hospital_id,
                hospital_ids=hospital_ids,
                total_samples=total_samples,
                x=x,
                y=y,
                ids=ids,
            )
            x = x[selected]
            y = y[selected]
            ids = ids[selected]
            cancer_types = cancer_types[selected]

        # Get split ratios from config, fallback to defaults
        split_cfg = self.config.get("data_split", {}) if self.config else {}
        holdout_test = float(split_cfg.get("holdout_test", 0.1))
        k_folds = int(split_cfg.get("k_folds", 1))
        current_fold = int(split_cfg.get("current_fold", 0))
        stratify = bool(split_cfg.get("stratify", True))

        if k_folds > 1:
            # Create holdout test set first
            if holdout_test <= 0.0 or holdout_test >= 1.0:
                raise ValueError("data_split.holdout_test must be in (0,1)")

            stratify_labels = self._resolve_stratify_labels(y, cancer_types, stratify)
            x_rest, x_test, y_rest, y_test, ids_rest, ids_test, cancer_rest, cancer_test = self._split_with_stratification(
                x,
                y,
                ids,
                cancer_types,
                test_size=holdout_test,
                stratify=stratify_labels,
                random_state=self.random_state,
            )

            if current_fold < 0 or current_fold >= k_folds:
                raise ValueError(f"current_fold must be between 0 and k_folds-1 ({k_folds-1})")

            stratify_labels_rest = self._resolve_stratify_labels(y_rest, cancer_rest, stratify)
            if stratify_labels_rest is not None:
                try:
                    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=self.random_state)
                    fold_splits = list(skf.split(x_rest, stratify_labels_rest))
                except ValueError:
                    skf = KFold(n_splits=k_folds, shuffle=True, random_state=self.random_state)
                    fold_splits = list(skf.split(x_rest))
            else:
                skf = KFold(n_splits=k_folds, shuffle=True, random_state=self.random_state)
                fold_splits = list(skf.split(x_rest))

            train_idx, val_idx = fold_splits[current_fold]

            x_train, y_train, cancer_train = x_rest[train_idx], y_rest[train_idx], cancer_rest[train_idx]
            x_val, y_val, cancer_val = x_rest[val_idx], y_rest[val_idx], cancer_rest[val_idx]
            test_ids = ids_test

        else:
            test_ratio = float(split_cfg.get("holdout_test", 0.1))
            if test_ratio <= 0.0 or test_ratio >= 1.0:
                raise ValueError("data_split.holdout_test must be in (0,1)")

            # Default behavior for AI-agent workflows: reserve a test set and use the
            # remaining data as validation. The training split is retained only for
            # interface compatibility and is intentionally empty.
            stratify_labels = self._resolve_stratify_labels(y, cancer_types, stratify)
            x_val, x_test, y_val, y_test, ids_val, ids_test, cancer_val, cancer_test = self._split_with_stratification(
                x,
                y,
                ids,
                cancer_types,
                test_size=test_ratio,
                stratify=stratify_labels,
                random_state=self.random_state,
            )
            x_train = np.zeros((0, x.shape[1]), dtype=np.float32)
            y_train = np.zeros((0,), dtype=np.int64)
            cancer_train = np.array([], dtype=str)
            test_ids = ids_test

        return HospitalSplits(
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
            x_test=x_test,
            y_test=y_test,
            test_ids=test_ids,
            cancer_train=cancer_train,
            cancer_val=cancer_val,
            cancer_test=cancer_test,
        )

    def _load_ham10000(self, metadata_csv: str | Path) -> pd.DataFrame:
        return load_or_build_ham10000_metadata(metadata_csv, config=self.config)

    def _load_isic2019(self, labels_csv: str | Path, metadata_csv: str | Path = None) -> pd.DataFrame:
        return normalize_isic2019_metadata(labels_csv, metadata_csv=metadata_csv, config=self.config)

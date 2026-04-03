from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config_helpers import get_malignant_ham, get_malignant_isic


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


MALIGNANT_HAM = {"mel", "bcc", "akiec", "scc"}
MALIGNANT_ISIC = {"MEL", "BCC", "AK", "SCC"}
HAM_TO_CANCER_TYPE = {
    "bcc": "BCC",
    "scc": "SCC",
    "mel": "MELANOMA",
    "akiec": "AKIEC",
}



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

    def load(
        self,
        ham_metadata_csv: str | Path = None,
        isic_labels_csv: str | Path = None,
        hospital_id: str = None,
        hospital_ids: list = None,
    ) -> HospitalSplits:
        # Load data
        dfs = []
        if ham_metadata_csv is not None:
            dfs.append(self._load_ham10000(ham_metadata_csv))
        if isic_labels_csv is not None:
            dfs.append(self._load_isic2019(isic_labels_csv))
        if not dfs:
            raise ValueError("No dataset enabled: at least one of ham_metadata_csv or isic_labels_csv must be provided.")
        data = pd.concat(dfs, axis=0, ignore_index=True)

        x = data.drop(columns=["target", "image_id", "cancer_type"]).to_numpy(dtype=np.float32)
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
            n_samples = total_samples // num_hospitals
            idx = hospital_ids.index(hospital_id)
            rng = np.random.default_rng(self.random_state)
            all_indices = np.arange(len(x))
            rng.shuffle(all_indices)
            if total_samples > len(x):
                raise ValueError(f"Requested total_samples ({total_samples}) exceeds available samples ({len(x)}).")
            selected = all_indices[idx * n_samples : (idx + 1) * n_samples]
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

            if stratify and len(np.unique(y)) > 1:
                x_rest, x_test, y_rest, y_test, ids_rest, ids_test, cancer_rest, cancer_test = train_test_split(
                    x,
                    y,
                    ids,
                    cancer_types,
                    test_size=holdout_test,
                    stratify=y,
                    random_state=self.random_state,
                )
            else:
                n = len(x)
                idx = np.arange(n)
                rng = np.random.default_rng(self.random_state)
                rng.shuffle(idx)
                test_count = max(1, int(n * holdout_test))
                test_idx = idx[:test_count]
                rest_idx = idx[test_count:]
                x_rest, x_test = x[rest_idx], x[test_idx]
                y_rest, y_test = y[rest_idx], y[test_idx]
                ids_rest, ids_test = ids[rest_idx], ids[test_idx]
                cancer_rest, cancer_test = cancer_types[rest_idx], cancer_types[test_idx]

            from sklearn.model_selection import StratifiedKFold
            if current_fold < 0 or current_fold >= k_folds:
                raise ValueError(f"current_fold must be between 0 and k_folds-1 ({k_folds-1})")

            if stratify and len(np.unique(y_rest)) > 1:
                skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=self.random_state)
                fold_splits = list(skf.split(x_rest, y_rest))
            else:
                skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=self.random_state)
                fold_splits = list(skf.split(x_rest, y_rest))

            train_idx, val_idx = fold_splits[current_fold]

            x_train, y_train, cancer_train = x_rest[train_idx], y_rest[train_idx], cancer_rest[train_idx]
            x_val, y_val, cancer_val = x_rest[val_idx], y_rest[val_idx], cancer_rest[val_idx]
            test_ids = ids_test

        else:
            train_ratio = float(split_cfg.get("train", 0.8))
            val_ratio = float(split_cfg.get("validation", 0.1))
            test_ratio = float(split_cfg.get("test", 0.1))

            total = train_ratio + val_ratio + test_ratio
            train_ratio /= total
            val_ratio /= total
            test_ratio /= total

            # stratified split by binary target (malignancy) if possible
            if stratify and len(np.unique(y)) > 1:
                x_temp, x_test, y_temp, y_test, ids_temp, ids_test, cancer_temp, cancer_test = train_test_split(
                    x,
                    y,
                    ids,
                    cancer_types,
                    test_size=test_ratio,
                    stratify=y,
                    random_state=self.random_state,
                )
                # split remaining into train/val
                val_adj = val_ratio / (train_ratio + val_ratio)
                x_train, x_val, y_train, y_val, ids_train, ids_val, cancer_train, cancer_val = train_test_split(
                    x_temp,
                    y_temp,
                    ids_temp,
                    cancer_temp,
                    test_size=val_adj,
                    stratify=y_temp,
                    random_state=self.random_state,
                )
                test_ids = ids_test
            else:
                # Shuffle and split indices for this hospital's chunk
                n = len(x)
                rng = np.random.default_rng(self.random_state)
                indices = np.arange(n)
                rng.shuffle(indices)
                n_train = int(n * train_ratio)
                n_val = int(n * val_ratio)
                n_test = n - n_train - n_val
                train_idx = indices[:n_train]
                val_idx = indices[n_train:n_train+n_val]
                test_idx = indices[n_train+n_val:]

                x_train, y_train, cancer_train = x[train_idx], y[train_idx], cancer_types[train_idx]
                x_val, y_val, cancer_val = x[val_idx], y[val_idx], cancer_types[val_idx]
                x_test, y_test, cancer_test = x[test_idx], y[test_idx], cancer_types[test_idx]
                test_ids = ids[test_idx]

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
        df = pd.read_csv(metadata_csv)
        image_col = "image_id" if "image_id" in df.columns else df.columns[0]

        dx = df["dx"].astype(str).str.lower()
        malignant_ham = get_malignant_ham(self.config)
        target = dx.isin(malignant_ham).astype(int)
        cancer_type = dx.map(HAM_TO_CANCER_TYPE).fillna("OTHER")
        base = pd.DataFrame(
            {
                "image_id": df[image_col].astype(str),
                "target": target,
                "cancer_type": cancer_type,
            }
        )
        return self._build_features(base, df, age_candidates=("age",), site_candidates=("localization",))

    def _load_isic2019(self, labels_csv: str | Path) -> pd.DataFrame:
        df = pd.read_csv(labels_csv)
        image_col = "image" if "image" in df.columns else df.columns[0]

        malignant_isic = get_malignant_isic(self.config)
        present_labels = [label for label in malignant_isic if label in df.columns]
        if not present_labels:
            raise ValueError("ISIC 2019 labels CSV must contain at least one malignant class column.")

        target = (df[present_labels].sum(axis=1) > 0).astype(int)
        cancer_type = self._infer_isic_cancer_type(df)
        base = pd.DataFrame(
            {
                "image_id": df[image_col].astype(str),
                "target": target,
                "cancer_type": cancer_type,
            }
        )
        return self._build_features(base, df, age_candidates=("age_approx", "age"), site_candidates=("anatom_site_general",))

    @staticmethod
    def _infer_isic_cancer_type(df: pd.DataFrame) -> pd.Series:
        mapping = {
            "BCC": "BCC",
            "SCC": "SCC",
            "MEL": "MELANOMA",
            "AK": "AKIEC",
        }
        resolved = pd.Series(np.full(len(df), "OTHER", dtype=object), index=df.index)

        # Priority order keeps assignment deterministic if malformed rows have multiple positives.
        for label in ("BCC", "SCC", "MEL", "AK"):
            if label not in df.columns:
                continue
            mask = pd.to_numeric(df[label], errors="coerce").fillna(0.0) > 0
            resolved = resolved.where(~mask, mapping[label])

        return resolved.astype(str)

    def _build_features(
        self,
        base: pd.DataFrame,
        src: pd.DataFrame,
        age_candidates: Iterable[str],
        site_candidates: Iterable[str],
    ) -> pd.DataFrame:
        out = base.copy()

        # Rule-based agent requires these first five normalized features.
        out["asymmetry"] = self._deterministic_feature(out["image_id"], salt="asym")
        out["border_irregularity"] = self._deterministic_feature(out["image_id"], salt="border")
        out["color_variegation"] = self._deterministic_feature(out["image_id"], salt="color")

        diameter_col = next((c for c in ("diameter_mm", "lesion_diameter", "diameter") if c in src.columns), None)
        if diameter_col:
            diameter = pd.to_numeric(src[diameter_col], errors="coerce").fillna(src[diameter_col].median())
            out["diameter_mm_scaled"] = self._min_max_scale(diameter)
        else:
            out["diameter_mm_scaled"] = self._deterministic_feature(out["image_id"], salt="diam")

        age_col = next((c for c in age_candidates if c in src.columns), None)
        if age_col:
            age = pd.to_numeric(src[age_col], errors="coerce")
            age = age.fillna(age.median() if not np.isnan(age.median()) else 50)
            out["age_scaled"] = self._min_max_scale(age)
        else:
            out["age_scaled"] = self._deterministic_feature(out["image_id"], salt="age")

        sex_col = "sex" if "sex" in src.columns else ("gender" if "gender" in src.columns else None)
        if sex_col:
            male = src[sex_col].astype(str).str.lower().str.startswith("m").astype(float)
            out["is_male"] = male
        else:
            out["is_male"] = 0.5

        # Keep explicit sex feature for interpretability and model leverage.
        out["sex_encoded"] = out["is_male"]

        site_col = next((c for c in site_candidates if c in src.columns), None)
        if site_col:
            site_codes = src[site_col].astype("category").cat.codes
            site_codes = site_codes.replace(-1, site_codes[site_codes >= 0].median() if (site_codes >= 0).any() else 0)
            out["site_code"] = site_codes.astype(float)
            out["site_scaled"] = self._min_max_scale(site_codes)
        else:
            out["site_code"] = self._deterministic_feature(out["image_id"], salt="site_code")
            out["site_scaled"] = self._deterministic_feature(out["image_id"], salt="site")

        return out

    @staticmethod
    def _deterministic_feature(series: pd.Series, salt: str) -> pd.Series:
        hashed = series.apply(lambda v: hash(f"{salt}:{v}"))
        values = (hashed.abs() % 1000) / 1000.0
        return values.astype(float)

    @staticmethod
    def _min_max_scale(values: pd.Series) -> pd.Series:
        v_min = float(values.min())
        v_max = float(values.max())
        if np.isclose(v_min, v_max):
            return pd.Series(np.full(len(values), 0.5), index=values.index, dtype=float)
        return (values - v_min) / (v_max - v_min)

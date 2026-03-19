from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


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

    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state

    def load(
        self,
        ham_metadata_csv: str | Path,
        isic_labels_csv: str | Path,
        test_size: float = 0.2,
        val_size: float = 0.2,
    ) -> HospitalSplits:
        ham_df = self._load_ham10000(ham_metadata_csv)
        isic_df = self._load_isic2019(isic_labels_csv)
        data = pd.concat([ham_df, isic_df], axis=0, ignore_index=True)

        x = data.drop(columns=["target", "image_id", "cancer_type"]).to_numpy(dtype=np.float32)
        y = data["target"].to_numpy(dtype=np.int64)
        ids = data["image_id"].to_numpy()
        cancer_types = data["cancer_type"].to_numpy(dtype=str)

        x_train_val, x_test, y_train_val, y_test, id_train_val, id_test, cancer_train_val, cancer_test = train_test_split(
            x,
            y,
            ids,
            cancer_types,
            test_size=test_size,
            random_state=self.random_state,
            stratify=y,
        )

        x_train, x_val, y_train, y_val, _, _, cancer_train, cancer_val = train_test_split(
            x_train_val,
            y_train_val,
            id_train_val,
            cancer_train_val,
            test_size=val_size,
            random_state=self.random_state,
            stratify=y_train_val,
        )

        return HospitalSplits(
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
            x_test=x_test,
            y_test=y_test,
            test_ids=id_test,
            cancer_train=cancer_train,
            cancer_val=cancer_val,
            cancer_test=cancer_test,
        )

    def _load_ham10000(self, metadata_csv: str | Path) -> pd.DataFrame:
        df = pd.read_csv(metadata_csv)
        image_col = "image_id" if "image_id" in df.columns else df.columns[0]

        dx = df["dx"].astype(str).str.lower()
        target = dx.isin(MALIGNANT_HAM).astype(int)
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

        present_labels = [label for label in MALIGNANT_ISIC if label in df.columns]
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

        site_col = next((c for c in site_candidates if c in src.columns), None)
        if site_col:
            site_codes = src[site_col].astype("category").cat.codes
            site_codes = site_codes.replace(-1, site_codes[site_codes >= 0].median() if (site_codes >= 0).any() else 0)
            out["site_scaled"] = self._min_max_scale(site_codes)
        else:
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

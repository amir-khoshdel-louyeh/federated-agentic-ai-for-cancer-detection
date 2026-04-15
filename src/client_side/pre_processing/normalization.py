from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..hospital.config_helpers import get_malignant_ham, get_malignant_isic

HAM_TO_CANCER_TYPE = {
    "bcc": "BCC",
    "scc": "SCC",
    "mel": "MELANOMA",
    "akiec": "AKIEC",
}


def _factorize_sorted(values: pd.Series) -> np.ndarray:
    values = values.astype(str).fillna("missing")
    codes, _ = pd.factorize(values, sort=True)
    return codes.astype(np.int64)


def _safe_column_name(prefix: str, value: str) -> str:
    safe_value = str(value).strip().lower()
    safe_value = safe_value.replace(" ", "_").replace("/", "_").replace("-", "_").replace("\n", "_")
    safe_value = "".join(ch for ch in safe_value if ch.isalnum() or ch == "_")
    return f"{prefix}_{safe_value or 'missing'}"


def _one_hot_encode(series: pd.Series, prefix: str, max_categories: int = 30) -> dict[str, np.ndarray]:
    if series is None or series.empty:
        return {}

    categories = pd.Categorical(series.astype(str).fillna("missing"))
    unique_values = sorted(categories.categories)
    if len(unique_values) == 0 or len(unique_values) > max_categories:
        return {}

    encoded = {}
    for value in unique_values:
        column_name = _safe_column_name(prefix, value)
        encoded[column_name] = (categories == value).astype(float).to_numpy()
    return encoded


def _label_encode(series: pd.Series) -> np.ndarray:
    return _factorize_sorted(series)


def normalize_ham10000_metadata(
    metadata_csv: str | Path | pd.DataFrame,
    config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Normalize HAM10000 metadata into the project feature schema."""
    df = pd.read_csv(metadata_csv) if isinstance(metadata_csv, (str, Path)) else metadata_csv.copy()
    image_col = "image_id" if "image_id" in df.columns else df.columns[0]

    dx = df["dx"].astype(str).str.lower()
    malignant_ham = get_malignant_ham(config)
    dx_encoded = _label_encode(dx)
    target_mode = (config or {}).get("preprocessing", {}).get("target_encoding", "binary")
    target = dx_encoded if target_mode == "dx" else dx.isin(malignant_ham).astype(int)
    cancer_type = dx.map(HAM_TO_CANCER_TYPE).fillna("OTHER")

    base = pd.DataFrame(
        {
            "image_id": df[image_col].astype(str),
            "target": target,
            "cancer_type": cancer_type,
            "dx_encoded": dx_encoded,
        }
    )
    return _build_features(base, df, age_candidates=("age",), site_candidates=("localization",))


def load_or_build_ham10000_metadata(
    metadata_csv: str | Path,
    config: dict[str, Any] | None = None,
    normalized_filename: str = "HAM10000_metadata_normalized.csv",
) -> pd.DataFrame:
    """Load a normalized HAM10000 file if present, otherwise normalize and persist it."""
    path = Path(metadata_csv)
    if path.name == normalized_filename:
        return pd.read_csv(path)

    normalized_path = path.parent / normalized_filename
    if normalized_path.exists():
        return pd.read_csv(normalized_path)

    df = normalize_ham10000_metadata(path, config=config)
    df.to_csv(normalized_path, index=False)
    return df


def normalize_isic2019_metadata(
    labels_csv: str | Path | pd.DataFrame,
    metadata_csv: str | Path | pd.DataFrame | None = None,
    config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Normalize ISIC 2019 labels and metadata into the project feature schema."""
    labels_df = pd.read_csv(labels_csv) if isinstance(labels_csv, (str, Path)) else labels_csv.copy()
    image_col = "image" if "image" in labels_df.columns else labels_df.columns[0]

    if metadata_csv is not None:
        metadata_df = pd.read_csv(metadata_csv) if isinstance(metadata_csv, (str, Path)) else metadata_csv.copy()
        metadata_image_col = "image" if "image" in metadata_df.columns else metadata_df.columns[0]
        metadata_df = metadata_df.rename(columns={metadata_image_col: image_col})
        if image_col not in metadata_df.columns:
            raise ValueError(
                f"ISIC metadata CSV must contain an image id column; got {metadata_df.columns.tolist()}"
            )
        labels_df = labels_df.merge(metadata_df, on=image_col, how="left")

    malignant_isic = get_malignant_isic(config)
    present_labels = [label for label in malignant_isic if label in labels_df.columns]
    if not present_labels:
        raise ValueError("ISIC 2019 labels CSV must contain at least one malignant class column.")

    cancer_type = _infer_isic_cancer_type(labels_df)
    cancer_type_encoded = _label_encode(cancer_type)
    target_mode = (config or {}).get("preprocessing", {}).get("target_encoding", "binary")
    target = cancer_type_encoded if target_mode == "dx" else (labels_df[present_labels].sum(axis=1) > 0).astype(int)

    base = pd.DataFrame(
        {
            "image_id": labels_df[image_col].astype(str),
            "target": target,
            "cancer_type": cancer_type,
            "cancer_type_encoded": cancer_type_encoded,
        }
    )
    return _build_features(
        base,
        labels_df,
        age_candidates=("age_approx", "age"),
        site_candidates=("anatom_site_general",),
    )


def _infer_isic_cancer_type(df: pd.DataFrame) -> pd.Series:
    mapping = {
        "BCC": "BCC",
        "SCC": "SCC",
        "MEL": "MELANOMA",
        "AK": "AKIEC",
    }
    resolved = pd.Series(np.full(len(df), "OTHER", dtype=object), index=df.index)
    for label in ("BCC", "SCC", "MEL", "AK"):
        if label not in df.columns:
            continue
        mask = pd.to_numeric(df[label], errors="coerce").fillna(0.0) > 0
        resolved = resolved.where(~mask, mapping[label])
    return resolved.astype(str)


def _build_features(
    base: pd.DataFrame,
    src: pd.DataFrame,
    age_candidates: Iterable[str],
    site_candidates: Iterable[str],
) -> pd.DataFrame:
    """Create normalized tabular clinical metadata features for AI agent input."""
    out = base.copy()

    out["asymmetry"] = _deterministic_feature(out["image_id"], salt="asym")
    out["border_irregularity"] = _deterministic_feature(out["image_id"], salt="border")
    out["color_variegation"] = _deterministic_feature(out["image_id"], salt="color")

    diameter_col = next((c for c in ("diameter_mm", "lesion_diameter", "diameter") if c in src.columns), None)
    if diameter_col:
        diameter = pd.to_numeric(src[diameter_col], errors="coerce").fillna(src[diameter_col].median())
        out["diameter_mm_scaled"] = _min_max_scale(diameter)
    else:
        out["diameter_mm_scaled"] = _deterministic_feature(out["image_id"], salt="diam")

    age_col = next((c for c in age_candidates if c in src.columns), None)
    if age_col:
        age = pd.to_numeric(src[age_col], errors="coerce")
        age = age.fillna(age.median() if not np.isnan(age.median()) else 50)
        out["age_scaled"] = _min_max_scale(age)
    else:
        out["age_scaled"] = _deterministic_feature(out["image_id"], salt="age")

    sex_col = "sex" if "sex" in src.columns else ("gender" if "gender" in src.columns else None)
    if sex_col:
        sex_series = src[sex_col].astype(str).fillna("missing")
        male = sex_series.str.lower().str.startswith("m").astype(float)
        out["is_male"] = male
        out["sex_encoded"] = out["is_male"]
        sex_one_hot = _one_hot_encode(sex_series, "sex")
        if sex_one_hot:
            out = pd.concat([out, pd.DataFrame(sex_one_hot, index=src.index)], axis=1)
    else:
        out["is_male"] = 0.5
        out["sex_encoded"] = out["is_male"]

    dx_type_col = "dx_type" if "dx_type" in src.columns else None
    if dx_type_col:
        out["dx_type_encoded"] = _label_encode(src[dx_type_col].astype(str).fillna("missing"))

    site_col = next((c for c in site_candidates if c in src.columns), None)
    if site_col:
        site_series = src[site_col].astype(str).fillna("missing")
        site_codes = _label_encode(site_series)
        out["site_code"] = site_codes.astype(float)
        out["site_scaled"] = _min_max_scale(site_codes)
        site_one_hot = _one_hot_encode(site_series, "localization")
        if site_one_hot:
            out = pd.concat([out, pd.DataFrame(site_one_hot, index=src.index)], axis=1)
    else:
        out["site_code"] = _deterministic_feature(out["image_id"], salt="site_code")
        out["site_scaled"] = _deterministic_feature(out["image_id"], salt="site")

    return out


def _deterministic_feature(series: pd.Series, salt: str) -> pd.Series:
    hashed = series.apply(lambda v: hash(f"{salt}:{v}"))
    values = (hashed.abs() % 1000) / 1000.0
    return values.astype(float)


def _min_max_scale(values: pd.Series) -> pd.Series:
    v_min = float(values.min())
    v_max = float(values.max())
    if np.isclose(v_min, v_max):
        return pd.Series(np.full(len(values), 0.5), index=values.index, dtype=float)
    return (values - v_min) / (v_max - v_min)

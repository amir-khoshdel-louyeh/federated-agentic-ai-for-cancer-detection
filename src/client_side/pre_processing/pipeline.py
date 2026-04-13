from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
from PIL import Image
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
    """Standard preprocessing for tabular data and simple RGB image pipelines."""

    config: Dict[str, Any]
    scaler: Optional[StandardScaler] = None

    IMAGE_NET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    IMAGE_NET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __post_init__(self):
        self.scaler = StandardScaler()

    def fit_transform(self, x_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
        """Fit preprocessing on training data and transform."""
        if self._is_image_mode(x_train):
            return self._preprocess_images(x_train)

        if x_train.ndim != 2:
            raise ValueError("x_train must be 2D for tabular preprocessing")

        return self.scaler.fit_transform(x_train)

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Transform validation/test data with already-fit preprocessing."""
        if self._is_image_mode(x):
            return self._preprocess_images(x)

        if self.scaler is None:
            raise RuntimeError("Pipeline has not been fit yet")
        if x.ndim != 2:
            raise ValueError("x must be 2D for tabular preprocessing")

        return self.scaler.transform(x)

    def _is_image_mode(self, x: np.ndarray) -> bool:
        mode = str(self.config.get("mode", "")).strip().lower()
        if mode == "image":
            return True
        return x.ndim == 4

    def _preprocess_images(self, x: np.ndarray) -> np.ndarray:
        if x.ndim != 4:
            raise ValueError("Image preprocessing requires a 4D image batch of shape (N,H,W,C) or (N,C,H,W)")

        resized = self._resize_images(x)
        return self._normalize_images(resized)

    def _resize_images(self, x: np.ndarray) -> np.ndarray:
        image_config = self.config.get("image", {}) or {}
        resize_dims = tuple(int(v) for v in image_config.get("resize", (224, 224)))
        if len(resize_dims) != 2:
            raise ValueError("preprocessing.image.resize must be a list or tuple of two integers")

        batch: list[np.ndarray] = []
        channel_first = self._is_channel_first(x)

        for image in x:
            image_hwc = self._to_hwc_uint8(image, channel_first)
            resized_image = Image.fromarray(image_hwc).resize((resize_dims[1], resize_dims[0]), Image.BILINEAR)
            resized_array = np.asarray(resized_image)
            if channel_first:
                resized_array = np.transpose(resized_array, (2, 0, 1))
            batch.append(resized_array.astype(np.float32))

        return np.stack(batch, axis=0)

    def _normalize_images(self, x: np.ndarray) -> np.ndarray:
        normalized = x.astype(np.float32)
        if normalized.max() > 1.5:
            normalized = np.clip(normalized, 0.0, 255.0) / 255.0
        else:
            normalized = np.clip(normalized, 0.0, 1.0)

        normalize_mode = str((self.config.get("image", {}) or {}).get("normalize", "minmax")).strip().lower()
        if normalize_mode in ("imagenet", "imagenet_mean_std", "imagenet-meanstd"):
            channel_last = normalized.ndim == 4 and normalized.shape[-1] == 3
            channel_first = normalized.ndim == 4 and normalized.shape[1] == 3
            if not (channel_last or channel_first):
                raise ValueError("ImageNet normalization requires 3-channel RGB images in HWC or NCHW format")

            if channel_last:
                mean = self.IMAGE_NET_MEAN.reshape(1, 1, 1, 3)
                std = self.IMAGE_NET_STD.reshape(1, 1, 1, 3)
            else:
                mean = self.IMAGE_NET_MEAN.reshape(1, 3, 1, 1)
                std = self.IMAGE_NET_STD.reshape(1, 3, 1, 1)

            normalized = (normalized - mean) / std
        return normalized

    def _is_channel_first(self, x: np.ndarray) -> bool:
        return x.ndim == 4 and x.shape[1] in (1, 3) and x.shape[-1] not in (1, 3)

    def _to_hwc_uint8(self, image: np.ndarray, channel_first: bool) -> np.ndarray:
        if channel_first:
            image = np.transpose(image, (1, 2, 0))

        if image.ndim == 2:
            image = image[..., np.newaxis]

        image = image.astype(np.float32)
        if image.max() <= 1.0 and image.min() >= 0.0:
            image = image * 255.0

        image = np.clip(image, 0.0, 255.0).round().astype(np.uint8)
        if image.shape[-1] == 1:
            image = np.repeat(image, 3, axis=-1)
        return image

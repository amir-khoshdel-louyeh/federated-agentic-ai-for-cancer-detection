from __future__ import annotations

import numpy as np


def apply_tabular_augmentation(x: np.ndarray, config: dict[str, any]) -> np.ndarray:
    """Apply lightweight tabular-style augmentation to numeric features.
    Works with existing feature vectors when image pixels are not available.
    """
    if not config.get("augmentation", {}).get("enabled", False):
        return x

    n_samples, n_features = x.shape
    out = x.copy().astype(np.float32)

    # Rotation/flipping semantics in synthetic tabular space:
    # apply feature permutation to represent orientation variants.
    rotation_prob = float(config.get("augmentation", {}).get("rotation_prob", 0.0))
    flip_prob = float(config.get("augmentation", {}).get("flip_prob", 0.0))
    color_jitter_strength = float(config.get("augmentation", {}).get("color_jitter_strength", 0.0))
    hair_removal_strength = float(config.get("augmentation", {}).get("hair_removal_strength", 0.0))

    # add noise to all features as “color jitter” analog
    if color_jitter_strength > 0:
        noise = np.random.default_rng().normal(0, color_jitter_strength, size=out.shape)
        out = out + noise

    # hair removal simulation: random feature transposition on subset
    if hair_removal_strength > 0:
        removal_mask = np.random.default_rng().random(n_samples) < hair_removal_strength
        if removal_mask.any():
            out[removal_mask] *= np.random.default_rng().uniform(0.8, 0.95, size=(removal_mask.sum(), 1))

    # rotation/flipping as data-specific transforms
    if rotation_prob > 0:
        apply = np.random.default_rng().random(n_samples) < rotation_prob
        if apply.any() and n_features > 1:
            perm = np.random.default_rng().permutation(n_features)
            out[apply] = out[apply][:, perm]

    if flip_prob > 0:
        apply = np.random.default_rng().random(n_samples) < flip_prob
        if apply.any() and n_features > 1:
            out[apply] = np.flip(out[apply], axis=1)

    # Clip to [0,1] if data is normalized, else keep value range
    out = np.clip(out, 0.0, 1.0)
    return out


def augment_dataset(x_train: np.ndarray, y_train: np.ndarray, config: dict[str, any]) -> tuple[np.ndarray, np.ndarray]:
    """Return augmented training data (appended to existing samples)."""
    if not config.get("augmentation", {}).get("enabled", False):
        return x_train, y_train

    n_add = int(config.get("augmentation", {}).get("num_augmented_copies", 0))
    if n_add <= 0:
        return x_train, y_train

    out_x = [x_train]
    out_y = [y_train]

    for _ in range(n_add):
        x_aug = apply_tabular_augmentation(x_train, config)
        out_x.append(x_aug)
        out_y.append(y_train.copy())

    x_augmented = np.vstack(out_x)
    y_augmented = np.concatenate(out_y)
    return x_augmented, y_augmented

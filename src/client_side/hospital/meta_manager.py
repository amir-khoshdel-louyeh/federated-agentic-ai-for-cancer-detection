from __future__ import annotations

import numpy as np


class MetaManager:
    """Combine specialist agent outputs with confidence + uncertainty weighting."""

    def __init__(self, soft_vote_temperature: float = 1.0):
        self.soft_vote_temperature = soft_vote_temperature

    def combine(self, predictions: dict[str, np.ndarray], uncertainties: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Return aggregated probability and consensus score."""
        # predictions is map cancer_type->probs (shape N)
        # uncertainties is map cancer_type->uncertainty (shape N)

        schema = {}
        for cancer_type, proba in predictions.items():
            uncert = uncertainties.get(cancer_type, np.zeros_like(proba))
            confidence = 1.0 - uncert
            # weight by confidence (and tiny epsilon to avoid zero)
            weight = (confidence + 1e-6) ** (1.0 / max(1.0, self.soft_vote_temperature))
            schema[cancer_type] = {
                "prob": proba,
                "uncertainty": uncert,
                "weight": weight,
            }

        # global consensus by weighted mean across cancer types per sample
        cancer_types = list(schema.keys())
        stacked_proba = np.stack([schema[c]["prob"] for c in cancer_types], axis=1)
        stacked_weights = np.stack([schema[c]["weight"] for c in cancer_types], axis=1)
        total_weight = np.sum(stacked_weights, axis=1, keepdims=True)
        total_weight = np.maximum(total_weight, 1e-6)

        weighted_proba = np.sum(stacked_proba * stacked_weights, axis=1, keepdims=True) / total_weight
        consensus = weighted_proba.reshape(-1)

        return {
            "cancer_types": cancer_types,
            "meta_proba": consensus,
            "details": schema,
        }

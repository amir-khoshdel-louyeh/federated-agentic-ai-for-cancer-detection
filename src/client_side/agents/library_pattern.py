from __future__ import annotations

import pickle
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

from .base import ThinkingPattern


class PretrainedLibraryThinkingPattern(ThinkingPattern):
    """High-capacity library-powered thinking pattern (gradient boosting).

    This is designed as the "powerful AI agent" mode requested in the new proposal:
    no hand-crafted logistic rules, modern ensemble learner from sklearn.
    """

    def __init__(
        self,
        max_iter: int = 200,
        learning_rate: float = 0.1,
        max_depth: int = 6,
        l2_regularization: float = 0.0,
        class_weight: str | dict | None = "balanced",
        random_state: int = 42,
    ) -> None:
        self.scaler = StandardScaler()
        self.model = HistGradientBoostingClassifier(
            max_iter=max_iter,
            learning_rate=learning_rate,
            max_leaf_nodes=2 ** max_depth,
            l2_regularization=l2_regularization,
            class_weight=class_weight,
            random_state=random_state,
        )
        self._is_fitted = False

    @classmethod
    def from_config(cls, config: dict | None = None) -> "PretrainedLibraryThinkingPattern":
        if not config:
            return cls()

        params = config.get("agents", {}).get("patterns", {}).get("pretrained_library", {})
        if not isinstance(params, dict):
            params = {}

        l2_reg = float(params.get("l2_regularization", 1.0))

        return cls(
            max_iter=int(params.get("max_iter", 200)),
            learning_rate=float(params.get("learning_rate", 0.1)),
            max_depth=int(params.get("max_depth", 6)),
            l2_regularization=l2_reg,
            class_weight=params.get("class_weight", "balanced"),
            random_state=int(params.get("random_state", 42)),
        )

    @property
    def name(self) -> str:
        return "pretrained_library"

    def fit(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
        x_scaled = self.scaler.fit_transform(x_train)
        self.model.fit(x_scaled, y_train)
        self._is_fitted = True

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("PretrainedLibraryThinkingPattern must be fitted before predict_proba.")

        x_scaled = self.scaler.transform(x)
        probs = self.model.predict_proba(x_scaled)[:, 1]
        return np.clip(probs, 0.0, 1.0)

    def predict_uncertainty(self, x: np.ndarray, n_samples: int = 25) -> np.ndarray:
        # For tree ensembles we approximate uncertainty by dropout-like random subset predictions.
        if not self._is_fitted:
            raise RuntimeError("PretrainedLibraryThinkingPattern must be fitted before predict_uncertainty.")

        x_scaled = self.scaler.transform(x)
        # as HistGradientBoosting doesn't support MC directly, use model's staged_predict_proba as proxy
        if hasattr(self.model, "staged_predict_proba"):
            p_list = []
            # use a small number of intermediate stages to estimate variance
            for p in self.model.staged_predict_proba(x_scaled):
                p_list.append(p[:, 1])
                if len(p_list) >= min(n_samples, 20):
                    break
            preds = np.stack(p_list, axis=0)
            uncertainty = np.std(preds, axis=0)
            return np.clip(uncertainty, 0.0, 1.0)
        # fallback: deterministic 0 uncertainty
        return np.zeros(x.shape[0], dtype=np.float32)

    def save_model(self, file_path: str) -> None:
        with open(file_path + ".pkl", "wb") as f:
            pickle.dump({"model": self.model, "scaler": self.scaler, "fitted": self._is_fitted}, f)

    def load_model(self, file_path: str) -> None:
        with open(file_path + ".pkl", "rb") as f:
            state = pickle.load(f)
        self.model = state["model"]
        self.scaler = state["scaler"]
        self._is_fitted = state.get("fitted", False)

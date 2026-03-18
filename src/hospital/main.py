from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from src.agents import BayesianAgent, DeepLearningAgent, RuleBasedAgent
from .hospital_env import VirtualHospital
from .meta_controller import LocalMetaController


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Single-hospital multi-agent training pipeline")
    parser.add_argument("--ham-csv", required=True, help="Path to HAM10000 metadata CSV")
    parser.add_argument("--isic-csv", required=True, help="Path to ISIC 2019 labels CSV")
    parser.add_argument("--out-dir", default="outputs", help="Output directory")
    return parser.parse_args()


def eval_probs(y_true: np.ndarray, probs: np.ndarray) -> dict[str, float]:
    preds = (probs >= 0.5).astype(int)
    metrics = {
        "accuracy": float(accuracy_score(y_true, preds)),
        "f1": float(f1_score(y_true, preds, zero_division=0)),
    }
    try:
        metrics["auc"] = float(roc_auc_score(y_true, probs))
    except ValueError:
        metrics["auc"] = 0.5
    return metrics


def main() -> None:
    args = parse_args()

    hospital = VirtualHospital(random_state=42)
    splits = hospital.load(args.ham_csv, args.isic_csv)

    agents = [
        RuleBasedAgent(),
        BayesianAgent(),
        DeepLearningAgent(epochs=20, batch_size=64, lr=1e-3),
    ]

    val_predictions: dict[str, np.ndarray] = {}
    test_predictions: dict[str, np.ndarray] = {}
    report: dict[str, dict[str, float]] = {}

    for agent in agents:
        agent.fit(splits.x_train, splits.y_train)
        val_probs = agent.predict_proba(splits.x_val)
        test_probs = agent.predict_proba(splits.x_test)

        val_predictions[agent.name] = val_probs
        test_predictions[agent.name] = test_probs
        report[agent.name] = eval_probs(splits.y_test, test_probs)

    controller = LocalMetaController()
    scores = controller.fit_weights(splits.y_val, val_predictions)
    ensemble_probs = controller.ensemble_predict(test_predictions)
    report["ensemble"] = eval_probs(splits.y_test, ensemble_probs)

    report["weights"] = {s.name: float(s.weight) for s in scores}

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    pred_df = pd.DataFrame(
        {
            "image_id": splits.test_ids,
            "y_true": splits.y_test,
            "ensemble_prob": ensemble_probs,
            **{f"prob_{k}": v for k, v in test_predictions.items()},
        }
    )
    pred_df.to_csv(out_dir / "predictions.csv", index=False)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

import logging
from typing import Dict, Any
from configs.config_loader import load_config
from src.simulator.controller import (
    initialize_system,
    train_system,
    validation_system,
    test_system,
    show_results,
    show_log_location,
)


def run_k_fold_experiment(config: Dict[str, Any]) -> Dict[str, Any]:
    """Run k-fold cross-validation with holdout test split.

    This function loops over folds by setting config['data_split']['current_fold'],
    reinitializes hospitals, and runs training+validation+testing for each fold.
    """
    data_split = config.get("data_split", {})
    k_folds = int(data_split.get("k_folds", 1))
    if k_folds <= 1:
        raise ValueError("k_folds must be greater than 1 to use run_k_fold_experiment")

    aggregated_results = {}
    for fold in range(k_folds):
        logging.info(f"=== K-Fold {fold + 1}/{k_folds} ===")
        print(f"=== K-Fold {fold + 1}/{k_folds} ===")
        config["data_split"]["current_fold"] = fold

        hospitals = initialize_system(config)

        logging.info("Starting federated training...")
        print("Starting federated training...")
        train_system(config, hospitals)
        logging.info("Training complete.")
        print("Training complete.")

        validation_system(hospitals)

        logging.info("Running evaluation on test data...")
        print("Running evaluation on test data...")
        results = test_system(hospitals)
        show_results(results)
        show_log_location(config)
        print(show_log_location(config))

        aggregated_results[f"fold_{fold}"] = results

    return aggregated_results

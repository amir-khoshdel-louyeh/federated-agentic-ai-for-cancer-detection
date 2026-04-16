import logging
from typing import Dict, Any
from configs.config_loader import load_config
from src.simulator.controller import (
    initialize_system,
    federated_evaluation_round,
    validation_system,
    test_system,
    test_system_on_external_data,
    show_results,
    show_log_location,
)


def _collect_external_holdout(hospitals, n_samples, random_seed):
    """Gather final holdout test holders from all hospitals and sample n_samples."""
    import numpy as np

    x_candidates = []
    cancer_candidates = []
    for hospital in hospitals.values():
        if hospital.local_data is None:
            continue
        x_candidates.append(hospital.local_data.bundle.x_test)
        cancer_candidates.append(hospital.local_data.cancer_test)

    if not x_candidates:
        raise RuntimeError("No hospital test data available to sample external holdout data.")

    x_all = np.vstack(x_candidates)
    cancer_all = np.concatenate(cancer_candidates)

    n_samples = min(n_samples, x_all.shape[0])
    rng = np.random.default_rng(random_seed)
    selected = rng.choice(x_all.shape[0], size=n_samples, replace=False)

    return x_all[selected], cancer_all[selected]


def run_k_fold_experiment(config: Dict[str, Any]) -> Dict[str, Any]:
    """Run k-fold cross-validation with holdout test split.

    This function loops over folds by setting config['data_split']['current_fold'],
    reinitializes hospitals, and runs federated evaluation rounds with validation
    and test evaluation for each fold.
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

        logging.info("Starting federated evaluation round...")
        print("Starting federated evaluation round...")
        federated_evaluation_round(config, hospitals)
        logging.info("Federated evaluation round complete.")
        print("Federated evaluation round complete.")

        validation_system(hospitals)

        logging.info("Running evaluation on test data...")
        print("Running evaluation on test data...")
        results = test_system(hospitals)
        show_results(results)
        show_log_location(config)
        print(show_log_location(config))

        aggregated_results[f"fold_{fold}"] = results

    # Optional final external holdout test after all folds are complete
    final_test_cfg = config.get("data_split", {}).get("final_test", {})
    if final_test_cfg.get("enabled", False):
        n_samples = int(final_test_cfg.get("n_samples", 100))
        random_seed = int(final_test_cfg.get("random_seed", config.get("sampling", {}).get("random_seed", 42)))

        x_holdout, cancer_holdout = _collect_external_holdout(hospitals, n_samples, random_seed)
        logging.info(f"Running final external holdout test with {x_holdout.shape[0]} samples")
        print(f"Running final external holdout test with {x_holdout.shape[0]} samples")

        final_results = test_system_on_external_data(hospitals, x_holdout, cancer_holdout)
        show_results(final_results)
        show_log_location(config)

        aggregated_results["final_external_test"] = final_results

    return aggregated_results

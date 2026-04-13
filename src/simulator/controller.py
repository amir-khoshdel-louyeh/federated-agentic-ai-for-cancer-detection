from configs.config_loader import load_config
from pathlib import Path
import logging
from src.client_side.hospital.agent_portfolio import AgentPortfolio
from src.client_side.hospital.hospital_node import HospitalNode
from src.client_side.hospital.data_pipeline import LocalDataPipeline
from src.client_side.hospital.orchestrator import FederatedRoundOrchestrator
from src.client_side.hospital.pattern_factory import create_thinking_pattern




def configure_logging(config):
    """Configure global logging so everything routes to file, warnings show on console."""
    log_dir = Path(config.get("tracking", {}).get("log_dir", "outputs/logs"))
    log_file = config.get("tracking", {}).get("log_file_name", "simulation.log")
    log_path = log_dir / log_file

    out_dir = Path(config.get("out_dir", "outputs"))
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    root_logger = logging.getLogger()
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)

    root_logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    logging.captureWarnings(True)
    logging.info(f"Logging initialized: {log_path}")


def make_hospitals(config, data_pipeline):
    """Instantiate hospitals and their agent portfolios."""
    hospital_ids = [h.strip() for h in config["hospital_ids"].split(",")]
    agent_patterns = config["agents"]["patterns"]["default_mapping"]
    enabled_datasets = config.get("enabled_datasets", [])
    num_epoch = int(config.get("simulation", {}).get("num_epoch", 1))
    hospitals = {}
    for hid in hospital_ids:
        patterns = {
            ct: create_thinking_pattern(agent_patterns[ct])
            for ct in agent_patterns
        }
        portfolio = AgentPortfolio(initial_patterns=patterns)
        # Each hospital gets its own pipeline with correct ids
        pipeline = LocalDataPipeline(
            hospital_id=hid,
            hospital_ids=hospital_ids,
            config=config
        )
        hospital_kwargs = dict(
            hospital_id=hid,
            data_pipeline=pipeline,
            agent_portfolio=portfolio,
            config=config
        )
        if "HAM10000" in enabled_datasets:
            hospital_kwargs["ham_metadata_csv"] = config["ham_csv"]
        if "ISIC2019" in enabled_datasets:
            hospital_kwargs["isic_labels_csv"] = config["isic_csv"]
            hospital_kwargs["isic_metadata_csv"] = config.get("isic_metadata_csv")
        hospitals[hid] = HospitalNode(**hospital_kwargs)
    return hospitals

def initialize_system(config):
    """Initialize config, output dirs, logging, data pipeline, and hospitals."""
    ensure_output_dirs(config)
    configure_logging(config)

    data_pipeline = LocalDataPipeline()
    hospitals = make_hospitals(config, data_pipeline)
    allocate_data(hospitals)
    logging.info("System initialized.")
    print("System initialized.")
    return hospitals


def _compute_validation_f1_score(validation_reports):
    """Return average f1 over all hospitals and cancer types in validation reports."""
    scores = []
    for hid, report in validation_reports.items():
        for cancer_type, cancer_data in report.items():
            f1 = cancer_data.get("metrics", {}).get("f1")
            if f1 is not None:
                scores.append(float(f1))
    if not scores:
        return 0.0
    return float(sum(scores) / len(scores))


def _save_epoch_checkpoint(hospitals, checkpoint_dir):
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    for hid, hospital in hospitals.items():
        if hasattr(hospital, "scope") and hasattr(hospital.scope, "agent_portfolio"):
            portfolio = hospital.scope.agent_portfolio
            if hasattr(portfolio, "save_all_models"):
                portfolio.save_all_models(str(checkpoint_dir), hid)


def _restore_epoch_checkpoint(hospitals, checkpoint_dir):
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")
    for hid, hospital in hospitals.items():
        if hasattr(hospital, "scope") and hasattr(hospital.scope, "agent_portfolio"):
            portfolio = hospital.scope.agent_portfolio
            if hasattr(portfolio, "load_all_models"):
                portfolio.load_all_models(str(checkpoint_dir), hid)


def train_system(config, hospitals, save_history=False, save_models=False):
    """Run federated training rounds.

    Args:
        config (dict): configuration object.
        hospitals (dict): hospital_id -> HospitalNode.
        save_history (bool): if True, writes per-round metrics and decision files.
        save_models (bool): if True, saves local models at end.

    Returns:
        FederatedRoundOrchestrator
    """
    import json
    from pathlib import Path
    federation_config = config.get("federation", {})
    aggregation_name = str(federation_config.get("aggregation_algorithm", "fedprox")).strip().lower()
    num_hospitals = len(hospitals)
    if num_hospitals == 1:
        if aggregation_name != "no_operation":
            logging.warning("Single-hospital mode detected. Overriding aggregation algorithm to 'no_operation'.")
        aggregation_name = "no_operation"
    elif aggregation_name == "no_operation":
        raise ValueError("'no_operation' aggregation can only be used with a single hospital.")
    # Build aggregator args from config for adaptive or fedprox behavior
    aggregator_kwargs = {}

    if aggregation_name == "fedprox":
        aggregator_kwargs["mu"] = float(federation_config.get("fedprox", {}).get("mu", 0.1))

    if aggregation_name == "adaptive":
        adaptive_cfg = federation_config.get("adaptive", {})
        from src.client_side.hospital.config_helpers import get_cancer_types
        aggregator_kwargs.update({
            "alpha": float(adaptive_cfg.get("alpha", 0.5)),
            "beta": float(adaptive_cfg.get("beta", 0.3)),
            "gamma": float(adaptive_cfg.get("gamma", 0.2)),
            "auc_weight": float(adaptive_cfg.get("auc_weight", 0.75)),
            "f1_weight": float(adaptive_cfg.get("f1_weight", 0.25)),
            "lifecycle_penalty": float(adaptive_cfg.get("lifecycle_penalty", 0.35)),
            "warning_penalty_per_item": float(adaptive_cfg.get("warning_penalty_per_item", 0.05)),
            "min_reliability_score": float(adaptive_cfg.get("min_reliability_score", 0.0)),
            "cancer_types": get_cancer_types(config),
        })

    orchestrator = FederatedRoundOrchestrator.from_algorithm(name=aggregation_name, **aggregator_kwargs)

    num_epochs = int(config.get("simulation", {}).get("num_epoch", 1))
    num_rounds = int(config.get("simulation", {}).get("num_rounds", 1))
    if num_epochs < 1 or num_rounds < 1:
        raise ValueError("simulation.num_epoch and simulation.num_rounds must both be positive integers.")

    # Optionally prepare output directory and log paths
    out_dir = Path(config.get("output", {}).get("history_dir", "outputs/history"))
    hospital_logs = {hid: [] for hid in hospitals}
    log_paths = {hid: out_dir / f"{hid}_metrics.json" for hid in hospitals}
    federated_decision_log = []
    federated_decision_path = out_dir / "federated_decision.json"
    if save_history:
        out_dir.mkdir(parents=True, exist_ok=True)

    early_stop_threshold = config.get("simulation", {}).get("early_stop_threshold")
    early_stop_metric = config.get("simulation", {}).get("early_stop_metric", "loss")
    early_stop_patience = int(config.get("simulation", {}).get("early_stop_patience", 10))
    checkpoint_base = Path(config.get("output", {}).get("checkpoint_dir", "outputs/checkpoints"))

    def validation_callback(hospitals_to_validate):
        validation_result = validation_system(
            hospitals_to_validate,
            output_dir=out_dir,
            early_stop_threshold=early_stop_threshold,
            save_to_disk=save_history,
        )
        if isinstance(validation_result, tuple):
            reports, stop_flag = validation_result
        else:
            reports, stop_flag = validation_result, False
        return {
            "validation_reports": reports,
            "should_stop": stop_flag,
        }

    out = orchestrator.run_with_early_stopping(
        hospitals=hospitals,
        num_epochs=num_epochs,
        num_rounds=num_rounds,
        validation_fn=validation_callback,
        monitor_metric=str(early_stop_metric).lower(),
        patience=early_stop_patience,
    )

    # Convert round history into legacy log format for idempotency.
    for round_output in out["rounds"]:
        epoch_idx = ((round_output.round_index - 1) // num_rounds) + 1
        round_idx = ((round_output.round_index - 1) % num_rounds) + 1

        for hid, hospital in hospitals.items():
            entry = {
                "epoch": epoch_idx,
                "round": round_idx,
                "global_round": round_output.round_index,
                "metrics": getattr(hospital, "metrics_store", {}),
                "local_update": hospital.get_local_update(),
                "global_state": round_output.global_state,
            }
            hospital_logs[hid].append(entry)
            if save_history:
                with open(log_paths[hid], "w") as f:
                    json.dump(hospital_logs[hid], f, indent=2)

        federated_decision_log.append({
            "epoch": epoch_idx,
            "round": round_idx,
            "global_round": round_output.round_index,
            "aggregator_name": round_output.aggregator_name,
            "global_state": round_output.global_state,
            "aggregation": {
                "algorithm": round_output.aggregation.algorithm,
                "global_metrics": round_output.aggregation.global_metrics,
                "hospital_weights": round_output.aggregation.hospital_weights,
                "included_hospital_ids": round_output.aggregation.included_hospital_ids,
                "dropped_hospitals": round_output.aggregation.dropped_hospitals,
                "details": round_output.aggregation.details,
            },
            "validation_report": round_output.validation_report,
        })

        if save_history:
            with open(federated_decision_path, "w") as f:
                json.dump(federated_decision_log, f, indent=2)

    # Checkpoint best epoch if available and stop state
    best_epoch = out.get("best_epoch")
    should_stop = out.get("stopped_early", False)

    if best_epoch is not None:
        _save_epoch_checkpoint(hospitals, checkpoint_base / f"epoch_{best_epoch}")

        if should_stop:
            logging.info(f"Early stopping triggered at epoch {best_epoch}. Restoring best model from epoch {best_epoch}.")
            _restore_epoch_checkpoint(hospitals, checkpoint_base / f"epoch_{best_epoch}")
            print(f"Early stopping triggered at epoch {best_epoch}. Restoring best model from epoch {best_epoch}.")
        else:
            logging.info(f"Training completed. Restoring best model from epoch {best_epoch}.")
            _restore_epoch_checkpoint(hospitals, checkpoint_base / f"epoch_{best_epoch}")
            print(f"Training completed. Restoring best model from epoch {best_epoch}." )

    # After training, optionally save all models for each hospital to outputs/system
    if save_models:
        system_dir = Path("outputs/system")
        system_dir.mkdir(parents=True, exist_ok=True)
        for hid, hospital in hospitals.items():
            if hasattr(hospital, "scope") and hasattr(hospital.scope, "agent_portfolio"):
                portfolio = hospital.scope.agent_portfolio
                if hasattr(portfolio, "save_all_models"):
                    portfolio.save_all_models(str(system_dir), hid)

    return orchestrator

    # on completion, if early stop was not triggered, optionally restore best model
    if best_epoch is not None and (not should_stop):
        logging.info(f"Training completed. Restoring best model from epoch {best_epoch}.")
        print(f"Training completed. Restoring best model from epoch {best_epoch}." )
        _restore_epoch_checkpoint(hospitals, checkpoint_base / f"epoch_{best_epoch}")

    # After training, optionally save all models for each hospital to outputs/system
    if save_models:
        system_dir = Path("outputs/system")
        system_dir.mkdir(parents=True, exist_ok=True)
        for hid, hospital in hospitals.items():
            if hasattr(hospital, "scope") and hasattr(hospital.scope, "agent_portfolio"):
                portfolio = hospital.scope.agent_portfolio
                if hasattr(portfolio, "save_all_models"):
                    portfolio.save_all_models(str(system_dir), hid)
    return orchestrator

def test_system(hospitals):
    """Evaluate all hospitals' agents on their test data."""
    results = {}
    import traceback
    for hid, hospital in hospitals.items():
        try:
            results[hid] = hospital.evaluate()
            logging.info(f"Tested hospital {hid}")
        except Exception as e:
            logging.error(f"Testing failed for hospital {hid}: {e}\nTraceback:\n{traceback.format_exc()}")
            results[hid] = None
    return results


def test_system_on_external_data(hospitals, x_external, cancer_external):
    """Evaluate all hospitals' agents on the same external data set."""
    results = {}
    import traceback
    for hid, hospital in hospitals.items():
        try:
            results[hid] = hospital.evaluate_on_external_data(x_external, cancer_external)
            logging.info(f"External test completed for hospital {hid}")
        except Exception as e:
            logging.error(f"External testing failed for hospital {hid}: {e}\nTraceback:\n{traceback.format_exc()}")
            results[hid] = None
    return results


def show_results(results):
    """Log evaluation results."""
    logging.info("===== Evaluation Results =====")
    print("===== Evaluation Results =====")
    for hid, res in results.items():
        logging.info(f"Hospital: {hid}")
        if res is None:
            logging.warning("  Evaluation failed.")
        else:
            for k, v in res.items():
                logging.info(f"  {k}: {v}")
    logging.info("============================")


def show_log_location(config):
    log_dir = Path(config.get("tracking", {}).get("log_dir", "outputs/logs"))
    log_file = config.get("tracking", {}).get("log_file_name", "simulation.log")
    logging.info(f"Logs saved at: {log_dir / log_file}")

def allocate_data(hospitals):
    """Load data and initialize each hospital."""
    for hospital in hospitals.values():
        hospital.initialize()

def ensure_output_dirs(config):
    """Ensure output and log directories exist."""
    out_dir = Path(config.get("out_dir", "outputs"))
    log_dir = Path(config.get("tracking", {}).get("log_dir", out_dir / "logs"))
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)


def validation_system(hospitals, output_dir=None, early_stop_threshold=None, save_to_disk=False):
    import json
    from pathlib import Path

    if output_dir is None:
        output_dir = Path("outputs/history")
    else:
        output_dir = Path(output_dir)

    if save_to_disk:
        output_dir.mkdir(parents=True, exist_ok=True)

    validation_reports = {}
    for hid, hospital in hospitals.items():
        if hospital.local_data is None:
            raise RuntimeError(f"Hospital {hid} must be initialized before validation_system().")
        if hospital.scope is None or hospital.scope.agent_portfolio is None:
            raise RuntimeError(f"Hospital {hid} must have an agent portfolio before validation_system().")

        hospital_validation = {}
        for cancer_type in hospital.scope.agent_portfolio.cancer_types:
            agent = hospital.scope.agent_portfolio.get_agent(cancer_type)
            x_val, y_val = hospital.get_cancer_filtered_split(cancer_type=cancer_type, split="val")
            if x_val is None or len(x_val) == 0:
                hospital_validation[cancer_type] = {
                    "agent": agent.name,
                    "metrics": {},
                    "warning": "No validation samples for this cancer type.",
                }
                continue

            val_probs = agent.predict_proba(x_val)
            metrics = hospital._compute_binary_metrics(y_val, val_probs)
            hospital_validation[cancer_type] = {
                "agent": agent.name,
                "metrics": metrics,
            }

        hospital.metrics_store.setdefault("validation", {})["results"] = hospital_validation
        hospital.metrics_store["lifecycle_state"] = "validated"

        validation_reports[hid] = hospital_validation

        if save_to_disk:
            with open(output_dir / f"{hid}_validation_metrics.json", "w") as f:
                json.dump({"hospital_id": hid, "validation": hospital_validation}, f, indent=2)

            logging.info(f"Saved validation metrics for hospital {hid} to {output_dir / f'{hid}_validation_metrics.json'}")
            print(f"Saved validation metrics for hospital {hid} to {output_dir / f'{hid}_validation_metrics.json'}")
        else:
            logging.info(f"Validation metrics computed for hospital {hid}; disk save is disabled.")
            print(f"Validation metrics computed for hospital {hid}; disk save is disabled.")


    should_stop = False
    if early_stop_threshold is not None:
        # simple early stopping rule: stop if average validation f1 drops below threshold.
        all_f1_scores = []
        for hid, hospital_validation in validation_reports.items():
            for cancer_cfg in hospital_validation.values():
                score = cancer_cfg.get("metrics", {}).get("f1")
                if score is not None:
                    all_f1_scores.append(score)

        if len(all_f1_scores) > 0:
            avg_f1 = float(sum(all_f1_scores) / len(all_f1_scores))
            should_stop = avg_f1 < float(early_stop_threshold)
            logging.info(f"Validation avg_f1={avg_f1:.4f}, early_stop_threshold={early_stop_threshold}, should_stop={should_stop}")

    if early_stop_threshold is not None:
        return validation_reports, should_stop
    return validation_reports


# For GUI integration: call these functions in the desired order.
# Example usage in GUI:
#   config = load_config()
#   hospitals = initialize_system(config)
#   orchestrator = train_system(config, hospitals)
#   results = test_system(hospitals)
#   show_results(results)
#   show_log_location(config)

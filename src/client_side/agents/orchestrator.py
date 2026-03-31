from typing import Dict

class OrchestratorAgent:
    """
    Aggregates outputs from all cancer agents and makes the final decision.
    If the highest confidence is below threshold, predicts 'CLEAN' (no cancer).
    """
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def predict(self, agent_outputs: Dict[str, float]) -> str:
        """
        agent_outputs: dict of {cancer_type: confidence}
        Returns: cancer_type or 'CLEAN'
        """
        best_type, best_score = max(agent_outputs.items(), key=lambda x: x[1])
        if best_score < self.threshold:
            return "CLEAN"
        return best_type

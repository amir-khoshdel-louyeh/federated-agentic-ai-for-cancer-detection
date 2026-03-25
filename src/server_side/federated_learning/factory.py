from __future__ import annotations

from typing import Any, Callable

from .aggregators import AdaptiveAggregator, FedAvgAggregator, FedProxAggregator, NoOperationAggregator
from .contracts import AggregatorContract

_SUPPORTED_BUILDERS: dict[str, Callable[..., AggregatorContract]] = {
    "fedavg": FedAvgAggregator,
    "fedprox": FedProxAggregator,
    "adaptive": AdaptiveAggregator,
    "no_operation": NoOperationAggregator,
}


def supported_aggregator_names() -> tuple[str, ...]:
    """Return the supported aggregator identifiers for runtime selection."""
    return tuple(_SUPPORTED_BUILDERS.keys())


def build_aggregator(name: str, **kwargs: Any) -> AggregatorContract:
    """Build a supported aggregator by name with optional constructor kwargs."""
    normalized_name = str(name).strip().lower()
    if normalized_name not in _SUPPORTED_BUILDERS:
        allowed = ", ".join(supported_aggregator_names())
        raise ValueError(
            f"Unsupported aggregation algorithm `{name}`. "
            f"Allowed values are: {allowed}."
        )

    builder = _SUPPORTED_BUILDERS[normalized_name]
    try:
        return builder(**kwargs)
    except TypeError as exc:
        raise ValueError(
            f"Invalid parameters for `{normalized_name}` aggregator: {exc}"
        ) from exc

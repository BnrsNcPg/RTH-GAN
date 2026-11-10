"""Configuration contract for the future radiative-transfer discriminator."""

from .context import (
    RTDContext,
    deterministic_context_json,
    load_rtd_context,
    rtd_context_from_mapping,
    validate_rtd_context,
)
from .normalized_proxy import NormalizedProxyRadiativeExtractor

__all__ = [
    "RTDContext",
    "deterministic_context_json",
    "load_rtd_context",
    "rtd_context_from_mapping",
    "validate_rtd_context",
    "NormalizedProxyRadiativeExtractor",
]

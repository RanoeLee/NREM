"""Compatibility imports for the historical ``mr_resa`` module name."""

from .rea import (
    MRResa,
    ResidualEvidenceAggregator,
    build_reference,
    load_reference,
    save_reference,
)

__all__ = [
    "MRResa",
    "ResidualEvidenceAggregator",
    "build_reference",
    "load_reference",
    "save_reference",
]

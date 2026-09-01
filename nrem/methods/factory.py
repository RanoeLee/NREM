from __future__ import annotations
from .advanced_baselines import AdvancedBaselineMethod
from .classical_baselines import ClassicalBaselineMethod
from .deepsvdd import DeepSVDDMethod
from .nrem import NREMMethod
from .official_baselines import OfficialBaselineMethod
from .official_usad import OfficialUSADMethod
from .reconstruction_baselines import ReconstructionBaselineMethod


def build_method(cfg: dict) -> NREMMethod | DeepSVDDMethod | ReconstructionBaselineMethod | OfficialUSADMethod | OfficialBaselineMethod | AdvancedBaselineMethod | ClassicalBaselineMethod:
    name = str((cfg.get("method") or {}).get("name", "nrem")).lower()
    if name == "nrem":
        return NREMMethod(cfg)
    if name == "deepsvdd":
        return DeepSVDDMethod(cfg)
    if name == "official_usad":
        return OfficialUSADMethod(cfg)
    if name in {"dagmm_official", "omnianomaly_official", "anomaly_transformer_official", "tranad_official"}:
        return OfficialBaselineMethod(cfg)
    if name in AdvancedBaselineMethod.SUPPORTED:
        return AdvancedBaselineMethod(cfg)
    if name in ClassicalBaselineMethod.SUPPORTED:
        return ClassicalBaselineMethod(cfg)
    if name in {"mlp_ae", "lstm_ae", "tcn_ae", "usad", "dagmm_style", "omni_lite", "transformer_ae"}:
        return ReconstructionBaselineMethod(cfg)
    raise ValueError(f"Unknown method: {name!r}")

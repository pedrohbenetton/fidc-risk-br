"""Subpacote de Engenharia de Features Point-in-Time e Grafo de Contágio (Gold)."""

from fidc_risk.features.gold import FEATURE_FAMILIES, build_gold_panel, compute_gold_features

__all__ = ["FEATURE_FAMILIES", "build_gold_panel", "compute_gold_features"]

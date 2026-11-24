"""Contratos Pydantic v2 para a API de Risco, Explicabilidade SHAP e Simulação Contrafactual."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ShapDriverOut(BaseModel):
    feature: str
    label: str
    shap_log_odds: float
    feature_value: float


class WatchlistItemOut(BaseModel):
    entity_id: str
    cnpj_raw: str
    denom_social: str
    dominant_sector: str
    nome_gestor: str
    is_aberto: int
    vl_pl: float
    prob_distress_6m: float
    va_p0: float
    va_p1: float
    va_width: float
    inad_90p_pct: float
    inad_early_1_60_pct: float
    subordinacao_pct: float
    recompra_masking_index: float
    scr_provision_gap_pct: float
    contagion_cedente_inad90p_wt: float
    cqr_inad90_h3_q10: float
    cqr_inad90_h3_q50: float
    cqr_inad90_h3_q90: float
    shap_drivers: list[ShapDriverOut] = Field(default_factory=list)


class CounterfactualRequest(BaseModel):
    entity_id: str | None = Field(
        default=None,
        description="CNPJ (14 dígitos) de um fundo real usado como estado-base para o choque contrafactual.",
    )
    feature_overrides: dict[str, float] = Field(
        default_factory=dict,
        description="Mapa {nome_da_feature: novo_valor} aplicando choques de estresse sobre o fundo.",
    )


class CounterfactualResponse(BaseModel):
    entity_id: str | None
    base_prob_6m: float
    shocked_prob_6m: float
    delta_prob_6m: float
    shocked_va_p0: float
    shocked_va_p1: float
    shocked_va_width: float
    shocked_cqr_inad90_h3_q50: float
    shocked_cqr_inad90_h3_q90: float
    top_shap_drivers: list[ShapDriverOut]


class HealthResponse(BaseModel):
    status: str
    version: str
    model_loaded: bool
    live_watchlist_count: int
    latest_competence: str | None

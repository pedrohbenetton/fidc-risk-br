"""Camada Gold: Feature Store Point-in-Time, Sinais Forenses, Contágio LOO e Alvos Preditivos."""

from __future__ import annotations

import json
import time

import duckdb
import numpy as np
import pandas as pd

from fidc_risk.config import (
    GOLD_DIR,
    LAKEHOUSE_DB_PATH,
    REPORTS_DIR,
    SILVER_DIR,
    ensure_directories,
)
from fidc_risk.lakehouse.transforms import strip_accents_upper

# Famílias ortogonais de features para estudo sistemático de ablação (Fase 3)
FEATURE_FAMILIES: dict[str, list[str]] = {
    "aging_structure": [
        "inad_1_30_pct",
        "inad_31_60_pct",
        "inad_61_90_pct",
        "inad_early_1_60_pct",
        "inad_early_1_90_pct",
        "inad_90p_pct",
        "inad_severe_180p_pct",
        "inad_total_pct",
        "prazo_curto_60d_share",
        "prazo_medio_180d_share",
        "prazo_longo_360p_share",
        "weighted_maturity_days",
        "inad_90p_lag1",
        "inad_90p_lag3",
        "inad_90p_delta_1m",
        "inad_90p_delta_3m",
        "inad_90p_delta_6m",
        "inad_90p_vol_6m",
        "inad_early_delta_1m",
        "inad_early_delta_3m",
        "early_to_late_ratio",
        "roll_30_to_60_rate",
        "roll_60_to_90_rate",
    ],
    "forensic_accounting": [
        "pdd_carteira_pct",
        "pdd_coverage_90p",
        "unprovisioned_inad90p_pl",
        "recompra_carteira_pct",
        "substituicao_carteira_pct",
        "recompra_3m_avg_pct",
        "substituicao_3m_avg_pct",
        "recompra_masking_index",
        "has_scr_report",
        "scr_efgh_share",
        "scr_provision_gap_pct",
        "emp_recup_judicial_pct",
        "acao_judicial_pct",
        "cota_fidc_cross_pct",
        "cota_fidc_np_pct",
    ],
    "waterfall_liquidity": [
        "subordinacao_pct",
        "mezanino_pct",
        "total_enhancement_pct",
        "net_cushion_over_bad_debt",
        "subordinacao_delta_3m",
        "subordinacao_delta_6m",
        "pl_log",
        "pl_delta_3m_pct",
        "pl_drawdown_12m_pct",
        "rentab_senior_mes_clean",
        "rentab_senior_excess_cdi",
        "rentab_senior_3m_excess",
        "rentab_subord_mes_clean",
        "senior_perf_shortfall",
        "cash_liquidity_pct",
        "resgate_solicitado_pl_pct",
        "net_flow_3m_pl_pct",
        "alm_liq_30d_ratio",
        "is_aberto",
        "is_exclusivo",
        "has_cotista_interesse",
        "redemption_lag_days",
    ],
    "originator_sector": [
        "n_cedentes_disclosed",
        "pj_cedente_share",
        "top1_cedente_pct",
        "top3_cedente_pct",
        "cedente_hhi",
        "top1_cedente_delta_3m",
        "cedente_churn_1m",
        "sec_industrial_pct",
        "sec_imobiliario_pct",
        "sec_comercial_varejo_pct",
        "sec_servicos_pct",
        "sec_agronegocio_pct",
        "sec_cred_pessoal_pct",
        "sec_consignado_pct",
        "sec_corporativo_pct",
        "sec_veiculos_pct",
        "sec_cartoes_pct",
        "sec_factoring_pct",
        "sec_publico_judicial_pct",
        "sector_hhi",
        "pr_garantia_pct",
        "fund_age_months",
    ],
    "network_contagion": [
        "contagion_shared_cedente_deg",
        "contagion_cedente_inad90p_wt",
        "contagion_cedente_early_wt",
        "contagion_cedente_recompra_wt",
        "contagion_cedente_stress_wt",
        "contagion_cedente_max_inad90p",
        "gestor_other_funds_cnt",
        "gestor_loo_inad90p",
        "gestor_loo_stress_rate",
        "admin_loo_stress_rate",
    ],
    "macro_cycle": [
        "cdi_mensal_pct",
        "selic_meta_aa",
        "selic_delta_12m",
        "ipca_12m_pct",
        "juro_real_expost_aa",
        "ibc_br_yoy_pct",
        "inad_sfn_pj_pct",
        "inad_sfn_pf_pct",
        "inad_sfn_pj_delta_6m",
        "spread_sfn_pp",
        "comprometimento_renda_pct",
    ],
}

ALL_FEATURE_COLS: list[str] = [col for cols in FEATURE_FAMILIES.values() for col in cols]

# --- Definições de estresse, elegibilidade e evento-alvo (citadas na página de Metodologia) ---
# Estresse agudo em t (fundo já em crise na data da previsão):
ACUTE_SENIOR_MONTH_PCT = -2.0  # cota sênior cai mais de 2% no mês t
ACUTE_BAD_LEVEL = 0.15  # estoque ruim (vencido 90+ ou PDD, o maior) >= 15% da carteira...
ACUTE_CUSHION_MARGIN = 0.02  # ...e acima da subordinação + mezanino + 2 p.p.
# Coorte elegível ao alerta precoce:
ELIGIBLE_MIN_PL = 5_000_000.0
ELIGIBLE_MIN_CARTEIRA = 2_000_000.0
ELIGIBLE_MAX_BAD = 0.20
# Evento-alvo em (t, t+h]: queda da cota sênior OU salto do estoque ruim que come o colchão.
TARGET_SENIOR_CUM_PCT = -2.0  # retorno acumulado da cota sênior <= -2%...
TARGET_SENIOR_MONTH_PCT = -1.5  # ...ou algum mês <= -1,5%
TARGET_BAD_JUMP = 0.05  # estoque ruim sobe >= 5 p.p. ...
TARGET_BAD_LEVEL = 0.12  # ...chega a >= 12% da carteira...
TARGET_CUSHION_MARGIN = 0.02  # ...e a folga de subordinação cai para <= 2 p.p.

# Restrições monotônicas fundamentadas em teoria de crédito estruturado (+1 aumenta risco, -1 reduz)
MONOTONIC_CONSTRAINTS_MAP: dict[str, int] = {
    "inad_early_1_60_pct": 1,
    "inad_90p_pct": 1,
    "inad_90p_delta_3m": 1,
    "unprovisioned_inad90p_pl": 1,
    "recompra_masking_index": 1,
    "scr_provision_gap_pct": 1,
    "emp_recup_judicial_pct": 1,
    "subordinacao_pct": -1,
    "net_cushion_over_bad_debt": -1,
    "cash_liquidity_pct": -1,
    "resgate_solicitado_pl_pct": 1,
    "contagion_cedente_inad90p_wt": 1,
    "contagion_cedente_stress_wt": 1,
}


def _detect_fidc_np_structural(denom_series: pd.Series) -> pd.Series:
    """Identifica FIDC-NP (Não-Padronizados vocacionados a créditos já vencidos/judiciais)."""
    up = denom_series.fillna("").astype(str).map(strip_accents_upper)
    pattern = r"(?:\bNP\b|FIDC[\s\-]*NP|NAO[\s\-]*PADRONIZADO|CREDITOS?\s+JUDICIA|PRECATORIO|RECUPERACAO\s+DE\s+CREDITO)"
    return up.str.contains(pattern, regex=True)


def _compute_cedente_churn(edges_df: pd.DataFrame) -> pd.DataFrame:
    """Calcula churn de Jaccard de cedentes entre t-1 e t (100% vetorizado e causal)."""
    if len(edges_df) == 0:
        return pd.DataFrame(columns=["entity_id", "dt_comptc", "cedente_churn_1m"])

    curr = edges_df[["entity_id", "dt_comptc", "cedente_doc"]].drop_duplicates().copy()
    curr["ym_int"] = curr["dt_comptc"].dt.year * 12 + curr["dt_comptc"].dt.month

    prev = curr.copy()
    prev["ym_int"] = prev["ym_int"] + 1  # Alinha t-1 exatamente com t

    n_curr = curr.groupby(["entity_id", "dt_comptc", "ym_int"], as_index=False).size().rename(columns={"size": "n_curr"})
    n_prev = prev.groupby(["entity_id", "ym_int"], as_index=False).size().rename(columns={"size": "n_prev"})
    inter = (
        curr.merge(prev[["entity_id", "ym_int", "cedente_doc"]], on=["entity_id", "ym_int", "cedente_doc"], how="inner")
        .groupby(["entity_id", "ym_int"], as_index=False)
        .size()
        .rename(columns={"size": "n_inter"})
    )

    merged = n_curr.merge(n_prev, on=["entity_id", "ym_int"], how="left").merge(
        inter, on=["entity_id", "ym_int"], how="left"
    )
    has_prev = merged["n_prev"].notna() & (merged["n_prev"] > 0)
    n_inter_v = merged["n_inter"].fillna(0.0)
    union_v = merged["n_curr"] + merged["n_prev"].fillna(0.0) - n_inter_v
    churn = np.where(has_prev & (union_v > 0), 1.0 - (n_inter_v / np.maximum(union_v, 1.0)), 0.0)
    merged["cedente_churn_1m"] = np.clip(churn, 0.0, 1.0)
    return merged[["entity_id", "dt_comptc", "cedente_churn_1m"]]


def _compute_loo_contagion_features(
    panel: pd.DataFrame,
    edges_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calcula sinais Leave-One-Out (LOO) de contágio de crédito estritamente dentro de cada mês t:
    1) Arestas FIDC <-> Cedente (ponderadas pela participação do cedente na carteira);
    2) Pares sob o mesmo Gestor (excluindo o próprio fundo);
    3) Pares sob o mesmo Administrador (excluindo o próprio fundo).
    Zero auto-vazamento e zero vazamento temporal futuro.
    """
    state_cols = [
        "dt_comptc",
        "entity_id",
        "inad_90p_pct",
        "inad_early_1_60_pct",
        "recompra_carteira_pct",
        "acute_stress_now",
        "cnpj_gestor",
        "cnpj_admin",
    ]
    st = panel[state_cols].copy()

    # 1. Contágio via Cedentes Compartilhados
    if len(edges_df) > 0:
        ew = edges_df[["dt_comptc", "entity_id", "cedente_doc", "share_pct"]].merge(
            st[["dt_comptc", "entity_id", "inad_90p_pct", "inad_early_1_60_pct", "recompra_carteira_pct", "acute_stress_now"]],
            on=["dt_comptc", "entity_id"],
            how="inner",
        )
        ced_tot = ew.groupby(["dt_comptc", "cedente_doc"], as_index=False).agg(
            c_cnt=("entity_id", "count"),
            c_sum_inad90=("inad_90p_pct", "sum"),
            c_sum_early=("inad_early_1_60_pct", "sum"),
            c_sum_rec=("recompra_carteira_pct", "sum"),
            c_sum_stress=("acute_stress_now", "sum"),
        )
        ew = ew.merge(ced_tot, on=["dt_comptc", "cedente_doc"], how="left")
        peer_cnt = (ew["c_cnt"] - 1).clip(lower=0)
        has_peer = peer_cnt > 0
        denom_p = np.maximum(peer_cnt, 1)

        ew["peer_cnt"] = peer_cnt
        ew["peer_inad90"] = np.where(has_peer, (ew["c_sum_inad90"] - ew["inad_90p_pct"]) / denom_p, 0.0)
        ew["peer_early"] = np.where(has_peer, (ew["c_sum_early"] - ew["inad_early_1_60_pct"]) / denom_p, 0.0)
        ew["peer_rec"] = np.where(has_peer, (ew["c_sum_rec"] - ew["recompra_carteira_pct"]) / denom_p, 0.0)
        ew["peer_stress"] = np.where(has_peer, (ew["c_sum_stress"] - ew["acute_stress_now"]) / denom_p, 0.0)

        w = (ew["share_pct"].clip(0.0, 100.0) / 100.0).values
        ew["w_inad90"] = w * ew["peer_inad90"]
        ew["w_early"] = w * ew["peer_early"]
        ew["w_rec"] = w * ew["peer_rec"]
        ew["w_stress"] = w * ew["peer_stress"]

        ced_feat = ew.groupby(["dt_comptc", "entity_id"], as_index=False).agg(
            contagion_shared_cedente_deg=("peer_cnt", "sum"),
            contagion_cedente_inad90p_wt=("w_inad90", "sum"),
            contagion_cedente_early_wt=("w_early", "sum"),
            contagion_cedente_recompra_wt=("w_rec", "sum"),
            contagion_cedente_stress_wt=("w_stress", "sum"),
            contagion_cedente_max_inad90p=("peer_inad90", "max"),
        )
    else:
        ced_feat = pd.DataFrame(
            columns=[
                "dt_comptc",
                "entity_id",
                "contagion_shared_cedente_deg",
                "contagion_cedente_inad90p_wt",
                "contagion_cedente_early_wt",
                "contagion_cedente_recompra_wt",
                "contagion_cedente_stress_wt",
                "contagion_cedente_max_inad90p",
            ]
        )

    out = st[["dt_comptc", "entity_id", "cnpj_gestor", "cnpj_admin", "inad_90p_pct", "acute_stress_now"]].merge(
        ced_feat, on=["dt_comptc", "entity_id"], how="left"
    )
    for col in (
        "contagion_shared_cedente_deg",
        "contagion_cedente_inad90p_wt",
        "contagion_cedente_early_wt",
        "contagion_cedente_recompra_wt",
        "contagion_cedente_stress_wt",
        "contagion_cedente_max_inad90p",
    ):
        out[col] = out[col].fillna(0.0)

    # 2. Leave-One-Out por Gestor
    valid_g = out["cnpj_gestor"].fillna("") != ""
    g_tot = (
        out.loc[valid_g]
        .groupby(["dt_comptc", "cnpj_gestor"], as_index=False)
        .agg(
            g_cnt=("entity_id", "count"),
            g_sum_inad=("inad_90p_pct", "sum"),
            g_sum_stress=("acute_stress_now", "sum"),
        )
    )
    out = out.merge(g_tot, on=["dt_comptc", "cnpj_gestor"], how="left")
    g_other = (out["g_cnt"].fillna(0) - 1).clip(lower=0)
    out["gestor_other_funds_cnt"] = np.where(valid_g, g_other, 0.0)
    out["gestor_loo_inad90p"] = np.where(
        valid_g & (g_other > 0),
        (out["g_sum_inad"] - out["inad_90p_pct"]) / np.maximum(g_other, 1),
        0.0,
    )
    out["gestor_loo_stress_rate"] = np.where(
        valid_g & (g_other > 0),
        (out["g_sum_stress"] - out["acute_stress_now"]) / np.maximum(g_other, 1),
        0.0,
    )

    # 3. Leave-One-Out por Administrador
    valid_a = out["cnpj_admin"].fillna("") != ""
    a_tot = (
        out.loc[valid_a]
        .groupby(["dt_comptc", "cnpj_admin"], as_index=False)
        .agg(a_cnt=("entity_id", "count"), a_sum_stress=("acute_stress_now", "sum"))
    )
    out = out.merge(a_tot, on=["dt_comptc", "cnpj_admin"], how="left")
    a_other = (out["a_cnt"].fillna(0) - 1).clip(lower=0)
    out["admin_loo_stress_rate"] = np.where(
        valid_a & (a_other > 0),
        (out["a_sum_stress"] - out["acute_stress_now"]) / np.maximum(a_other, 1),
        0.0,
    )

    keep = ["dt_comptc", "entity_id"] + FEATURE_FAMILIES["network_contagion"]
    return out[keep]


def compute_gold_features(
    core_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    macro_df: pd.DataFrame,
    include_targets: bool = True,
) -> pd.DataFrame:
    """
    Transforma tabelas Silver em painel Gold point-in-time.
    Garantia matemática: todas as colunas em ALL_FEATURE_COLS na data t dependem EXCLUSIVAMENTE
    de observações com dt_comptc <= t. Apenas colunas 'target_*' e 'fwd_*' olham para t+1..t+h.
    """
    df = core_df.sort_values(["entity_id", "dt_comptc"]).reset_index(drop=True).copy()

    # Merge macroeconômico (competência mensal <= t)
    macro_cols = [
        "dt_comptc",
        "cdi_mensal_pct",
        "cdi_12m_pct",
        "selic_meta_aa",
        "selic_delta_12m",
        "ipca_mensal_pct",
        "ipca_12m_pct",
        "juro_real_expost_aa",
        "ibc_br_yoy_pct",
        "inad_sfn_total_pct",
        "inad_sfn_pj_pct",
        "inad_sfn_pf_pct",
        "inad_sfn_pj_delta_6m",
        "spread_sfn_pp",
        "endividamento_familias_pct",
        "comprometimento_renda_pct",
        "desocupacao_pnad_pct",
    ]
    avail_mcols = [c for c in macro_cols if c in macro_df.columns]
    df = df.merge(macro_df[avail_mcols], on="dt_comptc", how="left")
    for mc in avail_mcols:
        if mc != "dt_comptc":
            df[mc] = df[mc].ffill().bfill().fillna(0.0)

    # Denominador robusto da carteira de crédito
    aging_total_inad = df[
        [
            "inad_venc_30",
            "inad_venc_60",
            "inad_venc_90",
            "inad_venc_120",
            "inad_venc_150",
            "inad_venc_180",
            "inad_venc_360",
            "inad_venc_720",
            "inad_venc_1080",
            "inad_venc_maior_1080",
        ]
    ].sum(axis=1)
    aging_total_prazo = df[
        [
            "prazo_venc_30",
            "prazo_venc_60",
            "prazo_venc_90",
            "prazo_venc_120",
            "prazo_venc_150",
            "prazo_venc_180",
            "prazo_venc_360",
            "prazo_venc_720",
            "prazo_venc_1080",
            "prazo_venc_maior_1080",
        ]
    ].sum(axis=1)

    carteira_denom = np.maximum.reduce(
        [
            df["vl_carteira"].values,
            (aging_total_prazo + aging_total_inad).values,
            np.ones(len(df), dtype="float64"),
        ]
    )
    pl_denom = np.maximum(df["vl_pl"].values, 1.0)
    ativo_denom = np.maximum(df["vl_ativo"].values, 1.0)

    # --- Família 1: Aging Structure & Roll-Rate Dynamics ---
    inad_90p_vl = df[
        [
            "inad_venc_120",
            "inad_venc_150",
            "inad_venc_180",
            "inad_venc_360",
            "inad_venc_720",
            "inad_venc_1080",
            "inad_venc_maior_1080",
        ]
    ].sum(axis=1)
    inad_180p_vl = df[
        ["inad_venc_360", "inad_venc_720", "inad_venc_1080", "inad_venc_maior_1080"]
    ].sum(axis=1)

    df["inad_1_30_pct"] = (df["inad_venc_30"] / carteira_denom).clip(0.0, 1.0)
    df["inad_31_60_pct"] = (df["inad_venc_60"] / carteira_denom).clip(0.0, 1.0)
    df["inad_61_90_pct"] = (df["inad_venc_90"] / carteira_denom).clip(0.0, 1.0)
    df["inad_early_1_60_pct"] = ((df["inad_venc_30"] + df["inad_venc_60"]) / carteira_denom).clip(0.0, 1.0)
    df["inad_early_1_90_pct"] = (
        (df["inad_venc_30"] + df["inad_venc_60"] + df["inad_venc_90"]) / carteira_denom
    ).clip(0.0, 1.0)
    df["inad_90p_pct"] = (inad_90p_vl / carteira_denom).clip(0.0, 1.0)
    df["inad_severe_180p_pct"] = (inad_180p_vl / carteira_denom).clip(0.0, 1.0)
    df["inad_total_pct"] = (np.maximum(df["vl_dircred_inad_total"], aging_total_inad) / carteira_denom).clip(0.0, 1.0)

    prazo_denom = np.maximum(aging_total_prazo.values, 1.0)
    df["prazo_curto_60d_share"] = ((df["prazo_venc_30"] + df["prazo_venc_60"]) / prazo_denom).clip(0.0, 1.0)
    df["prazo_medio_180d_share"] = (
        (df["prazo_venc_90"] + df["prazo_venc_120"] + df["prazo_venc_150"] + df["prazo_venc_180"]) / prazo_denom
    ).clip(0.0, 1.0)
    df["prazo_longo_360p_share"] = (
        (df["prazo_venc_360"] + df["prazo_venc_720"] + df["prazo_venc_1080"] + df["prazo_venc_maior_1080"])
        / prazo_denom
    ).clip(0.0, 1.0)
    df["weighted_maturity_days"] = (
        15.0 * df["prazo_venc_30"]
        + 45.0 * df["prazo_venc_60"]
        + 75.0 * df["prazo_venc_90"]
        + 135.0 * (df["prazo_venc_120"] + df["prazo_venc_150"] + df["prazo_venc_180"])
        + 270.0 * df["prazo_venc_360"]
        + 540.0 * df["prazo_venc_720"]
        + 900.0 * (df["prazo_venc_1080"] + df["prazo_venc_maior_1080"])
    ) / prazo_denom

    # Lags e dinâmicas temporais estritamente causais por entity_id
    grp = df.groupby("entity_id", sort=False)
    df["inad_90p_lag1"] = grp["inad_90p_pct"].shift(1).fillna(df["inad_90p_pct"])
    df["inad_90p_lag3"] = grp["inad_90p_pct"].shift(3).fillna(df["inad_90p_lag1"])
    df["inad_90p_lag6"] = grp["inad_90p_pct"].shift(6).fillna(df["inad_90p_lag3"])

    df["inad_90p_delta_1m"] = df["inad_90p_pct"] - df["inad_90p_lag1"]
    df["inad_90p_delta_3m"] = df["inad_90p_pct"] - df["inad_90p_lag3"]
    df["inad_90p_delta_6m"] = df["inad_90p_pct"] - df["inad_90p_lag6"]
    df["inad_90p_vol_6m"] = (
        grp["inad_90p_pct"].rolling(6, min_periods=2).std().reset_index(level=0, drop=True).fillna(0.0)
    )

    early_lag1 = grp["inad_early_1_60_pct"].shift(1).fillna(df["inad_early_1_60_pct"])
    early_lag3 = grp["inad_early_1_60_pct"].shift(3).fillna(early_lag1)
    df["inad_early_delta_1m"] = df["inad_early_1_60_pct"] - early_lag1
    df["inad_early_delta_3m"] = df["inad_early_1_60_pct"] - early_lag3
    df["early_to_late_ratio"] = (df["inad_early_1_60_pct"] / (df["inad_90p_pct"] + 0.01)).clip(0.0, 25.0)

    prev_v30 = grp["inad_venc_30"].shift(1)
    prev_v60 = grp["inad_venc_60"].shift(1)
    df["roll_30_to_60_rate"] = np.where(
        prev_v30 > 1_000.0, (df["inad_venc_60"] / np.maximum(prev_v30, 1.0)).clip(0.0, 2.0), 0.0
    )
    df["roll_60_to_90_rate"] = np.where(
        prev_v60 > 1_000.0, (df["inad_venc_90"] / np.maximum(prev_v60, 1.0)).clip(0.0, 2.0), 0.0
    )

    # --- Família 2: Sinais Contábeis Forenses (Recompras, Substituições, PDD vs SCR, RJ) ---
    df["pdd_carteira_pct"] = (df["vl_pdd_total"] / carteira_denom).clip(0.0, 1.5)
    df["pdd_coverage_90p"] = np.where(
        inad_90p_vl > 1_000.0, (df["vl_pdd_total"] / np.maximum(inad_90p_vl, 1.0)).clip(0.0, 3.0), 1.0
    )
    df["unprovisioned_inad90p_pl"] = (np.maximum(inad_90p_vl - df["vl_pdd_total"], 0.0) / pl_denom).clip(0.0, 3.0)

    df["recompra_carteira_pct"] = (df["vl_recompras_mes"] / carteira_denom).clip(0.0, 1.0)
    df["substituicao_carteira_pct"] = (df["vl_substituicoes_mes"] / carteira_denom).clip(0.0, 1.0)
    df["recompra_3m_avg_pct"] = (
        grp["recompra_carteira_pct"].rolling(3, min_periods=1).mean().reset_index(level=0, drop=True)
    )
    df["substituicao_3m_avg_pct"] = (
        grp["substituicao_carteira_pct"].rolling(3, min_periods=1).mean().reset_index(level=0, drop=True)
    )

    exit_bad_flow = df["vl_recompras_mes"] + df["vl_substituicoes_mes"]
    masking_share = exit_bad_flow / np.maximum(exit_bad_flow + inad_90p_vl, 1.0)
    df["recompra_masking_index"] = (
        masking_share * (df["inad_early_1_60_pct"] + df["recompra_3m_avg_pct"] + df["substituicao_3m_avg_pct"])
    ).clip(0.0, 1.5)

    has_scr = df["has_scr_report"].fillna(0).astype("int8")
    scr_denom = np.maximum(df["vl_scr_total"].values, 1.0)
    df["has_scr_report"] = has_scr
    df["scr_efgh_share"] = np.where(has_scr == 1, (df["vl_scr_efgh"] / scr_denom).clip(0.0, 1.0), 0.0)
    df["scr_provision_gap_pct"] = np.where(
        has_scr == 1,
        ((df["vl_scr_pdd_min_bacen2682"] - df["vl_pdd_total"]) / carteira_denom).clip(-1.0, 1.0),
        0.0,
    )

    df["emp_recup_judicial_pct"] = (df["vl_cred_emp_recup"] / carteira_denom).clip(0.0, 1.0)
    df["acao_judicial_pct"] = (df["vl_cred_acao_judic"] / carteira_denom).clip(0.0, 1.0)
    df["cota_fidc_cross_pct"] = (df["vl_cota_fidc"] / ativo_denom).clip(0.0, 1.0)
    df["cota_fidc_np_pct"] = (df["vl_cota_fidc_np"] / ativo_denom).clip(0.0, 1.0)

    # --- Família 3: Waterfall de Tranches, Subordinação e Descasamento de Liquidez ---
    tranche_sum = df["pl_senior"] + df["pl_mezanino"] + df["pl_subordinada"] + df["pl_unica"]
    wf_denom = np.where(tranche_sum > 0, tranche_sum, pl_denom)
    df["subordinacao_pct"] = np.where(tranche_sum > 0, (df["pl_subordinada"] / wf_denom).clip(0.0, 1.0), 0.0)
    df["mezanino_pct"] = np.where(tranche_sum > 0, (df["pl_mezanino"] / wf_denom).clip(0.0, 1.0), 0.0)
    df["total_enhancement_pct"] = (df["subordinacao_pct"] + df["mezanino_pct"]).clip(0.0, 1.0)
    df["net_cushion_over_bad_debt"] = (
        df["total_enhancement_pct"] - np.maximum(df["inad_90p_pct"] - df["pdd_carteira_pct"], 0.0)
    ).clip(-1.0, 1.0)

    sub_lag3 = grp["subordinacao_pct"].shift(3).fillna(df["subordinacao_pct"])
    sub_lag6 = grp["subordinacao_pct"].shift(6).fillna(sub_lag3)
    df["subordinacao_delta_3m"] = df["subordinacao_pct"] - sub_lag3
    df["subordinacao_delta_6m"] = df["subordinacao_pct"] - sub_lag6

    df["pl_log"] = np.log1p(np.maximum(df["vl_pl"], 0.0))
    pl_lag3 = grp["vl_pl"].shift(3)
    df["pl_delta_3m_pct"] = np.where(pl_lag3 > 10_000.0, ((df["vl_pl"] - pl_lag3) / pl_lag3).clip(-1.0, 3.0), 0.0)
    pl_roll_max = grp["vl_pl"].rolling(12, min_periods=1).max().reset_index(level=0, drop=True)
    df["pl_drawdown_12m_pct"] = np.where(
        pl_roll_max > 10_000.0, ((df["vl_pl"] - pl_roll_max) / pl_roll_max).clip(-1.0, 0.0), 0.0
    )

    # Retorno mensal efetivo da cota investível (Sênior quando existe, senão Única)
    eff_sr_ret = df["rentab_senior_mes"].combine_first(df["rentab_unica_mes"])
    df["eff_senior_ret_mes"] = eff_sr_ret
    df["rentab_senior_mes_clean"] = eff_sr_ret.fillna(df["cdi_mensal_pct"])
    df["rentab_senior_excess_cdi"] = (df["rentab_senior_mes_clean"] - df["cdi_mensal_pct"]).clip(-50.0, 30.0)
    df["rentab_senior_3m_excess"] = (
        df.groupby("entity_id", sort=False)["rentab_senior_excess_cdi"]
        .rolling(3, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )
    df["rentab_subord_mes_clean"] = df["rentab_subordinada_mes"].fillna(0.0).clip(-80.0, 50.0)
    df["senior_perf_shortfall"] = (
        (df["desemp_real_senior"] - df["desemp_esperado_senior"]).fillna(0.0).clip(-50.0, 50.0)
    )

    df["cash_liquidity_pct"] = ((df["vl_disp"] + df["vl_titpub_liq"]) / pl_denom).clip(0.0, 1.5)
    df["resgate_solicitado_pl_pct"] = (df["resgates_solicitados_total"] / pl_denom).clip(0.0, 1.5)
    net_flow_1m = (df["captacoes_total"] - df["resgates_pagos_total"] - df["amortizacoes_total"]) / pl_denom
    df["_nf_1m"] = net_flow_1m.clip(-1.0, 1.0)
    df["net_flow_3m_pl_pct"] = (
        df.groupby("entity_id", sort=False)["_nf_1m"].rolling(3, min_periods=1).sum().reset_index(level=0, drop=True)
    )
    df = df.drop(columns=["_nf_1m"])

    liq_demand_30d = np.maximum(df["resgates_solicitados_total"], 0.05 * df["vl_pl"])
    df["alm_liq_30d_ratio"] = ((df["liq_imediata_0"] + df["liq_30d"]) / np.maximum(liq_demand_30d, 1.0)).clip(0.0, 10.0)
    df["redemption_lag_days"] = (df["prazo_conversao_dias"] + df["prazo_pagto_resgate_dias"]).clip(0.0, 1080.0)
    df = df.copy()  # Desfragmenta blocos de memória antes da Família 4 e alvos

    # --- Família 4: Concentração de Cedentes, Churn e Setores ---
    df["pj_cedente_share"] = np.where(
        df["n_cedentes_disclosed"] > 0,
        (df["n_cedentes_pj"] / np.maximum(df["n_cedentes_disclosed"], 1.0)).clip(0.0, 1.0),
        0.0,
    )
    df["top1_cedente_pct"] = (df["top1_cedente_pct"] / 100.0).clip(0.0, 1.0)
    df["top3_cedente_pct"] = (df["top3_cedente_pct"] / 100.0).clip(0.0, 1.0)
    df["cedente_hhi"] = df["cedente_hhi"].clip(0.0, 1.0)
    top1_lag3 = df.groupby("entity_id", sort=False)["top1_cedente_pct"].shift(3).fillna(df["top1_cedente_pct"])
    df["top1_cedente_delta_3m"] = df["top1_cedente_pct"] - top1_lag3

    churn_df = _compute_cedente_churn(edges_df)
    df = df.merge(churn_df, on=["entity_id", "dt_comptc"], how="left")
    df["cedente_churn_1m"] = df["cedente_churn_1m"].fillna(0.0)

    sec_raw_map = {
        "sec_industrial_pct": "sec_industrial",
        "sec_imobiliario_pct": "sec_imobiliario",
        "sec_comercial_varejo_pct": "sec_comercial_varejo",
        "sec_servicos_pct": "sec_servicos",
        "sec_agronegocio_pct": "sec_agronegocio",
        "sec_cred_pessoal_pct": "sec_cred_pessoal",
        "sec_consignado_pct": "sec_consignado",
        "sec_corporativo_pct": "sec_corporativo",
        "sec_veiculos_pct": "sec_veiculos",
        "sec_cartoes_pct": "sec_cartoes_credito",
        "sec_factoring_pct": "sec_factoring",
    }
    sec_denom = np.maximum(df.get("vl_carteira_tab2", df["vl_carteira"]).values, carteira_denom)
    for out_s, raw_s in sec_raw_map.items():
        df[out_s] = (df[raw_s] / sec_denom).clip(0.0, 1.0)
    df["sec_publico_judicial_pct"] = ((df["sec_setor_publico"] + df["sec_judicial"]) / sec_denom).clip(0.0, 1.0)

    sec_pct_cols = list(sec_raw_map.keys()) + ["sec_publico_judicial_pct"]
    sec_mat = df[sec_pct_cols].values
    df["sector_hhi"] = np.sum(sec_mat**2, axis=1).clip(0.0, 1.0)

    sector_labels = [
        "Industrial",
        "Imobiliário",
        "Comercial/Varejo",
        "Serviços",
        "Agronegócio",
        "Crédito Pessoal",
        "Consignado",
        "Corporativo",
        "Veículos",
        "Cartões",
        "Factoring",
        "Público/Judicial",
    ]
    max_sec_idx = np.argmax(sec_mat, axis=1)
    max_sec_val = np.max(sec_mat, axis=1)
    df["dominant_sector"] = np.where(max_sec_val > 0.05, np.take(sector_labels, max_sec_idx), "Multicarteira")

    df["pr_garantia_pct"] = (df["pr_garantia_dircred"] / 100.0).clip(0.0, 2.0)

    first_seen = df.groupby("entity_id", sort=False)["dt_comptc"].transform("min")
    dt_const = pd.to_datetime(df["dt_constituicao"], errors="coerce").fillna(first_seen)
    dt_origin = np.minimum(dt_const, first_seen)
    df["fund_age_months"] = (((df["dt_comptc"] - dt_origin).dt.days) / 30.4375).clip(lower=0.0, upper=360.0)

    # --- Estado de Estresse Contemporâneo (em t) e Elegibilidade ---
    df["is_fidc_np_structural"] = _detect_fidc_np_structural(df["denom_social"]).astype("int8")
    bad_metric_t = np.maximum(df["inad_90p_pct"], df["pdd_carteira_pct"])
    df["bad_debt_metric"] = bad_metric_t

    # Um fundo já se encontra em estresse agudo na data t se:
    # - sua cota sênior caiu mais de -2.0% no mês t, OU
    # - o estoque ruim (inad >90d ou PDD) supera 15% e excede o colchão de subordinação/mezanino
    sr_shock_now = df["eff_senior_ret_mes"].notna() & (df["eff_senior_ret_mes"] < ACUTE_SENIOR_MONTH_PCT)
    structural_breach_now = (bad_metric_t >= ACUTE_BAD_LEVEL) & (bad_metric_t > df["total_enhancement_pct"] + ACUTE_CUSHION_MARGIN)
    df["acute_stress_now"] = ((sr_shock_now | structural_breach_now) & (df["is_fidc_np_structural"] == 0)).astype("int8")

    # --- Família 5: Contágio em Rede LOO (Cedentes, Gestores e Administradores em t) ---
    contagion_df = _compute_loo_contagion_features(df, edges_df)
    df = df.merge(contagion_df, on=["dt_comptc", "entity_id"], how="left")

    # Elegibilidade para a coorte de Alerta Precoce (fundos comerciais ativos e saudáveis em t)
    df["is_eligible_watchlist"] = (
        (df["is_fidc_np_structural"] == 0)
        & (df["vl_pl"] >= ELIGIBLE_MIN_PL)
        & (carteira_denom >= ELIGIBLE_MIN_CARTEIRA)
        & (df["acute_stress_now"] == 0)
        & (bad_metric_t < ELIGIBLE_MAX_BAD)
    ).astype("int8")

    # Sanitização final de NaN/Inf em todas as features causais
    for col in ALL_FEATURE_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)

    if not include_targets:
        return df.copy()

    # --- Construção dos Alvos Futuros (estritamente em t+1 .. t+h) ---
    df = df.sort_values(["entity_id", "dt_comptc"]).reset_index(drop=True).copy()
    g = df.groupby("entity_id", sort=False)

    # Verificar continuidade temporal (evitar saltos > 1 mês por buracos de entrega)
    ym_code = df["dt_comptc"].dt.year * 12 + df["dt_comptc"].dt.month
    df["_ym_code"] = ym_code

    for h in (1, 2, 3, 6):
        fut_ym = g["_ym_code"].shift(-h)
        exact_h = fut_ym == (ym_code + h)
        fut_inad90 = g["inad_90p_pct"].shift(-h)
        df[f"target_inad90p_h{h}"] = np.where(exact_h, fut_inad90, np.nan)
        fut_ret = g["eff_senior_ret_mes"].shift(-h)
        df[f"fwd_ret_senior_{h}m_step"] = np.where(exact_h, fut_ret, np.nan)
        fut_cdi = g["cdi_mensal_pct"].shift(-h)
        df[f"fwd_cdi_{h}m_step"] = np.where(exact_h, fut_cdi, np.nan)

    # Janelas futuras de h=3m e h=6m para Alerta Precoce de Estresse Severo (Distress Onset)
    for horizon in (3, 6):
        has_window = g["_ym_code"].shift(-1).notna()
        cum_sr_factor = np.ones(len(df), dtype="float64")
        any_sr_obs = np.zeros(len(df), dtype=bool)
        min_single_sr = np.zeros(len(df), dtype="float64")
        max_fut_bad = np.zeros(len(df), dtype="float64")
        min_fut_cush_diff = np.zeros(len(df), dtype="float64")

        for step in range(1, horizon + 1):
            valid_step = g["_ym_code"].shift(-step) == (ym_code + step)
            r_step = g["eff_senior_ret_mes"].shift(-step)
            has_r = valid_step & r_step.notna()
            any_sr_obs |= has_r.values
            r_clean = np.where(has_r, r_step.fillna(0.0), 0.0)
            cum_sr_factor *= 1.0 + (r_clean / 100.0)
            min_single_sr = np.minimum(min_single_sr, r_clean)

            b_step = np.where(valid_step, g["bad_debt_metric"].shift(-step).fillna(0.0), 0.0)
            enh_step = np.where(valid_step, g["total_enhancement_pct"].shift(-step).fillna(0.0), 1.0)
            max_fut_bad = np.maximum(max_fut_bad, b_step)
            min_fut_cush_diff = np.minimum(min_fut_cush_diff, enh_step - b_step)

        cum_sr_ret_pct = (cum_sr_factor - 1.0) * 100.0
        # Gatilho A: perda nominal acumulada na cota sênior < -2.0% ou tombo mensal < -1.5%
        trig_senior_dd = any_sr_obs & ((cum_sr_ret_pct <= TARGET_SENIOR_CUM_PCT) | (min_single_sr <= TARGET_SENIOR_MONTH_PCT))
        # Gatilho B: salto de >= +5 p.p. em inadimplência >90d / PDD levando estoque ruim >= 12%
        #            e rompendo (ou deixando margem < 2 p.p. sobre) a subordinação
        bad_jump = max_fut_bad - df["bad_debt_metric"].values
        trig_credit_surge = (
            (bad_jump >= TARGET_BAD_JUMP) & (max_fut_bad >= TARGET_BAD_LEVEL) & (min_fut_cush_diff <= TARGET_CUSHION_MARGIN)
        )

        composite = (trig_senior_dd | trig_credit_surge).astype("float64")
        df[f"target_distress_{horizon}m"] = np.where(has_window, composite, np.nan)
        df[f"target_senior_dd_{horizon}m"] = np.where(has_window, trig_senior_dd.astype("float64"), np.nan)
        df[f"target_credit_surge_{horizon}m"] = np.where(has_window, trig_credit_surge.astype("float64"), np.nan)
        df[f"fwd_cum_senior_ret_{horizon}m"] = np.where(has_window & any_sr_obs, cum_sr_ret_pct, np.nan)

    df = df.drop(columns=["_ym_code"])
    return df


def build_gold_panel() -> dict[str, object]:
    """Lê a camada Silver, materializa o painel Gold e registra views analíticas no DuckDB."""
    ensure_directories()
    t0 = time.perf_counter()

    print("[Gold] Carregando tabelas Silver...")
    core_df = pd.read_parquet(SILVER_DIR / "fidc_monthly_core.parquet")
    edges_df = pd.read_parquet(SILVER_DIR / "cedente_edges.parquet")
    macro_df = pd.read_parquet(SILVER_DIR / "macro_monthly.parquet")

    print(f"[Gold] Calculando {len(ALL_FEATURE_COLS)} features point-in-time + grafo LOO + alvos multi-horizonte...")
    gold_df = compute_gold_features(core_df, edges_df, macro_df, include_targets=True)

    gold_path = GOLD_DIR / "panel_gold.parquet"
    gold_df.to_parquet(gold_path, index=False)

    con = duckdb.connect(str(LAKEHOUSE_DB_PATH))
    try:
        con.execute("CREATE SCHEMA IF NOT EXISTS gold;")
        con.execute(f"CREATE OR REPLACE VIEW gold.panel AS SELECT * FROM read_parquet('{gold_path.as_posix()}');")
    finally:
        con.close()

    elig_3m = gold_df[(gold_df["is_eligible_watchlist"] == 1) & (gold_df["target_distress_3m"].notna())]
    elig_6m = gold_df[(gold_df["is_eligible_watchlist"] == 1) & (gold_df["target_distress_6m"].notna())]

    summary = {
        "elapsed_seconds": round(time.perf_counter() - t0, 2),
        "total_rows": int(len(gold_df)),
        "feature_count": len(ALL_FEATURE_COLS),
        "feature_families": {k: len(v) for k, v in FEATURE_FAMILIES.items()},
        "eligible_watchlist_rows_3m": int(len(elig_3m)),
        "eligible_watchlist_rows_6m": int(len(elig_6m)),
        "distress_3m_event_rate_eligible": round(float(elig_3m["target_distress_3m"].mean()), 5),
        "distress_6m_event_rate_eligible": round(float(elig_6m["target_distress_6m"].mean()), 5),
        "senior_dd_6m_event_rate_eligible": round(float(elig_6m["target_senior_dd_6m"].mean()), 5),
        "credit_surge_6m_event_rate_eligible": round(float(elig_6m["target_credit_surge_6m"].mean()), 5),
        "fidc_np_structural_share": round(float(gold_df["is_fidc_np_structural"].mean()), 5),
        "shared_cedente_positive_degree_share": round(float((gold_df["contagion_shared_cedente_deg"] > 0).mean()), 5),
    }
    out_meta = REPORTS_DIR / "metrics" / "gold_summary.json"
    out_meta.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[Gold] Concluído em {summary['elapsed_seconds']}s!")
    for k, v in summary.items():
        if k != "feature_families":
            print(f"  - {k}: {v}")
    return summary

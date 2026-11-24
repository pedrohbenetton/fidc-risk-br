"""Fase 6: Estudo Forense (Recompras, SCR 2682, Contágio) e Backtest de Alocação vs. CDI."""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

from fidc_risk.config import GOLD_DIR, METRICS_DIR, ensure_directories


def _run_forensic_hypotheses_and_event_study(panel: pd.DataFrame) -> dict[str, object]:
    """
    Testa empiricamente três hipóteses forenses de crédito estruturado sobre fundos comerciais
    aparentemente saudáveis na largada (is_eligible_watchlist == 1 e inad_90p_pct < 3%).
    """
    low_inad = panel[
        (panel["is_eligible_watchlist"] == 1)
        & (panel["inad_90p_pct"] < 0.03)
        & (panel["target_distress_6m"].notna())
    ].copy()

    # --- Hipótese 1: Maquiagem via Recompras/Substituições em fundos com inadimplência aparente < 3% ---
    zero_mask = low_inad["recompra_masking_index"] <= 1e-5
    pos_vals = low_inad.loc[~zero_mask, "recompra_masking_index"]
    q75 = float(pos_vals.quantile(0.75)) if len(pos_vals) > 20 else 0.02

    grp_clean = low_inad[zero_mask]
    grp_masked = low_inad[low_inad["recompra_masking_index"] >= q75]

    rate_clean = float(grp_clean["target_distress_6m"].mean())
    rate_masked = float(grp_masked["target_distress_6m"].mean())
    dd_clean = float(grp_clean["target_senior_dd_6m"].mean())
    dd_masked = float(grp_masked["target_senior_dd_6m"].mean())

    tab_h1 = [
        [int(grp_masked["target_distress_6m"].sum()), int((1 - grp_masked["target_distress_6m"]).sum())],
        [int(grp_clean["target_distress_6m"].sum()), int((1 - grp_clean["target_distress_6m"]).sum())],
    ]
    odds_h1, pval_h1 = fisher_exact(tab_h1, alternative="greater")

    # --- Hipótese 2: Hiato de Provisão Regulatória SCR Resolução CMN/BCB 2.682 (pós-RCVM 175) ---
    scr_sub = panel[
        (panel["is_eligible_watchlist"] == 1)
        & (panel["has_scr_report"] == 1)
        & (panel["target_distress_6m"].notna())
    ].copy()
    scr_ok = scr_sub[scr_sub["scr_provision_gap_pct"] <= 0.0]
    scr_under = scr_sub[scr_sub["scr_provision_gap_pct"] > 0.02]
    rate_scr_ok = float(scr_ok["target_distress_6m"].mean()) if len(scr_ok) > 0 else 0.0
    rate_scr_under = float(scr_under["target_distress_6m"].mean()) if len(scr_under) > 0 else 0.0
    if len(scr_ok) > 10 and len(scr_under) > 10:
        tab_h2 = [
            [int(scr_under["target_distress_6m"].sum()), int((1 - scr_under["target_distress_6m"]).sum())],
            [int(scr_ok["target_distress_6m"].sum()), int((1 - scr_ok["target_distress_6m"]).sum())],
        ]
        odds_h2, pval_h2 = fisher_exact(tab_h2, alternative="greater")
    else:
        odds_h2, pval_h2 = 1.0, 1.0

    # --- Hipótese 3: Contágio via Cedentes Corporativos Compartilhados em fundos com inad < 3% ---
    no_cont = low_inad[
        (low_inad["contagion_shared_cedente_deg"] > 0)
        & (low_inad["contagion_cedente_stress_wt"] <= 1e-5)
        & (low_inad["contagion_cedente_max_inad90p"] < 0.02)
    ]
    hi_cont = low_inad[
        (low_inad["contagion_cedente_stress_wt"] >= 0.02) | (low_inad["contagion_cedente_max_inad90p"] >= 0.12)
    ]
    rate_no_cont = float(no_cont["target_distress_6m"].mean()) if len(no_cont) > 0 else 0.0
    rate_hi_cont = float(hi_cont["target_distress_6m"].mean()) if len(hi_cont) > 0 else 0.0
    if len(no_cont) > 10 and len(hi_cont) > 10:
        tab_h3 = [
            [int(hi_cont["target_distress_6m"].sum()), int((1 - hi_cont["target_distress_6m"]).sum())],
            [int(no_cont["target_distress_6m"].sum()), int((1 - no_cont["target_distress_6m"]).sum())],
        ]
        odds_h3, pval_h3 = fisher_exact(tab_h3, alternative="greater")
    else:
        odds_h3, pval_h3 = 1.0, 1.0

    # --- Event Study em Tempo de Evento (tau = -9 .. +3 meses ao redor do 1º choque agudo) ---
    com = panel[panel["is_fidc_np_structural"] == 0].sort_values(["entity_id", "dt_comptc"]).copy()
    com["ym_int"] = com["dt_comptc"].dt.year * 12 + com["dt_comptc"].dt.month

    # Encontrar o primeiro evento de estresse agudo precedido por >= 4 meses saudáveis
    g = com.groupby("entity_id", sort=False)
    prev_clean = (
        (g["acute_stress_now"].shift(1).fillna(0) == 0)
        & (g["acute_stress_now"].shift(2).fillna(0) == 0)
        & (g["acute_stress_now"].shift(3).fillna(0) == 0)
    )
    onset_candidates = com[(com["acute_stress_now"] == 1) & prev_clean & (com["vl_pl"] >= 5_000_000.0)]
    first_onsets = onset_candidates.groupby("entity_id", as_index=False)["ym_int"].min().rename(
        columns={"ym_int": "onset_ym"}
    )

    ev_panel = com.merge(first_onsets, on="entity_id", how="inner")
    ev_panel["tau"] = ev_panel["ym_int"] - ev_panel["onset_ym"]
    ev_sub = ev_panel[(ev_panel["tau"] >= -9) & (ev_panel["tau"] <= 3)]

    event_study_rows = []
    for tau_val, tgrp in ev_sub.groupby("tau", sort=True):
        event_study_rows.append(
            {
                "tau": int(tau_val),
                "n_funds": int(len(tgrp)),
                "inad_early_1_60_mean_pct": float(tgrp["inad_early_1_60_pct"].mean() * 100.0),
                "inad_90p_mean_pct": float(tgrp["inad_90p_pct"].mean() * 100.0),
                "recompra_subst_mean_pct": float(
                    (tgrp["recompra_carteira_pct"] + tgrp["substituicao_carteira_pct"]).mean() * 100.0
                ),
                "pdd_mean_pct": float(tgrp["pdd_carteira_pct"].mean() * 100.0),
                "subordinacao_mean_pct": float(tgrp["subordinacao_pct"].mean() * 100.0),
                "contagion_peer_inad90_mean_pct": float(tgrp["contagion_cedente_inad90p_wt"].mean() * 100.0),
                "senior_ret_mean_pct": float(tgrp["eff_senior_ret_mes"].dropna().mean())
                if tgrp["eff_senior_ret_mes"].notna().any()
                else 0.0,
            }
        )
    pd.DataFrame(event_study_rows).to_csv(METRICS_DIR / "forensic_event_study.csv", index=False)

    return {
        "h1_recompra_masking": {
            "n_clean": int(len(grp_clean)),
            "n_high_masking": int(len(grp_masked)),
            "threshold_masking_q75": round(q75, 5),
            "distress_6m_rate_clean": round(rate_clean, 5),
            "distress_6m_rate_high_masking": round(rate_masked, 5),
            "relative_risk_ratio": round(rate_masked / max(rate_clean, 1e-6), 3),
            "senior_dd_rate_clean": round(dd_clean, 5),
            "senior_dd_rate_high_masking": round(dd_masked, 5),
            "fisher_odds_ratio": round(float(odds_h1), 3),
            "fisher_p_value": float(pval_h1),
        },
        "h2_scr_provision_gap": {
            "n_adequate_pdd": int(len(scr_ok)),
            "n_underprovisioned_gt2pp": int(len(scr_under)),
            "distress_6m_rate_adequate": round(rate_scr_ok, 5),
            "distress_6m_rate_underprovisioned": round(rate_scr_under, 5),
            "relative_risk_ratio": round(rate_scr_under / max(rate_scr_ok, 1e-6), 3),
            "fisher_odds_ratio": round(float(odds_h2), 3),
            "fisher_p_value": float(pval_h2),
        },
        "h3_shared_cedente_contagion": {
            "n_healthy_peers": int(len(no_cont)),
            "n_distressed_peers": int(len(hi_cont)),
            "distress_6m_rate_healthy_peers": round(rate_no_cont, 5),
            "distress_6m_rate_distressed_peers": round(rate_hi_cont, 5),
            "relative_risk_ratio": round(rate_hi_cont / max(rate_no_cont, 1e-6), 3),
            "fisher_odds_ratio": round(float(odds_h3), 3),
            "fisher_p_value": float(pval_h3),
        },
        "event_study_onset_count": int(len(first_onsets)),
    }


def _compute_portfolio_summary_stats(
    monthly_df: pd.DataFrame,
    ret_col: str,
    cdi_col: str = "cdi_ret_pct",
) -> dict[str, float]:
    r = monthly_df[ret_col].values / 100.0
    c = monthly_df[cdi_col].values / 100.0
    excess = r - c

    cum_port = float(np.prod(1.0 + r) - 1.0)
    cum_cdi = float(np.prod(1.0 + c) - 1.0)
    n_m = max(len(r), 1)
    cagr = float((1.0 + cum_port) ** (12.0 / n_m) - 1.0)
    cdi_cagr = float((1.0 + cum_cdi) ** (12.0 / n_m) - 1.0)

    vol_ann = float(np.std(r, ddof=1) * np.sqrt(12.0)) if len(r) > 1 else 0.0
    ex_mean_ann = float(np.mean(excess) * 12.0)
    ex_std_ann = float(np.std(excess, ddof=1) * np.sqrt(12.0)) if len(r) > 1 else 1e-4
    sharpe = ex_mean_ann / max(ex_std_ann, 1e-3)

    downside = np.minimum(excess, 0.0)
    down_std_ann = float(np.sqrt(np.mean(downside**2)) * np.sqrt(12.0))
    sortino = ex_mean_ann / max(down_std_ann, 2.5e-3)

    wealth = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(wealth)
    dd = (wealth - peak) / peak
    max_dd = float(dd.min())

    q05 = float(np.quantile(r, 0.05))
    tail = r[r <= q05]
    cvar_95_monthly = float(tail.mean()) if len(tail) > 0 else q05

    return {
        "months": int(n_m),
        "cumulative_return_pct": round(cum_port * 100.0, 2),
        "cdi_cumulative_pct": round(cum_cdi * 100.0, 2),
        "pct_of_cdi": round((cum_port / max(cum_cdi, 1e-6)) * 100.0, 1),
        "cagr_aa_pct": round(cagr * 100.0, 2),
        "excess_over_cdi_aa_pp": round((cagr - cdi_cagr) * 100.0, 2),
        "volatility_aa_pct": round(vol_ann * 100.0, 2),
        "sharpe_ratio": round(sharpe, 3),
        "sortino_ratio": round(sortino, 3),
        "max_drawdown_pct": round(max_dd * 100.0, 2),
        "cvar_95_monthly_pct": round(cvar_95_monthly * 100.0, 2),
    }


def _select_cvar_va_weights(sub: pd.DataFrame, max_single_w: float = 0.04, max_sec_w: float = 0.25) -> pd.Series:
    """
    Alocação sob incerteza epistêmica (Venn-Abers + Limite Setorial):
    Maximiza carry esperado descontado pela perda esperada e pela largura de ambiguidade Venn-Abers,
    respeitando teto de 4% por fundo e 25% por setor dominante (o saldo não alocado rende 100% CDI).
    """
    carry_est = np.clip(sub["rentab_senior_3m_excess"].values, -0.5, 0.60)
    lgd_eff = np.clip(0.18 * (1.0 - sub["total_enhancement_pct"].values), 0.04, 0.18) * 100.0
    p_hyb = sub["prob_risk"].values
    va_w = sub["va_width_risk"].values

    # Utilidade ajustada ao risco mensal (% excesso líquido esperado em 6m convertido p/ mês)
    utility = carry_est - (lgd_eff * p_hyb / 6.0) - 0.85 * va_w
    candidates = sub.assign(_u=utility).query("_u > 0.03 and prob_risk < 0.11").sort_values("_u", ascending=False)

    weights = pd.Series(0.0, index=sub.index, dtype="float64")
    sec_used: dict[str, float] = {}
    tot_used = 0.0

    for idx, row in candidates.iterrows():
        if tot_used >= 0.999:
            break
        sec = str(row["dominant_sector"])
        avail_sec = max_sec_w - sec_used.get(sec, 0.0)
        if avail_sec <= 1e-4:
            continue
        w_i = min(max_single_w, avail_sec, 1.0 - tot_used)
        weights.loc[idx] = w_i
        sec_used[sec] = sec_used.get(sec, 0.0) + w_i
        tot_used += w_i

    return weights


def run_portfolio_backtest(panel: pd.DataFrame, oof_master: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Simula 6 políticas institucionais de investimento em cotas sênior de FIDCs abertos (2022–2026),
    comparando:
      - Execução com Defasagem Contratual Realista (publicação CVM + resgate -> r_{t+2}) vs
      - Execução Instantânea Sem Fricção (r_{t+1}, para expor o viés de liquidez comum em backtests!).
    """
    cols_panel = [
        "entity_id",
        "dt_comptc",
        "is_aberto",
        "is_exclusivo",
        "vl_pl",
        "pl_senior",
        "subordinacao_pct",
        "total_enhancement_pct",
        "inad_90p_pct",
        "cash_liquidity_pct",
        "redemption_lag_days",
        "rentab_senior_mes",
        "rentab_senior_3m_excess",
        "fwd_ret_senior_1m_step",
        "fwd_ret_senior_2m_step",
        "fwd_cdi_1m_step",
        "fwd_cdi_2m_step",
    ]
    prob_col = "prob_hybrid_gbdt_gnn" if "prob_hybrid_gbdt_gnn" in oof_master.columns else "prob_gbdt_ensemble_va"
    width_col = "hybrid_va_width" if "hybrid_va_width" in oof_master.columns else "va_width"

    avail_panel_cols = [c for c in cols_panel if c in panel.columns]
    df = oof_master[["entity_id", "dt_comptc", "dominant_sector", "y_true", prob_col, width_col]].merge(
        panel[avail_panel_cols],
        on=["entity_id", "dt_comptc"],
        how="inner",
    )
    df = df.rename(columns={prob_col: "prob_risk", width_col: "va_width_risk"})

    # Universo institucional investível (filtrado EXCLUSIVAMENTE com informação conhecida na data t):
    # - FIDC aberto, não exclusivo, cota sênior viva >= R$ 5M (exclui placeholders QT_COTA=0 da tab_X_2),
    # - Cotação mensal válida em t (exclui zeros de placeholder e erros de preenchimento de taxa anualizada >2,8% a.m.).
    if "rentab_senior_mes" in df.columns:
        has_live_sr_quote = df["rentab_senior_mes"].isna() | (
            (df["rentab_senior_mes"].abs() > 1e-4) & (df["rentab_senior_mes"] <= 2.8)
        )
    else:
        has_live_sr_quote = pd.Series(True, index=df.index)
    inv = df[
        (df["is_aberto"] == 1)
        & (df["is_exclusivo"] == 0)
        & (df["vl_pl"] >= 10_000_000.0)
        & (df["pl_senior"] >= 5_000_000.0)
        & (df["redemption_lag_days"] <= 90.0)
        & has_live_sr_quote
        & (df["rentab_senior_3m_excess"] <= 1.5)
        & (df["fwd_ret_senior_1m_step"].notna())
        & (df["fwd_ret_senior_2m_step"].notna())
    ].copy()

    # Sanitização assimétrica de retorno realizado: preserva 100% dos tombos negativos (-99% a 0%),
    # substitui placeholders exatos 0.0000 por CDI e trunca apenas erros de escala anualizada no ganho (> CDI + 1,8% no mês).
    for step_k in (1, 2):
        rc = f"fwd_ret_senior_{step_k}m_step"
        cc = f"fwd_cdi_{step_k}m_step"
        raw_r = inv[rc].values
        cdi_v = inv[cc].values
        no_zero_placeholder = np.where(np.abs(raw_r) <= 1e-5, cdi_v, raw_r)
        inv[rc] = np.minimum(no_zero_placeholder, cdi_v + 1.80)

    monthly_records = []
    for dt_val, mgrp in inv.groupby("dt_comptc", sort=True):
        if len(mgrp) < 15:
            continue
        cdi_1 = float(mgrp["fwd_cdi_1m_step"].median())
        cdi_2 = float(mgrp["fwd_cdi_2m_step"].median())

        r1 = mgrp["fwd_ret_senior_1m_step"].values
        r2 = mgrp["fwd_ret_senior_2m_step"].values

        # 1. Equal-Weight Mercado
        ew_r2 = float(np.mean(r2))
        ew_shocks = int((r2 <= -1.5).sum())

        # 2. Yield-Chasing (Top 25 maior retorno histórico recente sem filtro de risco)
        top_yc = mgrp.nlargest(min(25, len(mgrp)), "rentab_senior_3m_excess")
        yc_r1 = float(top_yc["fwd_ret_senior_1m_step"].mean())
        yc_r2 = float(top_yc["fwd_ret_senior_2m_step"].mean())
        yc_shocks = int((top_yc["fwd_ret_senior_2m_step"] <= -1.5).sum())

        # 3. Regra Tradicional de Comitê (Subordinação >= 15%, Inad >90d <= 3%, Liquidez >= 5%)
        trad = mgrp[
            (mgrp["total_enhancement_pct"] >= 0.15)
            & (mgrp["inad_90p_pct"] <= 0.03)
            & (mgrp["cash_liquidity_pct"] >= 0.05)
        ]
        if len(trad) >= 5:
            trad_top = trad.nlargest(min(25, len(trad)), "rentab_senior_3m_excess")
            tr_r2 = float(trad_top["fwd_ret_senior_2m_step"].mean())
            tr_shocks = int((trad_top["fwd_ret_senior_2m_step"] <= -1.5).sum())
        else:
            tr_r2, tr_shocks = cdi_2, 0

        # 4. Filtro Watchlist ML Cost-Sensitive (PD_6m < 10% nos candidatos de maior spread)
        ml_safe = mgrp[mgrp["prob_risk"] < 0.10]
        if len(ml_safe) >= 5:
            ml_top = ml_safe.nlargest(min(25, len(ml_safe)), "rentab_senior_3m_excess")
            ml_r2 = float(ml_top["fwd_ret_senior_2m_step"].mean())
            ml_shocks = int((ml_top["fwd_ret_senior_2m_step"] <= -1.5).sum())
        else:
            ml_r2, ml_shocks = cdi_2, 0

        # 5. Alocação CVaR / Venn-Abers Uncertainty-Penalized (c/ caixa residual em 100% CDI)
        w_va = _select_cvar_va_weights(mgrp).values
        cash_w = max(0.0, 1.0 - float(w_va.sum()))
        va_r1 = float(np.sum(w_va * r1) + cash_w * cdi_1)
        va_r2 = float(np.sum(w_va * r2) + cash_w * cdi_2)
        va_shocks = int(((w_va > 0.005) & (r2 <= -1.5)).sum())

        monthly_records.append(
            {
                "decision_dt": dt_val,
                "realization_dt": (dt_val + pd.offsets.MonthEnd(2)).normalize(),
                "n_investable": int(len(mgrp)),
                "cdi_ret_pct": cdi_2,
                "cdi_ret_t1_pct": cdi_1,
                "equal_weight_lagged": ew_r2,
                "yield_chasing_lagged": yc_r2,
                "traditional_committee_lagged": tr_r2,
                "ml_watchlist_lagged": ml_r2,
                "cvar_venn_abers_lagged": va_r2,
                "cvar_venn_abers_frictionless_t1": va_r1,
                "yield_chasing_frictionless_t1": yc_r1,
                "ew_shocks": ew_shocks,
                "yc_shocks": yc_shocks,
                "trad_shocks": tr_shocks,
                "ml_shocks": ml_shocks,
                "va_shocks": va_shocks,
                "va_cash_share": round(cash_w, 4),
            }
        )

    m_df = pd.DataFrame(monthly_records)
    policy_cols = [
        ("benchmark_cdi", "cdi_ret_pct", "cdi_ret_pct", 0),
        ("equal_weight_senior", "equal_weight_lagged", "cdi_ret_pct", int(m_df["ew_shocks"].sum())),
        ("yield_chasing_top25", "yield_chasing_lagged", "cdi_ret_pct", int(m_df["yc_shocks"].sum())),
        ("traditional_committee", "traditional_committee_lagged", "cdi_ret_pct", int(m_df["trad_shocks"].sum())),
        ("ml_watchlist_cutoff", "ml_watchlist_lagged", "cdi_ret_pct", int(m_df["ml_shocks"].sum())),
        ("cvar_venn_abers_lagged", "cvar_venn_abers_lagged", "cdi_ret_pct", int(m_df["va_shocks"].sum())),
        (
            "cvar_venn_abers_frictionless_t1",
            "cvar_venn_abers_frictionless_t1",
            "cdi_ret_t1_pct",
            int(m_df["va_shocks"].sum()),
        ),
    ]
    summary_rows = []
    for pname, rcol, ccol, shocks_cnt in policy_cols:
        st = _compute_portfolio_summary_stats(m_df, rcol, cdi_col=ccol)
        summary_rows.append({"policy": pname, "severe_monthly_shocks_suffered": shocks_cnt, **st})
    summary_df = pd.DataFrame(summary_rows)

    # Curva pré-computada de Fronteira de Decisão p/ o Simulador Interativo Web (limiar tau em [0.02 .. 0.60])
    thresholds = np.round(np.linspace(0.02, 0.55, 28), 3)
    y_all = oof_master["y_true"].values.astype("int32")
    p_all = oof_master[prob_col].values.astype("float64")
    frontier_rows = []
    for tau in thresholds:
        flagged = p_all >= tau
        tp = int(((flagged) & (y_all == 1)).sum())
        fp = int(((flagged) & (y_all == 0)).sum())
        fn = int(((~flagged) & (y_all == 1)).sum())
        tn = int(((~flagged) & (y_all == 0)).sum())

        # Desempenho de carteira correspondente a excluir prob_risk >= tau nos top 25 yielders
        sim_rets = []
        for _, mgrp in inv.groupby("dt_comptc", sort=True):
            if len(mgrp) < 15:
                continue
            cdi_2 = float(mgrp["fwd_cdi_2m_step"].median())
            sub_s = mgrp[mgrp["prob_risk"] < tau]
            if len(sub_s) >= 5:
                top_s = sub_s.nlargest(min(25, len(sub_s)), "rentab_senior_3m_excess")
                sim_rets.append(float(top_s["fwd_ret_senior_2m_step"].mean()))
            else:
                sim_rets.append(cdi_2)
        tmp_df = m_df[["cdi_ret_pct"]].copy()
        tmp_df["sim_ret"] = sim_rets
        sim_st = _compute_portfolio_summary_stats(tmp_df, "sim_ret", "cdi_ret_pct")

        frontier_rows.append(
            {
                "threshold": float(tau),
                "flagged_rate": round(float(flagged.mean()), 4),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "recall": round(tp / max(tp + fn, 1), 4),
                "precision": round(tp / max(tp + fp, 1), 4),
                **sim_st,
            }
        )
    frontier_df = pd.DataFrame(frontier_rows)
    return m_df, summary_df, frontier_df


def run_forensic_and_allocation_suite() -> dict[str, object]:
    ensure_directories()
    t0 = time.perf_counter()
    print("[Fase 6] Executando estudo forense (Recompras, SCR 2682, Contágio) e backtest de portfólio...")

    panel = pd.read_parquet(GOLD_DIR / "panel_gold.parquet")
    oof_path = GOLD_DIR / "oof_master_h6.parquet"
    if not oof_path.exists():
        oof_path = GOLD_DIR / "oof_tabular_h6.parquet"
    oof_master = pd.read_parquet(oof_path)

    forensic = _run_forensic_hypotheses_and_event_study(panel)
    monthly_bt, summary_bt, frontier_bt = run_portfolio_backtest(panel, oof_master)

    monthly_bt.to_csv(METRICS_DIR / "portfolio_monthly_returns.csv", index=False)
    summary_bt.to_csv(METRICS_DIR / "portfolio_summary_metrics.csv", index=False)
    frontier_bt.to_csv(METRICS_DIR / "decision_frontier_thresholds.csv", index=False)

    out_payload = {
        "elapsed_seconds": round(time.perf_counter() - t0, 2),
        "forensic_tests": forensic,
        "portfolio_policies": summary_bt.to_dict(orient="records"),
    }
    (METRICS_DIR / "forensic_and_allocation_summary.json").write_text(
        json.dumps(out_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[Fase 6] Concluída em {out_payload['elapsed_seconds']}s!")
    print(summary_bt[["policy", "cumulative_return_pct", "pct_of_cdi", "sharpe_ratio", "max_drawdown_pct", "severe_monthly_shocks_suffered"]].to_string(index=False))
    return out_payload

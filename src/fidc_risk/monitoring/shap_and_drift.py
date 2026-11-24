"""Fase 7A: Explicabilidade TreeSHAP Global/Local e Monitoramento de Drift por Regime (PSI & JS)."""

from __future__ import annotations

import json
import time

import joblib
import numpy as np
import pandas as pd
import shap

from fidc_risk.config import GOLD_DIR, METRICS_DIR, MODELS_DIR, ensure_directories
from fidc_risk.features.gold import FEATURE_FAMILIES
from fidc_risk.models.metrics import expected_calibration_error
from fidc_risk.models.rollrate_cqr import QUANTILE_ALPHAS, _delinquency_stratum

FEATURE_LABELS_PTBR: dict[str, str] = {
    "inad_90p_pct": "Inadimplência >90 dias (% Carteira)",
    "inad_early_1_60_pct": "Atraso Curto 1–60 dias (% Carteira)",
    "inad_early_1_90_pct": "Atraso Total 1–90 dias (% Carteira)",
    "inad_1_30_pct": "Atraso 1–30 dias (% Carteira)",
    "inad_31_60_pct": "Atraso 31–60 dias (% Carteira)",
    "inad_61_90_pct": "Atraso 61–90 dias (% Carteira)",
    "inad_severe_180p_pct": "Inadimplência Longa >180 dias (%)",
    "inad_total_pct": "Inadimplência Total Vencida (% Carteira)",
    "inad_90p_delta_3m": "Aceleração 3m da Inadimplência >90d",
    "inad_90p_delta_6m": "Aceleração 6m da Inadimplência >90d",
    "inad_90p_vol_6m": "Volatilidade 6m da Inadimplência",
    "recompra_masking_index": "Índice Forense de Maquiagem via Recompras",
    "recompra_carteira_pct": "Recompras no Mês (% Carteira)",
    "substituicao_carteira_pct": "Substituições no Mês (% Carteira)",
    "recompra_3m_avg_pct": "Giro Médio 3m de Recompras (% Carteira)",
    "substituicao_3m_avg_pct": "Giro Médio 3m de Substituições (%)",
    "pdd_carteira_pct": "Provisão PDD Total (% Carteira)",
    "pdd_coverage_90p": "Cobertura de PDD sobre Vencidos >90d",
    "unprovisioned_inad90p_pl": "Inadimplência >90d Descoberta (% PL)",
    "scr_provision_gap_pct": "Déficit de Provisão vs. Bacen 2.682 (%)",
    "scr_efgh_share": "Fatia SCR Risco Alto E–H (%)",
    "emp_recup_judicial_pct": "Créditos em Recuperação Judicial (%)",
    "acao_judicial_pct": "Créditos em Ação Judicial (%)",
    "cota_fidc_cross_pct": "Alocação Cruzada em Cotas de FIDCs (%)",
    "cota_fidc_np_pct": "Exposição a Cotas de FIDC-NP (%)",
    "subordinacao_pct": "Colchão de Cota Subordinada (% PL)",
    "total_enhancement_pct": "Reforço Total Subord. + Mezanino (% PL)",
    "net_cushion_over_bad_debt": "Margem Estrutural Líquida de Subordinação",
    "subordinacao_delta_3m": "Variação 3m da Subordinação (Queima)",
    "pl_log": "Porte do Fundo (log PL)",
    "pl_drawdown_12m_pct": "Queda Acumulada 12m do PL (%)",
    "rentab_senior_excess_cdi": "Retorno Cota Sênior Acima do CDI (p.p.)",
    "rentab_senior_3m_excess": "Excesso Médio 3m Cota Sênior vs. CDI",
    "rentab_subord_mes_clean": "Retorno Mensal da Cota Subordinada (%)",
    "cash_liquidity_pct": "Caixa + Títulos Públicos (% PL)",
    "resgate_solicitado_pl_pct": "Fila de Resgates Solicitados (% PL)",
    "net_flow_3m_pl_pct": "Captação Líquida 3m (% PL)",
    "alm_liq_30d_ratio": "Cobertura de Liquidez 30 dias (ALM)",
    "is_aberto": "Condomínio Aberto (Resgatável)",
    "is_exclusivo": "Fundo Exclusivo",
    "redemption_lag_days": "Prazo Contratual de Resgate (Dias)",
    "top1_cedente_pct": "Concentração no Maior Cedente (%)",
    "top3_cedente_pct": "Concentração nos Top-3 Cedentes (%)",
    "cedente_hhi": "Índice HHI de Concentração de Cedentes",
    "cedente_churn_1m": "Rotatividade Mensal de Cedentes (Jaccard)",
    "contagion_shared_cedente_deg": "Grau de Cedentes Compartilhados na Rede",
    "contagion_cedente_inad90p_wt": "Contágio: Inadimplência >90d em Pares de Cedente",
    "contagion_cedente_early_wt": "Contágio: Atraso 1–60d em Pares de Cedente",
    "contagion_cedente_stress_wt": "Contágio: Estresse Agudo em Pares de Cedente",
    "gestor_loo_inad90p": "Inadimplência Média de Fundos Irmãos (Gestor)",
    "gestor_loo_stress_rate": "Taxa de Estresse de Fundos Irmãos (Gestor)",
    "admin_loo_stress_rate": "Taxa de Estresse na Administradora (LOO)",
    "selic_meta_aa": "Meta Selic Copom (% a.a.)",
    "juro_real_expost_aa": "Juro Real Ex-Post (% a.a.)",
    "inad_sfn_pj_pct": "Inadimplência PJ Sistema Financeiro (%)",
    "fund_age_months": "Idade do Fundo (Meses)",
}


def human_feature_label(col: str) -> str:
    return FEATURE_LABELS_PTBR.get(col, col.replace("_", " ").title())


def compute_psi(reference: np.ndarray, current: np.ndarray, n_bins: int = 10) -> float:
    """Calcula o Population Stability Index (PSI) usando quantis da distribuição de referência."""
    ref = np.asarray(reference, dtype="float64")
    cur = np.asarray(current, dtype="float64")
    ref = ref[np.isfinite(ref)]
    cur = cur[np.isfinite(cur)]
    if len(ref) < 20 or len(cur) < 20:
        return 0.0

    edges = np.unique(np.quantile(ref, np.linspace(0.0, 1.0, n_bins + 1)))
    if len(edges) <= 2:
        # Variável quase binária/discreta
        p_r = np.clip(np.mean(ref > 0), 1e-4, 1.0 - 1e-4)
        p_c = np.clip(np.mean(cur > 0), 1e-4, 1.0 - 1e-4)
        return float((p_c - p_r) * np.log(p_c / p_r) + ((1 - p_c) - (1 - p_r)) * np.log((1 - p_c) / (1 - p_r)))

    edges[0] = -np.inf
    edges[-1] = np.inf
    r_cnt, _ = np.histogram(ref, bins=edges)
    c_cnt, _ = np.histogram(cur, bins=edges)

    r_p = np.clip(r_cnt / max(len(ref), 1), 1e-4, 1.0)
    c_p = np.clip(c_cnt / max(len(cur), 1), 1e-4, 1.0)
    return float(np.sum((c_p - r_p) * np.log(c_p / r_p)))


def compute_js_divergence(reference: np.ndarray, current: np.ndarray, n_bins: int = 12) -> float:
    """Calcula a Divergência de Jensen-Shannon simétrica em [0, log(2)]."""
    ref = np.asarray(reference, dtype="float64")
    cur = np.asarray(current, dtype="float64")
    lo = min(float(np.min(ref)), float(np.min(cur)))
    hi = max(float(np.max(ref)), float(np.max(cur)))
    if hi - lo < 1e-9:
        return 0.0
    edges = np.linspace(lo, hi, n_bins + 1)
    p, _ = np.histogram(ref, bins=edges)
    q, _ = np.histogram(cur, bins=edges)
    p = np.clip(p / max(p.sum(), 1), 1e-6, 1.0)
    q = np.clip(q / max(q.sum(), 1), 1e-6, 1.0)
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    return float(0.5 * np.sum(p * np.log(p / m)) + 0.5 * np.sum(q * np.log(q / m)))


def run_shap_and_drift_analysis() -> dict[str, object]:
    ensure_directories()
    t0 = time.perf_counter()
    print("[Fase 7A] Calculando explicações TreeSHAP globais/locais e monitoramento de Drift (PSI & JS)...")

    panel = pd.read_parquet(GOLD_DIR / "panel_gold.parquet")
    served_tab = joblib.load(MODELS_DIR / "served_tabular_h6.pkl")
    served_cqr = joblib.load(MODELS_DIR / "served_cqr_h3.pkl")

    lgbm_model = served_tab["lgbm_monotonic"]
    va_cal = served_tab["venn_abers"]
    feat_cols: list[str] = served_tab["feature_cols"]

    col_to_family = {c: fam for fam, cols in FEATURE_FAMILIES.items() for c in cols}

    # --- 1. Importância Global TreeSHAP em amostra estratificada out-of-time ---
    elig = panel[(panel["is_eligible_watchlist"] == 1) & (panel["dt_comptc"] >= pd.Timestamp("2022-01-01"))]
    sample_df = elig.sample(n=min(4000, len(elig)), random_state=42)
    explainer = shap.TreeExplainer(lgbm_model)
    shap_vals = explainer.shap_values(sample_df[feat_cols])
    if isinstance(shap_vals, list):
        shap_mat = np.asarray(shap_vals[1], dtype="float64")
    else:
        shap_mat = np.asarray(shap_vals, dtype="float64")

    mean_abs_shap = np.mean(np.abs(shap_mat), axis=0)
    shap_global_df = pd.DataFrame(
        {
            "feature": feat_cols,
            "label_ptbr": [human_feature_label(c) for c in feat_cols],
            "family": [col_to_family.get(c, "other") for c in feat_cols],
            "mean_abs_shap": mean_abs_shap,
        }
    ).sort_values("mean_abs_shap", ascending=False)
    shap_global_df.to_csv(METRICS_DIR / "shap_global_importance.csv", index=False)

    fam_shap_df = (
        shap_global_df.groupby("family", as_index=False)["mean_abs_shap"]
        .sum()
        .sort_values("mean_abs_shap", ascending=False)
    )
    fam_shap_df["share_pct"] = fam_shap_df["mean_abs_shap"] / max(fam_shap_df["mean_abs_shap"].sum(), 1e-9) * 100.0
    fam_shap_df.to_csv(METRICS_DIR / "shap_family_importance.csv", index=False)

    # --- 2. Escore ao Vivo na Competência Mais Recente (ex.: 2026-10-31) + SHAP Local + CQR Fan ---
    latest_dt = panel["dt_comptc"].max()
    live_sub = panel[
        (panel["dt_comptc"] == latest_dt)
        & (panel["is_fidc_np_structural"] == 0)
        & (panel["vl_pl"] >= 5_000_000.0)
        & (panel["vl_carteira"] >= 2_000_000.0)
    ].copy()

    raw_live = lgbm_model.predict_proba(live_sub[feat_cols])[:, 1]
    va_live = va_cal.predict_full(raw_live)
    live_sub["prob_distress_6m"] = va_live.prob
    live_sub["va_p0"] = va_live.p0
    live_sub["va_p1"] = va_live.p1
    live_sub["va_width"] = va_live.width

    qmodels_h3 = served_cqr["qmodels_h3"]
    cqr_h3 = served_cqr["cqr_h3"]
    q_live_raw = {a: qmodels_h3[a].predict(live_sub[feat_cols]) for a in QUANTILE_ALPHAS}
    q_live_cqr = cqr_h3.apply(q_live_raw, _delinquency_stratum(live_sub["inad_90p_pct"].values))
    live_sub["cqr_inad90_h3_q10"] = q_live_cqr[0.10]
    live_sub["cqr_inad90_h3_q50"] = q_live_cqr[0.50]
    live_sub["cqr_inad90_h3_q90"] = q_live_cqr[0.90]
    live_sub["cqr_inad90_h3_q95"] = q_live_cqr[0.95]

    live_shap_vals = explainer.shap_values(live_sub[feat_cols])
    live_shap_mat = np.asarray(live_shap_vals[1] if isinstance(live_shap_vals, list) else live_shap_vals, dtype="float64")

    # Extrair os 5 principais fatores de risco (+) e 3 principais amortecedores (-) por fundo ao vivo
    top_drivers_json = []
    for row_i in range(len(live_sub)):
        sv = live_shap_mat[row_i]
        pos_idx = np.argsort(-sv)[:4]
        neg_idx = np.argsort(sv)[:2]
        drivers = []
        for fi in list(pos_idx) + list(neg_idx):
            val_s = float(sv[fi])
            if abs(val_s) < 1e-4:
                continue
            cname = feat_cols[fi]
            drivers.append(
                {
                    "feature": cname,
                    "label": human_feature_label(cname),
                    "shap_log_odds": round(val_s, 4),
                    "feature_value": round(float(live_sub[cname].iloc[row_i]), 4),
                }
            )
        top_drivers_json.append(json.dumps(drivers, ensure_ascii=False))
    live_sub["shap_drivers_json"] = top_drivers_json
    live_sub.to_parquet(GOLD_DIR / "live_watchlist_scored.parquet", index=False)

    # --- 3. Monitoramento Semestral de Drift (PSI & JS) e Saúde de Calibração ---
    ref_df = panel[(panel["is_eligible_watchlist"] == 1) & (panel["dt_comptc"] < pd.Timestamp("2022-01-01"))]
    monitored_cols = shap_global_df["feature"].head(12).tolist()

    panel_post = elig.copy()
    panel_post["semester"] = (
        panel_post["dt_comptc"].dt.year.astype(str)
        + "H"
        + np.where(panel_post["dt_comptc"].dt.month <= 6, "1", "2")
    )

    drift_rows = []
    for sem, sgrp in panel_post.groupby("semester", sort=True):
        for col in monitored_cols:
            psi_v = compute_psi(ref_df[col].values, sgrp[col].values)
            js_v = compute_js_divergence(ref_df[col].values, sgrp[col].values)
            drift_rows.append(
                {
                    "semester": str(sem),
                    "feature": col,
                    "label_ptbr": human_feature_label(col),
                    "psi": round(psi_v, 4),
                    "js_divergence": round(js_v, 4),
                    "status": "CRITICO" if psi_v >= 0.25 else ("MODERADO" if psi_v >= 0.10 else "ESTAVEL"),
                }
            )
    drift_df = pd.DataFrame(drift_rows)
    drift_df.to_csv(METRICS_DIR / "covariate_drift_psi.csv", index=False)

    # Evolução trimestral de Brier e ECE out-of-time
    oof_path = GOLD_DIR / "oof_master_h6.parquet"
    if not oof_path.exists():
        oof_path = GOLD_DIR / "oof_tabular_h6.parquet"
    oof = pd.read_parquet(oof_path)
    pcol = "prob_hybrid_gbdt_gnn" if "prob_hybrid_gbdt_gnn" in oof.columns else "prob_gbdt_ensemble_va"
    oof["quarter"] = oof["dt_comptc"].dt.to_period("Q").astype(str)
    cal_health_rows = []
    for qtr, qgrp in oof.groupby("quarter", sort=True):
        if len(qgrp) < 100:
            continue
        yt = qgrp["y_true"].values
        yp = qgrp[pcol].values
        cal_health_rows.append(
            {
                "quarter": str(qtr),
                "n_obs": int(len(yt)),
                "realized_event_rate": round(float(yt.mean()), 4),
                "mean_predicted_prob": round(float(yp.mean()), 4),
                "brier": round(float(np.mean((yt - yp) ** 2)), 4),
                "ece": round(expected_calibration_error(yt, yp, n_bins=8), 4),
            }
        )
    pd.DataFrame(cal_health_rows).to_csv(METRICS_DIR / "rolling_calibration_health.csv", index=False)

    summary = {
        "elapsed_seconds": round(time.perf_counter() - t0, 2),
        "latest_competence": str(latest_dt.date()),
        "live_scored_funds": int(len(live_sub)),
        "top5_shap_features": shap_global_df[["feature", "label_ptbr", "mean_abs_shap"]].head(5).to_dict(orient="records"),
    }
    (METRICS_DIR / "shap_and_drift_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[Fase 7A] Concluída em {summary['elapsed_seconds']}s! {len(live_sub)} fundos pontuados em {latest_dt.date()}.")
    return summary

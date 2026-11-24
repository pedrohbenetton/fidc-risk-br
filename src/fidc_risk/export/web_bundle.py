"""Exportador do pacote JSON estático verificado para o Observatório Interativo Web + Figuras."""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fidc_risk.config import (
    DEMO_CODES_PATH,
    FIGURES_DIR,
    GOLD_DIR,
    METRICS_DIR,
    PRIVATE_BUNDLE_PATH,
    SILVER_DIR,
    WEB_DATA_DIR,
    ensure_directories,
)
from fidc_risk.export.anonymize import CodeBook, anonymize_bundle
from fidc_risk.export.demo import LeakError, find_leaks, real_names, write_json

PUBLIC_BUNDLE_PATH = WEB_DATA_DIR / "observatory_bundle.json"
_LEGACY_PRIVATE_DIR = WEB_DATA_DIR / "private"  # local antigo do pacote nominal, dentro de web/


def _format_cnpj(digits: str) -> str:
    if len(digits) == 14:
        return f"{digits[:2]}.{digits[2:5]}.{digits[5:8]}/{digits[8:12]}-{digits[12:]}"
    return digits


def _mask_cpf_if_needed(doc: str, is_pj: int) -> str:
    """Exibe CNPJ corporativo formatado (público CVM) e mascara parcialmente CPF de pessoa física."""
    if is_pj == 1 and len(doc) == 14:
        return _format_cnpj(doc)
    if len(doc) == 11:
        return f"***.{doc[3:6]}.{doc[6:9]}-**"
    return doc


def _build_contagion_network_payload(
    live_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    latest_dt: pd.Timestamp,
) -> dict[str, list[dict[str, Any]]]:
    """
    Extrai o subgrafo real de contágio corporativo na competência ativa:
    Cedentes PJ compartilhados por >= 2 FIDCs comerciais, ponderados por exposição e risco.
    """
    e_cur = edges_df[(edges_df["dt_comptc"] == latest_dt) & (edges_df["is_pj"] == 1)].copy()
    live_lookup = live_df.set_index("entity_id")
    e_cur = e_cur[e_cur["entity_id"].isin(live_lookup.index)].copy()

    if len(e_cur) == 0:
        return {"nodes": [], "links": []}

    e_cur["fund_pl"] = e_cur["entity_id"].map(live_lookup["vl_pl"])
    e_cur["fund_pd"] = e_cur["entity_id"].map(live_lookup["prob_distress_6m"])
    e_cur["exp_brl"] = e_cur["fund_pl"] * (e_cur["share_pct"] / 100.0)

    ced_stats = (
        e_cur.groupby("cedente_doc", as_index=False)
        .agg(
            deg=("entity_id", "nunique"),
            tot_exp_brl=("exp_brl", "sum"),
            max_pd=("fund_pd", "max"),
            mean_pd=("fund_pd", "mean"),
        )
        .query("deg >= 2")
    )
    # Priorizar hubs com maior risco sistêmico combinado (grau * exposição * risco)
    ced_stats["priority"] = ced_stats["deg"] * np.log1p(ced_stats["tot_exp_brl"] / 1e6) * (0.3 + ced_stats["max_pd"])
    top_ceds = set(ced_stats.nlargest(65, "priority")["cedente_doc"].tolist())

    sub_edges = e_cur[e_cur["cedente_doc"].isin(top_ceds)].copy()
    # Se um cedente tiver dezenas de micro-fundos, mantemos as 8 maiores exposições p/ legibilidade visual
    sub_edges = (
        sub_edges.sort_values(["cedente_doc", "exp_brl"], ascending=[True, False])
        .groupby("cedente_doc")
        .head(8)
        .reset_index(drop=True)
    )

    nodes = []
    seen_funds: set[str] = set()
    ced_meta = ced_stats.set_index("cedente_doc")

    for cdoc in sorted(set(sub_edges["cedente_doc"])):
        cm = ced_meta.loc[cdoc]
        nodes.append(
            {
                "id": f"C_{cdoc}",
                "kind": "cedente",
                "label": f"Cedente {_mask_cpf_if_needed(cdoc, 1)}",
                "degree": int(cm["deg"]),
                "exposure_m": round(float(cm["tot_exp_brl"]) / 1e6, 2),
                "risk": round(float(cm["max_pd"]), 4),
            }
        )

    links = []
    for row in sub_edges.itertuples(index=False):
        eid = str(row.entity_id)
        if eid not in seen_funds:
            seen_funds.add(eid)
            fr = live_lookup.loc[eid]
            short_name = str(fr["denom_social"])[:42]
            nodes.append(
                {
                    "id": f"F_{eid}",
                    "kind": "fidc",
                    "entity_id": eid,
                    "cnpj": _format_cnpj(eid),
                    "label": short_name,
                    "full_name": str(fr["denom_social"]),
                    "sector": str(fr["dominant_sector"]),
                    "gestor": str(fr.get("nome_gestor", ""))[:36],
                    "pl_m": round(float(fr["vl_pl"]) / 1e6, 2),
                    "risk": round(float(fr["prob_distress_6m"]), 4),
                    "inad90": round(float(fr["inad_90p_pct"]) * 100.0, 2),
                    "subord": round(float(fr["subordinacao_pct"]) * 100.0, 2),
                }
            )
        links.append(
            {
                "source": f"C_{row.cedente_doc}",
                "target": f"F_{eid}",
                "share_pct": round(float(row.share_pct), 2),
                "exp_m": round(float(row.exp_brl) / 1e6, 2),
            }
        )

    return {"nodes": nodes, "links": links}


def _build_reliability_curve(oof: pd.DataFrame) -> list[dict[str, Any]]:
    y = oof["y_true"].values
    out: list[dict[str, Any]] = []
    specs = [
        ("lgbm_raw", "LightGBM Bruto", "prob_lgbm_unc_raw"),
        ("platt", "Platt Scaling", "prob_lgbm_unc_platt"),
        ("isotonic", "Isotônica (PAV)", "prob_lgbm_unc_iso"),
        ("venn_abers", "Ensemble + Venn-Abers", "prob_gbdt_ensemble_va"),
    ]
    if "prob_hybrid_gbdt_gnn" in oof.columns:
        specs.append(("hybrid_stack", "Stack Híbrido GBDT + GNN", "prob_hybrid_gbdt_gnn"))

    for key, label, col in specs:
        p = oof[col].values
        edges = np.quantile(p, np.linspace(0.0, 1.0, 11))
        pts = []
        for b in range(10):
            m = (p >= edges[b]) & (p <= edges[b + 1] if b == 9 else p < edges[b + 1])
            if m.sum() > 10:
                pts.append(
                    {
                        "bin": b + 1,
                        "pred": round(float(p[m].mean()), 4),
                        "obs": round(float(y[m].mean()), 4),
                        "n": int(m.sum()),
                    }
                )
        out.append({"key": key, "label": label, "points": pts})
    return out


def watchlist_item(r: Any, history: list[dict[str, object]]) -> dict[str, Any]:
    """Um fundo da carteira em observação (pacote nominal; a versão pública passa por anonymize_bundle)."""
    eid_s = str(r.entity_id)
    return {
        "entity_id": eid_s,
        "cnpj": _format_cnpj(eid_s),
        "name": str(r.denom_social),
        "sector": str(r.dominant_sector),
        "gestor": str(r.nome_gestor) if r.nome_gestor else "Não informado",
        "is_aberto": int(r.is_aberto),
        "pl_m": round(float(r.vl_pl) / 1e6, 2),
        "prob_6m": round(float(r.prob_distress_6m), 4),
        "va_p0": round(float(r.va_p0), 4),
        "va_p1": round(float(r.va_p1), 4),
        "va_width": round(float(r.va_width), 4),
        "inad90_pct": round(float(r.inad_90p_pct) * 100.0, 2),
        "late_61_90_pct": round(float(r.inad_61_90_pct) * 100.0, 2),
        "early_pct": round(float(r.inad_early_1_60_pct) * 100.0, 2),
        "pdd_pct": round(float(r.pdd_carteira_pct) * 100.0, 2),
        "subord_pct": round(float(r.subordinacao_pct) * 100.0, 2),
        "recompra_idx": round(float(r.recompra_masking_index) * 100.0, 2),
        "scr_gap_pct": round(float(r.scr_provision_gap_pct) * 100.0, 2),
        "contagion_pct": round(float(r.contagion_cedente_inad90p_wt) * 100.0, 2),
        "cqr_h3_q10": round(float(r.cqr_inad90_h3_q10) * 100.0, 2),
        "cqr_h3_q50": round(float(r.cqr_inad90_h3_q50) * 100.0, 2),
        "cqr_h3_q90": round(float(r.cqr_inad90_h3_q90) * 100.0, 2),
        "shap": json.loads(r.shap_drivers_json),
        "history": history,
    }


def build_nominal_bundle(figure_names: list[str] | None = None) -> dict[str, Any]:
    """Monta o pacote completo (com nomes e CNPJs) a partir dos artefatos Gold/Silver e das métricas."""
    if figure_names is None:
        figure_names = sorted(p.name for p in FIGURES_DIR.glob("fig*.png"))
    panel = pd.read_parquet(GOLD_DIR / "panel_gold.parquet")
    edges_df = pd.read_parquet(SILVER_DIR / "cedente_edges.parquet")
    live_df = pd.read_parquet(GOLD_DIR / "live_watchlist_scored.parquet")

    oof_path = GOLD_DIR / "oof_master_h6.parquet"
    if not oof_path.exists():
        oof_path = GOLD_DIR / "oof_tabular_h6.parquet"
    oof = pd.read_parquet(oof_path)

    silver_audit = json.loads((METRICS_DIR / "silver_quality_audit.json").read_text(encoding="utf-8"))
    gold_summary = json.loads((METRICS_DIR / "gold_summary.json").read_text(encoding="utf-8"))
    tab_stats = json.loads((METRICS_DIR / "tabular_statistical_tests.json").read_text(encoding="utf-8"))
    forensic_alloc = json.loads((METRICS_DIR / "forensic_and_allocation_summary.json").read_text(encoding="utf-8"))
    deep_summary = (
        json.loads((METRICS_DIR / "deep_contagion_summary.json").read_text(encoding="utf-8"))
        if (METRICS_DIR / "deep_contagion_summary.json").exists()
        else {}
    )

    # Série histórica mensal da indústria (106 meses)
    com = panel[panel["is_fidc_np_structural"] == 0].copy()
    ind_rows = []
    for dt_v, igrp in com.groupby("dt_comptc", sort=True):
        w = np.maximum(igrp["vl_carteira"].values, 1.0)
        ind_rows.append(
            {
                "dt": str(dt_v.date()),
                "n_funds": int(igrp["entity_id"].nunique()),
                "pl_bi": round(float(igrp["vl_pl"].sum()) / 1e9, 2),
                "inad90_pct": round(float(np.average(igrp["inad_90p_pct"].values, weights=w)) * 100.0, 2),
                "inad_early_pct": round(float(np.average(igrp["inad_early_1_60_pct"].values, weights=w)) * 100.0, 2),
                "recompra_pct": round(float(np.average(igrp["recompra_carteira_pct"].values, weights=w)) * 100.0, 2),
                "selic_aa": round(float(igrp["selic_meta_aa"].median()), 2),
                "cdi_mes": round(float(igrp["cdi_mensal_pct"].median()), 3),
            }
        )

    # Seleção de 350 fundos destacados p/ busca e inspeção rica no frontend (Top Risco + Top PL)
    top_risk_ids = set(live_df.nlargest(210, "prob_distress_6m")["entity_id"].tolist())
    top_aum_ids = set(live_df.nlargest(180, "vl_pl")["entity_id"].tolist())
    featured_ids = top_risk_ids | top_aum_ids
    feat_live = live_df[live_df["entity_id"].isin(featured_ids)].sort_values("prob_distress_6m", ascending=False)

    # Histórico recente (últimos 18 meses) apenas p/ os fundos destacados
    min_hist_dt = panel["dt_comptc"].max() - pd.DateOffset(months=18)
    hist_sub = panel[(panel["entity_id"].isin(featured_ids)) & (panel["dt_comptc"] >= min_hist_dt)].sort_values(
        ["entity_id", "dt_comptc"]
    )
    hist_by_eid: dict[str, list[dict[str, object]]] = {}
    for eid_k, hgrp in hist_sub.groupby("entity_id", sort=False):
        hist_by_eid[str(eid_k)] = [
            {
                "dt": str(r.dt_comptc.date())[:7],
                "inad90": round(float(r.inad_90p_pct) * 100.0, 2),
                "early": round(float(r.inad_early_1_60_pct) * 100.0, 2),
                "pdd": round(float(r.pdd_carteira_pct) * 100.0, 2),
                "subord": round(float(r.subordinacao_pct) * 100.0, 2),
                "pl_m": round(float(r.vl_pl) / 1e6, 1),
            }
            for r in hgrp.itertuples(index=False)
        ]

    watchlist_items = [
        watchlist_item(r, hist_by_eid.get(str(r.entity_id), [])) for r in feat_live.itertuples(index=False)
    ]

    latest_dt = pd.to_datetime(live_df["dt_comptc"].max())
    contagion_graph = _build_contagion_network_payload(live_df, edges_df, latest_dt)

    # Tabelas consolidadas de benchmark
    tab_pooled = pd.read_csv(METRICS_DIR / "tabular_pooled_metrics.csv")
    benchmarks_h6 = tab_pooled[tab_pooled["horizon_months"] == 6].to_dict(orient="records")
    benchmarks_h3 = tab_pooled[tab_pooled["horizon_months"] == 3].to_dict(orient="records")

    if "pooled_comparison" in deep_summary:
        existing_m = {d["model"] for d in benchmarks_h6}
        for mkey, mvals in deep_summary["pooled_comparison"].items():
            if mkey not in existing_m:
                benchmarks_h6.append({"horizon_months": 6, "model": mkey, **mvals})

    folds_df = pd.read_csv(METRICS_DIR / "tabular_folds_metrics.csv")
    folds_h6 = folds_df[folds_df["horizon_months"] == 6].to_dict(orient="records")
    if (METRICS_DIR / "deep_folds_metrics.csv").exists():
        deep_f = pd.read_csv(METRICS_DIR / "deep_folds_metrics.csv").to_dict(orient="records")
        folds_h6.extend(deep_f)

    bundle = {
        "meta": {
            "project": "fidc-risk-br",
            "latest_competence": str(latest_dt.date()),
            "min_competence": silver_audit["min_competence"],
            "months_processed": silver_audit["months_processed"],
            "generated_figures": figure_names,
        },
        "kpis": {
            "fund_month_rows": silver_audit["fund_month_rows"],
            "unique_entities": silver_audit["unique_entities"],
            "unique_gestores": silver_audit["unique_gestores"],
            "cedente_edge_rows": silver_audit["cedente_edge_rows"],
            "unique_pj_cedentes": silver_audit["unique_pj_cedentes"],
            "gestor_coverage_rate": silver_audit["gestor_coverage_rate"],
            "aging_invariant_rate": silver_audit["aging_bucket_sum_invariant_pass_rate"],
            "feature_count": gold_summary["feature_count"],
            "live_commercial_funds": int(len(live_df)),
            "live_total_pl_bi": round(float(live_df["vl_pl"].sum()) / 1e9, 2),
        },
        "industry_series": ind_rows,
        "watchlist_live": watchlist_items,
        "contagion_graph": contagion_graph,
        "benchmarks_h6": benchmarks_h6,
        "benchmarks_h3": benchmarks_h3,
        "folds_h6": folds_h6,
        "ablation_h6": pd.read_csv(METRICS_DIR / "tabular_ablation_h6.csv").to_dict(orient="records"),
        "calibration_curves": _build_reliability_curve(oof),
        "rollrate_cqr": pd.read_csv(METRICS_DIR / "rollrate_cqr_pooled_metrics.csv").to_dict(orient="records"),
        "markov_matrix": pd.read_csv(METRICS_DIR / "markov_global_transition_matrix.csv", index_col=0).to_dict(
            orient="split"
        ),
        "deep_summary": deep_summary,
        "statistical_tests": tab_stats["bootstrap_and_sensitivity"],
        "forensic": forensic_alloc["forensic_tests"],
        "event_study": pd.read_csv(METRICS_DIR / "forensic_event_study.csv").to_dict(orient="records"),
        "portfolio_summary": forensic_alloc["portfolio_policies"],
        "portfolio_monthly": pd.read_csv(METRICS_DIR / "portfolio_monthly_returns.csv").to_dict(orient="records"),
        "decision_frontier": pd.read_csv(METRICS_DIR / "decision_frontier_thresholds.csv").to_dict(orient="records"),
        "shap_global": pd.read_csv(METRICS_DIR / "shap_global_importance.csv").head(20).to_dict(orient="records"),
        "shap_families": pd.read_csv(METRICS_DIR / "shap_family_importance.csv").to_dict(orient="records"),
        "drift_psi": pd.read_csv(METRICS_DIR / "covariate_drift_psi.csv").to_dict(orient="records"),
        "rolling_calibration": pd.read_csv(METRICS_DIR / "rolling_calibration_health.csv").to_dict(orient="records"),
    }
    return bundle


def write_bundles(
    bundle: dict[str, Any],
    codebook: CodeBook,
    private_path: Path = PRIVATE_BUNDLE_PATH,
    public_path: Path = PUBLIC_BUNDLE_PATH,
) -> dict[str, Any]:
    """Grava o pacote nominal (data/private/, fora do git) e o público anonimizado (web/data/)."""
    write_json(private_path, bundle)
    public = anonymize_bundle(bundle, codebook)
    write_json(public_path, public)
    leaks = [f for f in find_leaks(public_path.parent, real_names(bundle)) if Path(f.path) == public_path]
    if leaks:
        public_path.unlink()
        raise LeakError(f"pacote público com dados identificáveis: {leaks[:5]}")
    # O pacote nominal ficava em web/data/private/, que seria publicado junto com web/.
    if _LEGACY_PRIVATE_DIR.exists():
        shutil.rmtree(_LEGACY_PRIVATE_DIR)
        print(f"[Export] Removido pacote nominal antigo de {_LEGACY_PRIVATE_DIR} (agora em {private_path.parent}).")
    return public


def export_all() -> dict[str, object]:
    """Executa Fase 7A (SHAP & Drift), gera as figuras e exporta os pacotes nominal e público."""
    ensure_directories()
    t0 = time.perf_counter()

    from fidc_risk.export.figures import generate_all_figures
    from fidc_risk.monitoring.shap_and_drift import run_shap_and_drift_analysis

    run_shap_and_drift_analysis()
    figure_names = generate_all_figures()

    print("[Export] Construindo pacote estático verificado para o Observatório Web...")
    bundle = build_nominal_bundle(figure_names)
    codebook = CodeBook.load(DEMO_CODES_PATH)
    write_bundles(bundle, codebook)
    codebook.save(DEMO_CODES_PATH)

    size_mb = PUBLIC_BUNDLE_PATH.stat().st_size / (1024 * 1024)
    print(
        f"[Export] Pacote web exportado em {time.perf_counter() - t0:.2f}s -> "
        f"{PUBLIC_BUNDLE_PATH} ({size_mb:.2f} MB, {len(bundle['watchlist_live'])} fundos interativos, "
        f"{len(bundle['contagion_graph']['nodes'])} nós na rede de contágio)."
    )
    return {"bundle_mb": round(size_mb, 2), "figures": figure_names}


def export_demo(out_dir: Path, rebuild_bundle: bool = False) -> Path:
    """Gera o site demo anonimizado em ``out_dir`` a partir do pacote nominal local."""
    from fidc_risk.export.demo import build_demo_site

    ensure_directories()
    if rebuild_bundle or not PRIVATE_BUNDLE_PATH.exists():
        legacy = _LEGACY_PRIVATE_DIR / "observatory_bundle.json"
        if not rebuild_bundle and legacy.exists():
            bundle = json.loads(legacy.read_text(encoding="utf-8"))
        else:
            print("[Demo] Montando pacote nominal a partir dos artefatos Gold/Silver e métricas...")
            bundle = build_nominal_bundle()
    else:
        bundle = json.loads(PRIVATE_BUNDLE_PATH.read_text(encoding="utf-8"))

    codebook = CodeBook.load(DEMO_CODES_PATH)
    write_bundles(bundle, codebook)
    site = build_demo_site(bundle, codebook, out_dir)
    codebook.save(DEMO_CODES_PATH)
    print(f"[Demo] Site anonimizado em {site} (códigos estáveis em {DEMO_CODES_PATH}).")
    return site

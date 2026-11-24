"""Testes unitários adicionais cobrindo ingestão, suíte tabular, forense, snapshots GNN e exportação."""

from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from fidc_risk.allocation.portfolio import _run_forensic_hypotheses_and_event_study
from fidc_risk.config import ensure_directories
from fidc_risk.export.web_bundle import (
    _build_contagion_network_payload,
    _build_reliability_curve,
)
from fidc_risk.features.gold import ALL_FEATURE_COLS, compute_gold_features
from fidc_risk.ingestion.run import _is_valid_zip, _sha256_file
from fidc_risk.models.deep_contagion import (
    _standardize_snapshots_for_fold,
    _train_single_gpu_run,
    build_monthly_graph_snapshots,
)
from fidc_risk.models.tabular_suite import (
    _committee_heuristic_raw,
    _fit_lgbm,
    _fit_logistic_hazard,
)
from tests.test_anti_leakage import _make_synthetic_silver_panel


def test_sha256_and_zip_validators(tmp_path: Path) -> None:
    ensure_directories()
    bad_file = tmp_path / "corrupt.zip"
    bad_file.write_bytes(b"not a zip file at all" * 20)
    assert _is_valid_zip(bad_file) is False
    assert len(_sha256_file(bad_file)) == 64

    good_zip = tmp_path / "valid.zip"
    with zipfile.ZipFile(good_zip, "w") as zf:
        zf.writestr("sample.csv", "a;b\n1;2\n" * 40)
    assert _is_valid_zip(good_zip) is True


def test_tabular_helpers_heuristic_logistic_and_lgbm() -> None:
    core, edges, macro = _make_synthetic_silver_panel()
    gold = compute_gold_features(core, edges, macro, include_targets=True)
    # Criar alvo sintético balanceado p/ testar ajuste rápido
    rng = np.random.default_rng(42)
    y = rng.binomial(1, 0.35, size=len(gold)).astype("int32")

    h_scores = _committee_heuristic_raw(gold)
    assert h_scores.shape == (len(gold),)
    assert np.all((h_scores > 0.0) & (h_scores < 1.0))

    scaler, lr, va = _fit_logistic_hazard(
        gold[ALL_FEATURE_COLS].values[:20],
        y[:20],
        gold[ALL_FEATURE_COLS].values[20:],
        y[20:],
    )
    preds_lr = va.predict(lr.predict_proba(scaler.transform(gold[ALL_FEATURE_COLS].values))[:, 1])
    assert np.all((preds_lr > 0.0) & (preds_lr < 1.0))

    m_lgb = _fit_lgbm(gold.iloc[:20], y[:20], gold.iloc[20:], y[20:], ALL_FEATURE_COLS[:15], monotonic=True)
    p_lgb = np.asarray(m_lgb.predict_proba(gold[ALL_FEATURE_COLS[:15]]), dtype="float64")[:, 1]
    assert p_lgb.shape == (len(gold),)


def test_forensic_hypotheses_and_event_study_on_synthetic_gold() -> None:
    core, edges, macro = _make_synthetic_silver_panel()
    gold = compute_gold_features(core, edges, macro, include_targets=True)
    gold["is_eligible_watchlist"] = 1
    gold["inad_90p_pct"] = 0.01
    gold["target_distress_6m"] = np.where(gold.index % 3 == 0, 1.0, 0.0)
    gold["target_senior_dd_6m"] = np.where(gold.index % 5 == 0, 1.0, 0.0)
    gold.loc[gold.index % 4 == 0, "recompra_masking_index"] = 0.0
    gold.loc[gold.index % 4 != 0, "recompra_masking_index"] = 0.15
    gold.loc[gold.index == 6, "acute_stress_now"] = 1

    res = _run_forensic_hypotheses_and_event_study(gold)
    assert "h1_recompra_masking" in res
    assert "h2_scr_provision_gap" in res
    assert "h3_shared_cedente_contagion" in res


def test_build_snapshots_and_single_mini_neural_run() -> None:
    core, edges, macro = _make_synthetic_silver_panel()
    gold = compute_gold_features(core, edges, macro, include_targets=True)
    gold["target_distress_6m"] = (gold.index % 2).astype("float32")
    gold["target_distress_3m"] = (gold.index % 3 == 0).astype("float32")
    gold["target_inad90p_h3"] = 0.03

    snaps, sec_map = build_monthly_graph_snapshots(gold, edges, device=torch.device("cpu"))
    assert len(snaps) == 10
    assert len(sec_map) >= 1

    n = len(gold)
    fit_m = np.zeros(n, dtype=bool)
    cal_m = np.zeros(n, dtype=bool)
    te_m = np.zeros(n, dtype=bool)
    fit_m[:18] = True
    cal_m[18:24] = True
    te_m[24:] = True

    mean_t, std_t = _standardize_snapshots_for_fold(snaps, fit_m)
    assert torch.isfinite(mean_t).all() and torch.isfinite(std_t).all()

    c_sc, t_sc, st_dict = _train_single_gpu_run(
        snaps,
        fit_m,
        cal_m,
        te_m,
        n_sectors=len(sec_map),
        use_graph=True,
        seed=11,
        device=torch.device("cpu"),
        max_epochs=2,
    )
    assert c_sc.shape == (6,)
    assert t_sc.shape == (6,)
    assert np.all(np.isfinite(c_sc)) and np.all(np.isfinite(t_sc))
    assert len(st_dict) > 0


def test_web_bundle_builders_contagion_and_reliability() -> None:
    dt = pd.Timestamp("2026-10-31")
    live = pd.DataFrame(
        {
            "entity_id": ["11111111000191", "22222222000192"],
            "denom_social": ["FUNDO A FIDC", "FUNDO B FIDC"],
            "dominant_sector": ["Varejo", "Industrial"],
            "nome_gestor": ["GESTORA X", "GESTORA Y"],
            "vl_pl": [50e6, 80e6],
            "prob_distress_6m": [0.15, 0.25],
            "inad_90p_pct": [0.02, 0.06],
            "subordinacao_pct": [0.20, 0.15],
        }
    )
    edges = pd.DataFrame(
        {
            "dt_comptc": [dt, dt],
            "entity_id": ["11111111000191", "22222222000192"],
            "cedente_doc": ["55555555000188", "55555555000188"],
            "share_pct": [25.0, 40.0],
            "is_pj": [1, 1],
        }
    )
    graph = _build_contagion_network_payload(live, edges, dt)
    assert len(graph["nodes"]) == 3  # 1 cedente compartilhado + 2 FIDCs
    assert len(graph["links"]) == 2

    rng = np.random.default_rng(9)
    oof = pd.DataFrame(
        {
            "y_true": rng.binomial(1, 0.2, size=300),
            "prob_lgbm_unc_raw": rng.uniform(0.01, 0.8, size=300),
            "prob_lgbm_unc_platt": rng.uniform(0.01, 0.6, size=300),
            "prob_lgbm_unc_iso": rng.uniform(0.01, 0.6, size=300),
            "prob_gbdt_ensemble_va": rng.uniform(0.01, 0.6, size=300),
            "prob_hybrid_gbdt_gnn": rng.uniform(0.01, 0.6, size=300),
        }
    )
    rel = _build_reliability_curve(oof)
    assert len(rel) == 5
    assert all(len(c["points"]) > 0 for c in rel)



def test_item_da_carteira_traz_a_faixa_de_61_a_90_dias() -> None:
    """A régua de atraso do observatório usa 1–60, 61–90 e 90+ dias; o export precisa da faixa do meio."""
    from types import SimpleNamespace

    from fidc_risk.export.web_bundle import watchlist_item

    r = SimpleNamespace(
        entity_id="11111111000191", denom_social="FIDC FICTICIO", dominant_sector="Consignado", nome_gestor="",
        is_aberto=1, vl_pl=12_345_678.0, prob_distress_6m=0.1234, va_p0=0.1, va_p1=0.15, va_width=0.05,
        inad_90p_pct=0.042, inad_61_90_pct=0.013, inad_early_1_60_pct=0.021, pdd_carteira_pct=0.05,
        subordinacao_pct=0.3, recompra_masking_index=0.01, scr_provision_gap_pct=0.0,
        contagion_cedente_inad90p_wt=0.0, cqr_inad90_h3_q10=0.03, cqr_inad90_h3_q50=0.04, cqr_inad90_h3_q90=0.06,
        shap_drivers_json="[]",
    )
    item = watchlist_item(r, [])
    assert item["late_61_90_pct"] == 1.3 and item["early_pct"] == 2.1 and item["inad90_pct"] == 4.2
    assert item["gestor"] == "Não informado"


def test_regua_do_observatorio_desenha_a_faixa_61_90() -> None:
    from fidc_risk.config import WEB_DIR

    js = (WEB_DIR / "app.js").read_text(encoding="utf-8")
    html = (WEB_DIR / "index.html").read_text(encoding="utf-8")
    assert "late_61_90_pct" in js and "seg-6190" in js
    assert "61–90 dias" in html and "Demais (em dia ou 61–90d)" not in html


def test_faixas_curtas_de_pl_no_front_seguem_as_faixas_do_anonimizador() -> None:
    import re

    from fidc_risk.config import WEB_DIR
    from fidc_risk.export.anonymize import PL_BAND_LABELS

    js = (WEB_DIR / "app.js").read_text(encoding="utf-8")
    curtas = re.search(r"PL_BAND_SHORT = \[(.*?)\];", js).group(1).split(", ")
    assert len(curtas) == len(PL_BAND_LABELS)


def test_faixa_de_indicadores_usa_o_modelo_que_gera_o_score() -> None:
    """Os scores na tela vêm do LightGBM monotônico calibrado (modelo servido); a faixa de KPIs mostra esse modelo."""
    from fidc_risk.config import WEB_DIR

    js = (WEB_DIR / "app.js").read_text(encoding="utf-8")
    html = (WEB_DIR / "index.html").read_text(encoding="utf-8")
    assert 'r.model === "lgbm_monotonic_va"' in js
    assert "GBDT+GNN" not in html

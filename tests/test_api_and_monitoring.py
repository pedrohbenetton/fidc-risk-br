"""Testes de Monitoramento de Drift (PSI / JS), Explicabilidade e Endpoints FastAPI."""

from __future__ import annotations

import json

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from fidc_risk.api.app import RiskServiceState, create_app
from fidc_risk.features.gold import ALL_FEATURE_COLS
from fidc_risk.models.calibration import InductiveVennAbersCalibrator
from fidc_risk.monitoring.shap_and_drift import (
    compute_js_divergence,
    compute_psi,
    human_feature_label,
)


def test_psi_is_near_zero_for_identical_distributions_and_high_under_shift() -> None:
    rng = np.random.default_rng(101)
    ref = rng.normal(0.0, 1.0, size=2000)
    same = rng.normal(0.0, 1.0, size=2000)
    shifted = rng.normal(1.5, 1.2, size=2000)

    psi_same = compute_psi(ref, same)
    psi_shift = compute_psi(ref, shifted)
    assert 0.0 <= psi_same < 0.03
    assert psi_shift > 0.35


def test_js_divergence_symmetry_and_bounds() -> None:
    rng = np.random.default_rng(55)
    a = rng.exponential(1.0, size=1000)
    b = rng.exponential(2.5, size=1000)
    js_ab = compute_js_divergence(a, b)
    js_ba = compute_js_divergence(b, a)
    assert np.isclose(js_ab, js_ba, atol=1e-9)
    assert 0.0 <= js_ab <= np.log(2.0) + 1e-6


@pytest.mark.parametrize(
    "col",
    [
        "inad_90p_pct",
        "recompra_masking_index",
        "scr_provision_gap_pct",
        "subordinacao_pct",
        "contagion_cedente_inad90p_wt",
        "unknown_custom_col",
    ],
)
def test_human_feature_label_returns_non_empty_string(col: str) -> None:
    lbl = human_feature_label(col)
    assert isinstance(lbl, str) and len(lbl) >= 4


def _build_mock_service_state() -> RiskServiceState:
    rng = np.random.default_rng(2026)
    n = 120
    x = pd.DataFrame(rng.uniform(0.0, 0.3, size=(n, len(ALL_FEATURE_COLS))), columns=ALL_FEATURE_COLS)
    y = ((x["inad_90p_pct"] + x["inad_early_1_60_pct"] - 0.5 * x["subordinacao_pct"]) > 0.22).astype(int).values

    clf = lgb.LGBMClassifier(n_estimators=25, max_depth=3, min_child_samples=10, random_state=42, verbose=-1)
    clf.fit(x, y)
    raw = clf.predict_proba(x)[:, 1]
    va = InductiveVennAbersCalibrator(grid_size=64).fit(raw, y)

    live_rows = x.iloc[:6].copy()
    live_rows["entity_id"] = [f"1000000000010{i}" for i in range(6)]
    live_rows["cnpj_raw"] = live_rows["entity_id"]
    live_rows["denom_social"] = [
        "ALPHA CREDITO CORPORATIVO FIDC",
        "BETA CONSIGNADO ABERTO FIDC",
        "GAMMA VAREJO MULTICARTEIRA FIDC",
        "DELTA AGRONEGOCIO FIDC",
        "EPSILON RECEBIVEIS INDUSTRIAIS FIDC",
        "ZETA FINTECH CARTAO FIDC",
    ]
    live_rows["dominant_sector"] = ["Corporativo", "Consignado", "Comercial/Varejo", "Agronegócio", "Industrial", "Cartões"]
    live_rows["nome_gestor"] = "GESTORA QUANTITATIVA LTDA"
    live_rows["is_aberto"] = [1, 1, 0, 0, 1, 1]
    live_rows["vl_pl"] = [50e6, 120e6, 30e6, 80e6, 15e6, 200e6]
    live_rows["dt_comptc"] = pd.Timestamp("2026-10-31")
    live_rows["prob_distress_6m"] = [0.28, 0.04, 0.19, 0.11, 0.06, 0.34]
    live_rows["va_p0"] = [0.24, 0.02, 0.15, 0.08, 0.04, 0.29]
    live_rows["va_p1"] = [0.33, 0.06, 0.23, 0.14, 0.09, 0.40]
    live_rows["va_width"] = live_rows["va_p1"] - live_rows["va_p0"]
    live_rows["cqr_inad90_h3_q10"] = 0.01
    live_rows["cqr_inad90_h3_q50"] = 0.04
    live_rows["cqr_inad90_h3_q90"] = 0.11
    live_rows["shap_drivers_json"] = json.dumps(
        [{"feature": "inad_90p_pct", "label": "Inadimplência >90d", "shap_log_odds": 0.42, "feature_value": 0.08}]
    )

    return RiskServiceState(
        live_df=live_rows,
        served_tab={"lgbm_monotonic": clf, "venn_abers": va, "feature_cols": ALL_FEATURE_COLS},
        served_cqr=None,
    )


def test_fastapi_health_watchlist_and_counterfactual_endpoints() -> None:
    app = create_app(custom_state=_build_mock_service_state())
    client = TestClient(app)

    r_health = client.get("/health")
    assert r_health.status_code == 200
    js_h = r_health.json()
    assert js_h["status"] == "ok"
    assert js_h["model_loaded"] is True
    assert js_h["live_watchlist_count"] == 6

    r_wl = client.get("/api/v1/watchlist?only_open_end=true&min_pl_millions=40")
    assert r_wl.status_code == 200
    wl = r_wl.json()
    assert len(wl) == 3
    # Ordenado por maior risco primeiro
    assert wl[0]["prob_distress_6m"] >= wl[1]["prob_distress_6m"]

    r_search = client.get("/api/v1/watchlist?search=ALPHA")
    assert r_search.status_code == 200
    assert len(r_search.json()) == 1

    # Inferência contrafactual ao vivo
    r_score = client.post(
        "/api/v1/score",
        json={
            "entity_id": "10000000000101",
            "feature_overrides": {"inad_90p_pct": 0.45, "inad_early_1_60_pct": 0.35, "subordinacao_pct": 0.01},
        },
    )
    assert r_score.status_code == 200
    sc = r_score.json()
    assert sc["shocked_prob_6m"] >= sc["base_prob_6m"]
    assert sc["shocked_va_p0"] <= sc["shocked_va_p1"]
    assert len(sc["top_shap_drivers"]) > 0

    # Feature inválida deve retornar 422
    r_bad = client.post("/api/v1/score", json={"feature_overrides": {"non_existent_feature": 1.0}})
    assert r_bad.status_code == 422

    # Fundo inexistente deve retornar 404
    r_404 = client.post("/api/v1/score", json={"entity_id": "99999999000199", "feature_overrides": {}})
    assert r_404.status_code == 404


def test_pacote_nominal_so_vem_de_data_private(tmp_path, monkeypatch) -> None:
    """O pacote com fundos identificados é servido pela API local a partir de data/private/, nunca de web/."""
    from fidc_risk import config

    app = create_app(custom_state=_build_mock_service_state())
    client = TestClient(app)

    monkeypatch.setattr(config, "PRIVATE_BUNDLE_PATH", tmp_path / "ausente.json")
    assert client.get("/data/private/observatory_bundle.json").status_code == 404

    private = tmp_path / "observatory_bundle.json"
    private.write_text('{"meta": {"nominal": true}}', encoding="utf-8")
    monkeypatch.setattr(config, "PRIVATE_BUNDLE_PATH", private)
    r = client.get("/data/private/observatory_bundle.json")
    assert r.status_code == 200
    assert r.json() == {"meta": {"nominal": True}}

"""Testes de Embargo Walk-Forward, Calibração Venn-Abers, Roll-Rate de Markov e CQR."""

from __future__ import annotations

import numpy as np
import pandas as pd

from fidc_risk.models.calibration import (
    InductiveVennAbersCalibrator,
    IsotonicCalibrator,
    PlattCalibrator,
)
from fidc_risk.models.metrics import (
    compute_binary_warning_metrics,
    expected_calibration_error,
    weighted_interval_score,
)
from fidc_risk.models.rollrate_cqr import (
    QUANTILE_ALPHAS,
    EmpiricalBayesMarkovRollRate,
    StratifiedCQRCalibrator,
    _enforce_quantile_monotonicity,
)
from fidc_risk.models.validation import build_walk_forward_splits


def test_walk_forward_splits_enforce_strict_horizon_embargo() -> None:
    dates = pd.date_range("2020-01-31", "2026-06-30", freq="ME")
    rows = []
    for dt in dates:
        for k in range(15):
            rows.append(
                {
                    "entity_id": f"E{k:03d}",
                    "dt_comptc": dt,
                    "is_eligible_watchlist": 1,
                    "target_distress_6m": float(k % 5 == 0),
                    "target_distress_3m": float(k % 7 == 0),
                }
            )
    panel = pd.DataFrame(rows)

    for h in (3, 6):
        splits = build_walk_forward_splits(panel, f"target_distress_{h}m", horizon_months=h)
        assert len(splits) == 4
        for sp in splits:
            # Nenhum ponto de treino pode sobrepor teste
            assert not (sp.train_full_mask & sp.test_mask).any()
            assert not (sp.fit_mask & sp.cal_mask).any()
            # A data máxima de realização do rótulo de treino (t + h) deve ser ESTRITAMENTE anterior ao teste!
            assert sp.max_train_label_realization < sp.min_test_origin
            # E o embargo entre fit_mask e cal_mask também deve ser >= h meses
            max_fit_ym = (
                panel.loc[sp.fit_mask, "dt_comptc"].max().year * 12
                + panel.loc[sp.fit_mask, "dt_comptc"].max().month
            )
            min_cal_ym = (
                panel.loc[sp.cal_mask, "dt_comptc"].min().year * 12
                + panel.loc[sp.cal_mask, "dt_comptc"].min().month
            )
            assert max_fit_ym + h < min_cal_ym


def test_inductive_venn_abers_monotonicity_and_bounds() -> None:
    rng = np.random.default_rng(123)
    scores_cal = rng.beta(2.0, 5.0, size=600)
    probs_true = 1.0 / (1.0 + np.exp(-(scores_cal * 8.0 - 2.5)))
    y_cal = rng.binomial(1, probs_true)

    va = InductiveVennAbersCalibrator(grid_size=128).fit(scores_cal, y_cal)
    test_grid = np.linspace(0.01, 0.99, 200)
    out = va.predict_full(test_grid)

    assert np.all((out.p0 >= 0.0) & (out.p0 <= 1.0))
    assert np.all((out.p1 >= out.p0 - 1e-12) & (out.p1 <= 1.0))
    assert np.all(out.width >= -1e-12)
    # Monotonicidade não-decrescente ao longo do escore bruto
    assert np.all(np.diff(out.p0) >= -1e-9)
    assert np.all(np.diff(out.p1) >= -1e-9)
    assert np.all(np.diff(out.prob) >= -1e-9)


def test_platt_and_isotonic_calibrators_produce_valid_probabilities() -> None:
    rng = np.random.default_rng(7)
    s = rng.uniform(0.01, 0.99, size=300)
    y = (s + rng.normal(0, 0.15, size=300) > 0.55).astype(int)

    p_platt = PlattCalibrator().fit(s, y).predict(s)
    p_iso = IsotonicCalibrator().fit(s, y).predict(s)
    assert np.all((p_platt > 0.0) & (p_platt < 1.0))
    assert np.all((p_iso > 0.0) & (p_iso < 1.0))
    m = compute_binary_warning_metrics(y, p_iso)
    assert 0.5 <= m["roc_auc"] <= 1.0
    assert expected_calibration_error(y, p_iso) < 0.15


def test_markov_roll_rate_matrices_are_row_stochastic() -> None:
    rng = np.random.default_rng(99)
    dates = pd.date_range("2022-01-31", periods=8, freq="ME")
    rows = []
    for dt in dates:
        for i in range(30):
            s = rng.dirichlet([12.0, 1.5, 0.8, 0.5, 1.2])
            rows.append(
                {
                    "entity_id": f"F{i:02d}",
                    "dt_comptc": dt,
                    "dominant_sector": "Comercial/Varejo" if i < 15 else "Financeiro",
                    "inad_1_30_pct": s[1],
                    "inad_31_60_pct": s[2],
                    "inad_61_90_pct": s[3],
                    "inad_90p_pct": s[4],
                }
            )
    df = pd.DataFrame(rows)
    mk = EmpiricalBayesMarkovRollRate(shrinkage_kappa=50.0).fit(df)

    assert mk.global_matrix.shape == (5, 5)
    assert np.all(mk.global_matrix >= -1e-12)
    assert np.allclose(mk.global_matrix.sum(axis=1), 1.0, atol=1e-6)
    for sec_mat in mk.sector_matrices.values():
        assert np.all(sec_mat >= -1e-12)
        assert np.allclose(sec_mat.sum(axis=1), 1.0, atol=1e-6)

    preds = mk.predict_inad90_point(df, horizon=3)
    assert preds.shape == (len(df),)
    assert np.all((preds >= 0.0) & (preds <= 1.0))


def test_stratified_cqr_achieves_finite_sample_coverage_and_preserves_median() -> None:
    rng = np.random.default_rng(2026)
    n = 1200
    x = rng.uniform(0.0, 0.25, size=n)
    noise = rng.normal(0.0, 0.04, size=n)
    y = np.clip(x + noise, 0.0, 1.0)
    stratum = np.zeros(n, dtype="int32")
    stratum[x > 0.10] = 2

    # Quantis brutos propositalmente estreitos demais (sub-cobertos!)
    q_raw = {a: np.clip(x + (a - 0.50) * 0.02, 0.0, 1.0) for a in QUANTILE_ALPHAS}
    q_cal = {a: v[:600] for a, v in q_raw.items()}
    q_te = {a: v[600:] for a, v in q_raw.items()}

    cqr = StratifiedCQRCalibrator().fit(q_cal, y[:600], stratum[:600])
    q_adj = cqr.apply(q_te, stratum[600:])

    # Mediana intacta
    assert np.allclose(q_adj[0.50], q_te[0.50])
    # Não-cruzamento de quantis
    for lo, hi in zip(QUANTILE_ALPHAS[:-1], QUANTILE_ALPHAS[1:], strict=False):
        assert np.all(q_adj[lo] <= q_adj[hi] + 1e-12)

    wm = weighted_interval_score(y[600:], q_adj)
    # Cobertura empírica de 90% em dados permutáveis deve ficar próxima de 90% (>= 85%)
    assert wm["coverage_90"] >= 0.85
    assert len(_enforce_quantile_monotonicity(q_adj)) == len(QUANTILE_ALPHAS)

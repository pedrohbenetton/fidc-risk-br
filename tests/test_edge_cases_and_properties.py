"""Suíte de Testes de Propriedades Matemáticas, Casos Extremos CVM e Identidades Financeiras."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fidc_risk.allocation.portfolio import _compute_portfolio_summary_stats, _select_cvar_va_weights
from fidc_risk.export.web_bundle import _format_cnpj, _mask_cpf_if_needed
from fidc_risk.features.gold import ALL_FEATURE_COLS, FEATURE_FAMILIES, MONOTONIC_CONSTRAINTS_MAP
from fidc_risk.lakehouse.transforms import (
    classify_tranche_label,
    harmonize_percentage_share,
    normalize_doc,
    sanitize_monthly_return_pct,
    strip_accents_upper,
)
from fidc_risk.models.calibration import (
    InductiveVennAbersCalibrator,
    IsotonicCalibrator,
    PlattCalibrator,
)
from fidc_risk.models.metrics import (
    compute_binary_warning_metrics,
    expected_calibration_error,
)
from fidc_risk.models.rollrate_cqr import (
    QUANTILE_ALPHAS,
    _delinquency_stratum,
    _enforce_quantile_monotonicity,
    _state_matrix_5,
)
from fidc_risk.monitoring.shap_and_drift import compute_js_divergence, compute_psi


def test_feature_families_partition_is_disjoint_and_complete() -> None:
    seen: set[str] = set()
    for fam, cols in FEATURE_FAMILIES.items():
        assert len(cols) > 0, f"Família vazia: {fam}"
        overlap = seen.intersection(cols)
        assert not overlap, f"Colunas duplicadas entre famílias: {overlap}"
        seen.update(cols)
    assert len(seen) == len(ALL_FEATURE_COLS) == 103


@pytest.mark.parametrize("col,sign", list(MONOTONIC_CONSTRAINTS_MAP.items()))
def test_monotonic_constraints_reference_valid_features_and_signs(col: str, sign: int) -> None:
    assert col in ALL_FEATURE_COLS
    assert sign in (-1, 1)


@pytest.mark.parametrize(
    ("raw_label", "expected"),
    [
        ("SUBCLASSE SENIOR SERIE 10", "SENIOR"),
        ("COTA SÊNIOR ÚNICA", "SENIOR"),
        ("SR 1", "SENIOR"),
        ("SÉRIE 4", "SENIOR"),
        ("SUBORDINADA MEZANINO ESPECIAL", "MEZANINO"),
        ("MEZANINO A", "MEZANINO"),
        ("CLASSE MEZ 2", "MEZANINO"),
        ("COTA SUBORDINADA JUNIOR", "SUBORDINADA"),
        ("SUBORDINADA", "SUBORDINADA"),
        ("JR 1", "SUBORDINADA"),
        ("JUNIOR PREFERENCIAL", "SUBORDINADA"),
        ("COTA UNICA", "UNICA"),
        ("CLASSE ÚNICA", "UNICA"),
        ("OUTROS", "UNICA"),
    ],
)
def test_tranche_classifier_exhaustive_variants(raw_label: str, expected: str) -> None:
    assert classify_tranche_label(raw_label) == expected


@pytest.mark.parametrize(
    ("doc_in", "expected_out"),
    [
        ("00360305000104.0", "00360305000104"),
        ("07652226000116.00", "07652226000116"),
        ("  61.064.911/0001-77  ", "61064911000177"),
        ("000.000.000-00", ""),
        ("11.111.111/1111-11", ""),
        ("22222222222222", ""),
        ("88888888888", ""),
        ("1234567890123", ""),   # 13 dígitos -> inválido
        ("123456789012345", ""), # 15 dígitos -> inválido
    ],
)
def test_normalize_doc_float_suffix_and_repeated_digit_edge_cases(doc_in: str, expected_out: str) -> None:
    assert normalize_doc(doc_in) == expected_out


@pytest.mark.parametrize(
    ("inad_val", "expected_stratum"),
    [
        (0.0, 0),
        (0.00005, 0),
        (0.005, 1),
        (0.020, 1),
        (0.021, 2),
        (0.080, 2),
        (0.081, 3),
        (0.450, 3),
    ],
)
def test_delinquency_stratum_boundaries(inad_val: float, expected_stratum: int) -> None:
    out = _delinquency_stratum(np.array([inad_val]))
    assert int(out[0]) == expected_stratum


def test_state_matrix_5_projects_onto_probability_simplex_even_with_noisy_inputs() -> None:
    df = pd.DataFrame(
        {
            "inad_1_30_pct": [0.0, 0.40, 0.50],
            "inad_31_60_pct": [0.0, 0.30, 0.50],
            "inad_61_90_pct": [0.0, 0.20, 0.40],
            "inad_90p_pct": [0.0, 0.05, 0.30],  # Linha 2 soma 1.70 (> 1.0) -> deve reescalonar p/ soma=1
        }
    )
    s = _state_matrix_5(df)
    assert s.shape == (3, 5)
    assert np.all(s >= -1e-12)
    assert np.allclose(s.sum(axis=1), 1.0, atol=1e-9)
    assert np.isclose(s[0, 0], 1.0)  # Carteira 100% adimplente


def test_enforce_quantile_monotonicity_repairs_inverted_quantiles() -> None:
    # Quantis propositalmente invertidos (crossing quantiles)
    crossed = {
        0.05: np.array([0.30, -0.10]),
        0.10: np.array([0.28, 0.05]),
        0.25: np.array([0.25, 0.12]),
        0.50: np.array([0.20, 0.10]),
        0.75: np.array([0.15, 0.08]),
        0.90: np.array([0.12, 0.50]),
        0.95: np.array([0.10, 1.40]),
    }
    fixed = _enforce_quantile_monotonicity(crossed)
    assert np.allclose(fixed[0.50], [0.20, 0.10])
    for lo, hi in zip(QUANTILE_ALPHAS[:-1], QUANTILE_ALPHAS[1:], strict=False):
        assert np.all(fixed[lo] <= fixed[hi] + 1e-12)
        assert np.all((fixed[lo] >= 0.0) & (fixed[hi] <= 1.0))


@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
def test_venn_abers_interval_shrinks_as_sample_size_grows(seed: int) -> None:
    """Propriedade estatística fundamental do IVAP: largura média |p1 - p0| diminui com mais dados."""
    rng = np.random.default_rng(seed)
    s_small = rng.uniform(0, 1, size=60)
    y_small = rng.binomial(1, s_small)
    s_large = rng.uniform(0, 1, size=1500)
    y_large = rng.binomial(1, s_large)

    grid_eval = np.linspace(0.05, 0.95, 100)
    w_small = InductiveVennAbersCalibrator(grid_size=64).fit(s_small, y_small).predict_full(grid_eval).width.mean()
    w_large = InductiveVennAbersCalibrator(grid_size=128).fit(s_large, y_large).predict_full(grid_eval).width.mean()
    assert w_large < w_small


def test_calibrators_gracefully_handle_single_class_degenerate_data() -> None:
    s = np.linspace(0.1, 0.9, 30)
    y_zeros = np.zeros(30, dtype="int32")

    p_platt = PlattCalibrator().fit(s, y_zeros).predict(s)
    p_iso = IsotonicCalibrator().fit(s, y_zeros).predict(s)
    p_va = InductiveVennAbersCalibrator(grid_size=32).fit(s, y_zeros).predict(s)

    assert np.all(np.isfinite(p_platt))
    assert np.all(np.isfinite(p_iso))
    assert np.all(np.isfinite(p_va))


def test_formatting_and_cpf_privacy_masking() -> None:
    assert _format_cnpj("05754060000113") == "05.754.060/0001-13"
    assert _mask_cpf_if_needed("05754060000113", is_pj=1) == "05.754.060/0001-13"
    masked_cpf = _mask_cpf_if_needed("12345678901", is_pj=0)
    assert masked_cpf == "***.456.789-**"
    assert "123" not in masked_cpf and "01" not in masked_cpf


@pytest.mark.parametrize(
    ("raw_txt", "expected_ascii"),
    [
        ("Cota Sênior Única", "COTA SENIOR UNICA"),
        ("Crédito Agronegócio & Imobiliário", "CREDITO AGRONEGOCIO & IMOBILIARIO"),
        ("  Subordinada Júnior  ", "SUBORDINADA JUNIOR"),
    ],
)
def test_strip_accents_upper(raw_txt: str, expected_ascii: str) -> None:
    assert strip_accents_upper(raw_txt) == expected_ascii


def test_portfolio_100pct_cdi_has_zero_excess_return_and_zero_drawdown() -> None:
    df = pd.DataFrame({"cdi": [0.95, 1.02, 1.08, 0.99, 1.05, 1.10], "cdi_ret_pct": [0.95, 1.02, 1.08, 0.99, 1.05, 1.10]})
    st = _compute_portfolio_summary_stats(df, "cdi", "cdi_ret_pct")
    assert np.isclose(st["pct_of_cdi"], 100.0, atol=0.1)
    assert np.isclose(st["excess_over_cdi_aa_pp"], 0.0, atol=1e-6)
    assert np.isclose(st["max_drawdown_pct"], 0.0, atol=1e-6)


def test_cvar_weights_park_100pct_in_cdi_when_all_funds_are_high_risk() -> None:
    bad_universe = pd.DataFrame(
        {
            "entity_id": ["F1", "F2", "F3"],
            "dominant_sector": ["Varejo", "Pessoal", "Cartões"],
            "rentab_senior_3m_excess": [0.20, 0.30, 0.25],
            "total_enhancement_pct": [0.05, 0.04, 0.02],
            "prob_risk": [0.35, 0.42, 0.55],  # Todos acima do limite de segurança!
            "va_width_risk": [0.12, 0.15, 0.20],
        }
    )
    w = _select_cvar_va_weights(bad_universe)
    assert np.allclose(w.values, 0.0)


@pytest.mark.parametrize("n_bins", [5, 10, 15])
def test_ece_zero_for_perfectly_calibrated_bernoulli_oracle(n_bins: int) -> None:
    rng = np.random.default_rng(2026)
    p = np.repeat(np.linspace(0.05, 0.85, 9), 4000)
    y = rng.binomial(1, p)
    ece = expected_calibration_error(y, p, n_bins=n_bins)
    assert ece < 0.015
    m = compute_binary_warning_metrics(y, p)
    assert m["roc_auc"] > 0.75


def test_psi_and_js_handle_constant_and_binary_arrays() -> None:
    zeros = np.zeros(200)
    ones = np.ones(200)
    assert compute_js_divergence(zeros, zeros) == 0.0
    b_ref = np.concatenate([np.zeros(150), np.ones(50)])
    b_cur = np.concatenate([np.zeros(50), np.ones(150)])
    assert compute_psi(b_ref, b_cur) > 0.50
    assert compute_psi(ones[:5], zeros[:5]) == 0.0


def test_harmonize_percentage_and_return_sanitization_idempotence() -> None:
    s = pd.Series([0.0, 12.5, 99.9])
    assert np.allclose(harmonize_percentage_share(s), s)
    r = pd.Series([-12.4, 0.0, 1.15, 3.8])
    assert np.allclose(sanitize_monthly_return_pct(r), r)

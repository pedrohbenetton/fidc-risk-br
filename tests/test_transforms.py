"""Testes unitários das transformações contábeis, documentos e classificação de tranches CVM."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fidc_risk.lakehouse.transforms import (
    classify_tranche_label,
    harmonize_percentage_share,
    normalize_doc,
    normalize_doc_series,
    sanitize_monthly_return_pct,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("05.754.060/0001-13", "05754060000113"),
        ("123.456.789-01", "12345678901"),
        ("00.000.000/0000-00", ""),
        ("99999999999", ""),
        ("12345", ""),
        (None, ""),
        (np.nan, ""),
    ],
)
def test_normalize_doc_handles_punctuation_and_placeholders(raw: object, expected: str) -> None:
    assert normalize_doc(raw) == expected


def test_normalize_doc_series_matches_scalar() -> None:
    raw = pd.Series(["05.754.060/0001-13", "00000000000000", "123.456.789-01", None, "abc"])
    out = normalize_doc_series(raw)
    assert out.tolist() == ["05754060000113", "", "12345678901", "", ""]


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("Subclasse Senior Série 1", "SENIOR"),
        ("Cota Sênior 2ª Série", "SENIOR"),
        ("Série 1", "SENIOR"),
        ("Subclasse Subordinada Mezanino 1 | Série 1", "MEZANINO"),
        ("Cota Subordinada Mezanino A", "MEZANINO"),
        ("Subclasse Subordinada Subordinada 1 |", "SUBORDINADA"),
        ("Cota Subordinada Júnior", "SUBORDINADA"),
        ("Subordinada Série 1", "SUBORDINADA"),
        ("Classe Única", "UNICA"),
        ("", "UNICA"),
        (None, "UNICA"),
    ],
)
def test_classify_tranche_label_respects_waterfall_precedence(label: object, expected: str) -> None:
    # Crítico: 'Subordinada Mezanino Série 1' contém SUBORD, MEZ e SERIE -> deve ser MEZANINO!
    assert classify_tranche_label(label) == expected


def test_harmonize_percentage_share_rescales_basis_points_and_caps() -> None:
    s = pd.Series([-5.0, 0.0, 24.5, 2540.0, 99999.0, np.nan])
    res = harmonize_percentage_share(s, max_cap=100.0)
    assert np.allclose(res.values, [0.0, 0.0, 24.5, 25.4, 100.0, 0.0])


def test_sanitize_monthly_return_preserves_defaults_and_masks_glitches() -> None:
    s = pd.Series([1.12, -35.4, -98.0, -150.0, 12500.0, np.nan])
    res = sanitize_monthly_return_pct(s)
    assert np.isclose(res.iloc[0], 1.12)
    assert np.isclose(res.iloc[1], -35.4)
    assert np.isclose(res.iloc[2], -98.0)
    assert np.isnan(res.iloc[3])
    assert np.isnan(res.iloc[4])
    assert np.isnan(res.iloc[5])

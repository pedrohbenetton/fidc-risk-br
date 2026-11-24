"""Testes extensivos de reconciliação RCVM 175, zeros à esquerda, axiomas do WIS e Bootstrap."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fidc_risk.lakehouse.silver import _deduplicate_entity_table, _parse_single_month_tables
from fidc_risk.models.metrics import (
    paired_entity_bootstrap_ci,
    pinball_loss,
    watchlist_topk_metrics,
    weighted_interval_score,
)


@pytest.mark.parametrize(
    ("raw_cnpj", "raw_classe", "expected_id"),
    [
        ("00.360.305/0001-04", "", "00360305000104"),
        ("05.754.060/0001-13", "05754060000113", "05754060000113"),
        ("08.632.394/0001-02", "00762723000128", "00762723000128"),
        ("12.345.678/0001-95", "00000000000000", "12345678000195"),
        ("01.201.140/0001-90", None, "01201140000190"),
    ],
)
def test_leading_zero_cnpjs_preserved_in_entity_resolution(
    raw_cnpj: str, raw_classe: str | None, expected_id: str
) -> None:
    df = pd.DataFrame(
        {
            "TP_FUNDO_CLASSE": ["Classe"],
            "CNPJ_FUNDO_CLASSE": [raw_cnpj],
            "CNPJ_CLASSE": [raw_classe],
            "DT_COMPTC": ["2025-06-30"],
            "TAB_I_VL_ATIVO": ["10000000.00"],
        }
    )
    dedup, removed = _deduplicate_entity_table(df)
    assert removed == 0
    assert dedup["entity_id"].iloc[0] == expected_id
    assert len(dedup["entity_id"].iloc[0]) == 14


def test_rcvm175_transition_duplicate_prioritizes_classe_over_fundo() -> None:
    df = pd.DataFrame(
        {
            "TP_FUNDO_CLASSE": ["Fundo", "Classe"],
            "CNPJ_FUNDO_CLASSE": ["08.632.394/0001-02", "08.632.394/0001-02"],
            "CNPJ_CLASSE": ["08632394000102", "08632394000102"],
            "DENOM_SOCIAL": ["VERSAO LEGADA FUNDO", "VERSAO MIGRADA CLASSE RCVM175"],
            "DT_COMPTC": ["2024-01-31", "2024-01-31"],
            "TAB_I_VL_ATIVO": ["50000000.0", "51000000.0"],
        }
    )
    dedup, removed = _deduplicate_entity_table(df)
    assert removed == 1
    assert len(dedup) == 1
    assert dedup["DENOM_SOCIAL"].iloc[0] == "VERSAO MIGRADA CLASSE RCVM175"


def test_parse_single_month_tables_reconciles_tranches_and_cedente_edges() -> None:
    t1 = pd.DataFrame(
        {
            "TP_FUNDO_CLASSE": ["Classe"],
            "CNPJ_FUNDO_CLASSE": ["05.754.060/0001-13"],
            "DENOM_SOCIAL": ["CATERPILLAR FIDC"],
            "DT_COMPTC": ["2025-06-30"],
            "CNPJ_ADMIN": ["00.066.670/0001-00"],
            "ADMIN": ["BEM DTVM"],
            "CONDOM": ["ABERTO"],
            "FUNDO_EXCLUSIVO": ["N"],
            "PRAZO_CONVERSAO_COTA": ["21"],
            "TP_PRAZO_CONVERSAO_COTA": ["DIAS ÚTEIS"],
            "PRAZO_PAGTO_RESGATE": ["1"],
            "TP_PRAZO_PAGTO_RESGATE": ["DIAS CORRIDOS"],
            "TAB_I_VL_ATIVO": ["100000000"],
            "TAB_I1_VL_DISP": ["5000000"],
            "TAB_I2_VL_CARTEIRA": ["90000000"],
            "TAB_I2A11_VL_REDUCAO_RECUP": ["2000000"],
            "TAB_I2A12_CPF_CNPJ_CEDENTE_1": ["00360305000104"],
            "TAB_I2A12_PR_CEDENTE_1": ["35.5"],
            "TAB_I2A12_CPF_CNPJ_CEDENTE_2": ["00000000000000"],
            "TAB_I2A12_PR_CEDENTE_2": ["0.0"],
        }
    )
    tx2 = pd.DataFrame(
        {
            "CNPJ_FUNDO_CLASSE": ["05.754.060/0001-13", "05.754.060/0001-13", "05.754.060/0001-13"],
            "DT_COMPTC": ["2025-06-30"] * 3,
            "TAB_X_CLASSE_SERIE": [
                "Subclasse Senior Série 1",
                "Subclasse Subordinada Mezanino 1 | Série 1",
                "Subclasse Subordinada Júnior",
            ],
            "TAB_X_QT_COTA": ["70000", "10000", "20000"],
            "TAB_X_VL_COTA": ["1000", "1000", "1000"],
        }
    )
    tx3 = pd.DataFrame(
        {
            "CNPJ_FUNDO_CLASSE": ["05.754.060/0001-13", "05.754.060/0001-13"],
            "DT_COMPTC": ["2025-06-30"] * 2,
            "TAB_X_CLASSE_SERIE": ["Subclasse Senior Série 1", "Subclasse Subordinada Júnior"],
            "TAB_X_VL_RENTAB_MES": ["1.15", "2.40"],
        }
    )
    core, edges, _ = _parse_single_month_tables("202506", {"tab_I": t1, "tab_X_2": tx2, "tab_X_3": tx3})
    assert len(core) == 1
    assert len(edges) == 1
    assert edges["cedente_doc"].iloc[0] == "00360305000104"
    assert edges["is_pj"].iloc[0] == 1
    assert np.isclose(core["pl_senior"].iloc[0], 70_000_000.0)
    assert np.isclose(core["pl_mezanino"].iloc[0], 10_000_000.0)
    assert np.isclose(core["pl_subordinada"].iloc[0], 20_000_000.0)
    # 21 dias úteis convertidos p/ ~30 dias corridos
    assert np.isclose(core["prazo_conversao_dias"].iloc[0], 30.0)


def test_wis_proper_scoring_rule_axiom_true_quantiles_beat_miscalibrated() -> None:
    """Axioma de escore próprio estrito: quantis verdadeiros de N(mu, sigma^2) minimizam o WIS esperado."""
    rng = np.random.default_rng(777)
    from scipy.stats import norm

    n = 15_000
    mu = rng.uniform(0.05, 0.25, size=n)
    sigma = 0.03
    y = mu + rng.normal(0.0, sigma, size=n)

    alphas = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
    q_true = {a: mu + norm.ppf(a) * sigma for a in alphas}
    q_biased = {a: (mu + 0.04) + norm.ppf(a) * (sigma * 0.4) for a in alphas}

    wis_true = weighted_interval_score(y, q_true)["wis"]
    wis_biased = weighted_interval_score(y, q_biased)["wis"]
    assert wis_true < wis_biased


@pytest.mark.parametrize("alpha", [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95])
def test_pinball_loss_non_negative_and_zero_on_exact_match(alpha: float) -> None:
    y = np.array([0.02, 0.10, 0.35])
    assert np.isclose(pinball_loss(y, y, alpha), 0.0)
    assert pinball_loss(y, y + 0.05, alpha) > 0.0


def test_paired_entity_bootstrap_detects_clear_superior_model() -> None:
    rng = np.random.default_rng(12)
    entities = np.repeat([f"E{i:03d}" for i in range(80)], 5)
    y = rng.binomial(1, 0.2, size=len(entities))
    prob_good = np.clip(0.75 * y + rng.uniform(0.0, 0.25, size=len(y)), 0.01, 0.99)
    prob_random = rng.uniform(0.01, 0.99, size=len(y))

    res = paired_entity_bootstrap_ci(entities, y, prob_good, prob_random, n_boot=200, seed=42)
    assert res["pr_auc_diff_mean"] > 0.20
    assert res["pr_auc_diff_ci95_low"] > 0.10
    assert res["pr_auc_p_value_one_sided"] < 0.01
    assert res["brier_diff_mean"] < -0.05


@pytest.mark.parametrize("top_frac", [0.01, 0.05, 0.10, 0.20])
def test_watchlist_topk_metrics_bounds(top_frac: float) -> None:
    y = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    s = np.array([0.95, 0.85, 0.40, 0.30, 0.25, 0.20, 0.15, 0.10, 0.05, 0.01])
    m = watchlist_topk_metrics(y, s, top_frac=top_frac)
    assert 0.0 <= m["recall_top5"] <= 1.0
    assert 0.0 <= m["precision_top5"] <= 1.0
    assert m["lift_top5"] >= 1.0

"""Testes do Motor de Alocação CVaR + Venn-Abers, Estatísticas de Portfólio e CLI."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fidc_risk.allocation.portfolio import (
    _compute_portfolio_summary_stats,
    _select_cvar_va_weights,
    run_portfolio_backtest,
)
from fidc_risk.cli import build_parser
from fidc_risk.features.gold import _detect_fidc_np_structural


def test_cvar_venn_abers_weights_enforce_single_fund_and_sector_caps() -> None:
    rows = []
    for i in range(40):
        rows.append(
            {
                "entity_id": f"F{i:03d}",
                "dominant_sector": "Comercial/Varejo" if i < 20 else ("Consignado" if i < 30 else "Industrial"),
                "rentab_senior_3m_excess": 0.45 - i * 0.005,
                "total_enhancement_pct": 0.25,
                "prob_risk": 0.03,
                "va_width_risk": 0.02,
            }
        )
    sub = pd.DataFrame(rows)
    w = _select_cvar_va_weights(sub, max_single_w=0.05, max_sec_w=0.25)

    assert np.all(w >= 0.0)
    assert np.all(w <= 0.05 + 1e-9)
    assert float(w.sum()) <= 1.0 + 1e-9

    # Nenhum setor individual pode ultrapassar 25% da carteira
    sec_sums = sub.assign(w=w).groupby("dominant_sector")["w"].sum()
    assert np.all(sec_sums <= 0.25 + 1e-9)


def test_portfolio_summary_stats_computes_consistent_cdi_and_drawdown() -> None:
    df = pd.DataFrame(
        {
            "ret": [1.2, 1.1, -2.5, 1.3, 1.4, 1.2, 1.1, 1.2, 1.0, 1.3, 1.1, 1.2],
            "cdi_ret_pct": [1.0] * 12,
        }
    )
    st = _compute_portfolio_summary_stats(df, "ret", "cdi_ret_pct")
    assert st["months"] == 12
    assert st["max_drawdown_pct"] <= -2.4
    assert st["cvar_95_monthly_pct"] <= -2.0
    assert st["volatility_aa_pct"] > 0.0


def test_portfolio_backtest_runs_end_to_end_on_synthetic_panel() -> None:
    dates = pd.date_range("2023-01-31", periods=6, freq="ME")
    p_rows, o_rows = [], []
    rng = np.random.default_rng(19)
    for dt in dates:
        for i in range(25):
            risky = i < 4
            eid = f"200000000001{i:02d}"
            p_rows.append(
                {
                    "entity_id": eid,
                    "dt_comptc": dt,
                    "is_aberto": 1,
                    "is_exclusivo": 0,
                    "vl_pl": 25_000_000.0,
                    "pl_senior": 20_000_000.0,
                    "subordinacao_pct": 0.05 if risky else 0.25,
                    "total_enhancement_pct": 0.05 if risky else 0.25,
                    "inad_90p_pct": 0.08 if risky else 0.01,
                    "cash_liquidity_pct": 0.08,
                    "redemption_lag_days": 30.0,
                    "rentab_senior_3m_excess": 0.55 if risky else 0.22,
                    "fwd_ret_senior_1m_step": float(-3.5 if risky else rng.normal(1.18, 0.08)),
                    "fwd_ret_senior_2m_step": float(-4.0 if risky else rng.normal(1.20, 0.08)),
                    "fwd_cdi_1m_step": 1.00,
                    "fwd_cdi_2m_step": 1.00,
                }
            )
            o_rows.append(
                {
                    "entity_id": eid,
                    "dt_comptc": dt,
                    "dominant_sector": f"Sec_{i % 5}",
                    "y_true": int(risky),
                    "prob_gbdt_ensemble_va": 0.42 if risky else 0.03,
                    "va_width": 0.09 if risky else 0.02,
                }
            )
    m_df, sum_df, front_df = run_portfolio_backtest(pd.DataFrame(p_rows), pd.DataFrame(o_rows))
    assert len(m_df) == 6
    assert len(front_df) == 28
    # Yield-chasing deve cair na armadilha dos fundos arriscados de maior spread recente,
    # enquanto Watchlist ML e CVaR + Venn-Abers evitam os choques!
    yc_ret = float(sum_df.loc[sum_df["policy"] == "yield_chasing_top25", "cumulative_return_pct"].iloc[0])
    va_ret = float(sum_df.loc[sum_df["policy"] == "cvar_venn_abers_lagged", "cumulative_return_pct"].iloc[0])
    assert va_ret > yc_ret


@pytest.mark.parametrize(
    ("name", "expected_np"),
    [
        ("FUNDO DE INVESTIMENTO EM DIREITOS CREDITORIOS NAO PADRONIZADOS OMEGA", True),
        ("ALPHA FIDC-NP DE CREDITOS JUDICIAIS", True),
        ("PRECATORIOS FEDERAIS FIDC", True),
        ("CATERPILLAR FIDC DO SEGMENTO INDUSTRIAL", False),
        ("MERCADO CREDITO FIDC CONSIGNADO", False),
    ],
)
def test_fidc_np_structural_detector(name: str, expected_np: bool) -> None:
    res = _detect_fidc_np_structural(pd.Series([name]))
    assert bool(res.iloc[0]) is expected_np


@pytest.mark.parametrize(
    "subcmd",
    [
        ["ingest"],
        ["silver"],
        ["gold"],
        ["train-tabular"],
        ["train-rollrate"],
        ["train-gnn"],
        ["backtest-allocation"],
        ["export-web"],
        ["serve", "--port", "8080"],
    ],
)
def test_cli_parser_accepts_all_pipeline_commands(subcmd: list[str]) -> None:
    parser = build_parser()
    args = parser.parse_args(subcmd)
    assert args.command == subcmd[0]

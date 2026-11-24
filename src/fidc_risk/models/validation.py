"""Harness de Validação Walk-Forward com Embargo Estrito por Horizonte (Zero Label Overlap)."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from fidc_risk.config import WALK_FORWARD_FOLDS, WalkForwardFold


@dataclass(frozen=True)
class TemporalSplitIndices:
    fold: WalkForwardFold
    horizon_months: int
    fit_mask: pd.Series
    cal_mask: pd.Series
    train_full_mask: pd.Series
    test_mask: pd.Series
    max_train_origin: pd.Timestamp
    max_train_label_realization: pd.Timestamp
    min_test_origin: pd.Timestamp


def _ym_int(dt_series: pd.Series) -> pd.Series:
    return dt_series.dt.year * 12 + dt_series.dt.month


def build_walk_forward_splits(
    panel: pd.DataFrame,
    target_col: str,
    horizon_months: int,
    eligible_only: bool = True,
    cal_months: int = 6,
) -> list[TemporalSplitIndices]:
    """
    Gera divisões expanding-window com embargo exato de `horizon_months`:
    - Se o conjunto de teste começa em T_start (ex.: jan/2023) e o horizonte é h=6 meses,
      qualquer linha de treino com origem t tem rótulo realizado até t + 6 meses.
      Exigimos estritamente: ym(t_train) + h < ym(T_start) (ex.: t_train <= jun/2022 -> rótulo até dez/2022!).
    - Da mesma forma, ao separar a janela interna de calibração/early-stopping (`cal_months`),
      impomos embargo de `horizon_months` entre `fit_mask` e `cal_mask`!
    """
    base_valid = panel[target_col].notna()
    if eligible_only and "is_eligible_watchlist" in panel.columns:
        base_valid = base_valid & (panel["is_eligible_watchlist"] == 1)

    ym = _ym_int(panel["dt_comptc"])
    splits: list[TemporalSplitIndices] = []

    for fold in WALK_FORWARD_FOLDS:
        t_start = pd.Timestamp(fold.test_start)
        t_end = pd.Timestamp(fold.test_end)
        ym_test_start = t_start.year * 12 + t_start.month
        ym_test_end = t_end.year * 12 + t_end.month

        # Regra áurea anti-vazamento: ym_origin + horizon < ym_test_start
        max_train_ym = ym_test_start - horizon_months - 1
        train_full_mask = base_valid & (ym <= max_train_ym)
        test_mask = base_valid & (ym >= ym_test_start) & (ym <= ym_test_end)

        if not train_full_mask.any() or not test_mask.any():
            continue

        # Janela interna de calibração (últimos cal_months de train_full) + embargo antes dela
        cal_start_ym = max_train_ym - cal_months + 1
        max_fit_ym = cal_start_ym - horizon_months - 1

        cal_mask = train_full_mask & (ym >= cal_start_ym)
        fit_mask = train_full_mask & (ym <= max_fit_ym)

        # Fallback defensivo caso janela inicial sintética seja curta, mantendo SEMPRE o embargo h!
        if fit_mask.sum() < 50 or cal_mask.sum() < 20:
            cal_start_fb = max_train_ym - 2
            fit_mask = train_full_mask & (ym <= cal_start_fb - horizon_months - 1)
            cal_mask = train_full_mask & (ym >= cal_start_fb)

        max_tr_dt = panel.loc[train_full_mask, "dt_comptc"].max()
        max_tr_realization = (max_tr_dt + pd.offsets.MonthEnd(horizon_months)).normalize()
        min_te_dt = panel.loc[test_mask, "dt_comptc"].min()

        assert max_tr_realization < min_te_dt, (
            f"Violação de embargo em {fold.name}: realização máxima de treino {max_tr_realization} "
            f">= início do teste {min_te_dt}"
        )

        splits.append(
            TemporalSplitIndices(
                fold=fold,
                horizon_months=horizon_months,
                fit_mask=fit_mask,
                cal_mask=cal_mask,
                train_full_mask=train_full_mask,
                test_mask=test_mask,
                max_train_origin=max_tr_dt,
                max_train_label_realization=max_tr_realization,
                min_test_origin=min_te_dt,
            )
        )
    return splits

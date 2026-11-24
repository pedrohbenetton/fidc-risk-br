"""Fase 3: Suíte Tabular (Heurística, Hazard Logístico, LightGBM Monotônico, CatBoost), Calibração e Ablações."""

from __future__ import annotations

import json
import time
import warnings
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import RobustScaler

from fidc_risk.config import GOLD_DIR, METRICS_DIR, MODELS_DIR, ensure_directories
from fidc_risk.features.gold import ALL_FEATURE_COLS, FEATURE_FAMILIES, MONOTONIC_CONSTRAINTS_MAP
from fidc_risk.models.calibration import (
    InductiveVennAbersCalibrator,
    IsotonicCalibrator,
    PlattCalibrator,
)
from fidc_risk.models.metrics import compute_binary_warning_metrics, paired_entity_bootstrap_ci
from fidc_risk.models.validation import TemporalSplitIndices, build_walk_forward_splits


def _committee_heuristic_raw(df: pd.DataFrame) -> np.ndarray:
    """
    Regra estrutural clássica de comitê de crédito privado:
    Combina inadimplência >90d, atraso curto 1-60d, inadimplência descoberta sobre PL
    e insuficiência de subordinação.
    """
    raw = (
        2.5 * df["inad_90p_pct"].values
        + 1.8 * df["inad_early_1_60_pct"].values
        + 1.5 * df["unprovisioned_inad90p_pl"].values
        + 1.2 * np.maximum(0.15 - df["total_enhancement_pct"].values, 0.0)
        + 0.8 * np.maximum(-df["rentab_senior_excess_cdi"].values / 10.0, 0.0)
    )
    return 1.0 / (1.0 + np.exp(-(raw * 6.0 - 2.2)))


def _fit_logistic_hazard(
    x_fit: np.ndarray,
    y_fit: np.ndarray,
    x_cal: np.ndarray,
    y_cal: np.ndarray,
) -> tuple[RobustScaler, LogisticRegression, InductiveVennAbersCalibrator]:
    scaler = RobustScaler(quantile_range=(5.0, 95.0))
    xf = np.clip(scaler.fit_transform(x_fit), -8.0, 8.0)
    xc = np.clip(scaler.transform(x_cal), -8.0, 8.0)
    clf = LogisticRegression(C=0.25, max_iter=600, class_weight="balanced", solver="lbfgs")
    clf.fit(xf, y_fit)
    cal_raw = clf.predict_proba(xc)[:, 1]
    va = InductiveVennAbersCalibrator(grid_size=256).fit(cal_raw, y_cal)
    return scaler, clf, va


def _fit_lgbm(
    x_fit: pd.DataFrame,
    y_fit: np.ndarray,
    x_cal: pd.DataFrame,
    y_cal: np.ndarray,
    feature_cols: list[str],
    monotonic: bool = False,
    seed: int = 42,
) -> lgb.LGBMClassifier:
    pos_rate = max(float(y_fit.mean()), 1e-4)
    scale_pw = min(max((1.0 - pos_rate) / pos_rate, 1.0), 12.0)

    mono_vec = [MONOTONIC_CONSTRAINTS_MAP.get(c, 0) for c in feature_cols] if monotonic else None
    params: dict[str, Any] = {
        "n_estimators": 900,
        "learning_rate": 0.025,
        "num_leaves": 31,
        "max_depth": 6,
        "min_child_samples": 90,
        "subsample": 0.80,
        "subsample_freq": 1,
        "colsample_bytree": 0.75,
        "reg_alpha": 1.5,
        "reg_lambda": 5.0,
        "scale_pos_weight": np.sqrt(scale_pw),
        "random_state": seed,
        "n_jobs": -1,
        "verbose": -1,
    }
    if mono_vec is not None:
        params["monotone_constraints"] = mono_vec

    model = lgb.LGBMClassifier(**params)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(
            x_fit[feature_cols],
            y_fit,
            eval_set=[(x_cal[feature_cols], y_cal)],
            eval_metric="average_precision",
            callbacks=[lgb.early_stopping(stopping_rounds=60, verbose=False)],
        )
    return model


def _fit_catboost(
    df_fit: pd.DataFrame,
    y_fit: np.ndarray,
    df_cal: pd.DataFrame,
    y_cal: np.ndarray,
    feature_cols: list[str],
    seed: int = 42,
) -> tuple[CatBoostClassifier, list[str]]:
    cb_cols = feature_cols + ["dominant_sector"]
    fit_x = df_fit[cb_cols].copy()
    cal_x = df_cal[cb_cols].copy()
    fit_x["dominant_sector"] = fit_x["dominant_sector"].astype(str)
    cal_x["dominant_sector"] = cal_x["dominant_sector"].astype(str)

    pos_rate = max(float(y_fit.mean()), 1e-4)
    scale_pw = float(np.sqrt(min(max((1.0 - pos_rate) / pos_rate, 1.0), 12.0)))

    cb = CatBoostClassifier(
        iterations=900,
        learning_rate=0.035,
        depth=6,
        l2_leaf_reg=6.0,
        scale_pos_weight=scale_pw,
        eval_metric="PRAUC",
        random_seed=seed,
        verbose=False,
        allow_writing_files=False,
    )
    fit_pool = Pool(fit_x, y_fit, cat_features=["dominant_sector"])
    cal_pool = Pool(cal_x, y_cal, cat_features=["dominant_sector"])
    cb.fit(fit_pool, eval_set=cal_pool, early_stopping_rounds=60, verbose=False)
    return cb, cb_cols


def _evaluate_fold_models(
    panel: pd.DataFrame,
    split: TemporalSplitIndices,
    target_col: str,
) -> tuple[pd.DataFrame, dict[str, dict[str, float]], lgb.LGBMClassifier, InductiveVennAbersCalibrator]:
    df_fit = panel.loc[split.fit_mask]
    df_cal = panel.loc[split.cal_mask]
    df_te = panel.loc[split.test_mask].copy()

    y_fit = df_fit[target_col].values.astype("int32")
    y_cal = df_cal[target_col].values.astype("int32")
    y_te = df_te[target_col].values.astype("int32")

    # 1. Baseline Comitê Estrutural
    heur_cal = _committee_heuristic_raw(df_cal)
    heur_te_raw = _committee_heuristic_raw(df_te)
    heur_platt = PlattCalibrator().fit(heur_cal, y_cal)
    p_heur = heur_platt.predict(heur_te_raw)

    # 2. Hazard Logístico Regularizado + Venn-Abers
    scaler, lr_clf, lr_va = _fit_logistic_hazard(
        df_fit[ALL_FEATURE_COLS].values,
        y_fit,
        df_cal[ALL_FEATURE_COLS].values,
        y_cal,
    )
    x_te_sc = np.clip(scaler.transform(df_te[ALL_FEATURE_COLS].values), -8.0, 8.0)
    p_logit = lr_va.predict(lr_clf.predict_proba(x_te_sc)[:, 1])

    # 3. LightGBM Irrestrito (comparando Raw vs Platt vs Isotônica vs Venn-Abers)
    lgb_unc = _fit_lgbm(df_fit, y_fit, df_cal, y_cal, ALL_FEATURE_COLS, monotonic=False)
    unc_cal_raw = np.asarray(lgb_unc.predict_proba(df_cal[ALL_FEATURE_COLS]), dtype="float64")[:, 1]
    unc_te_raw = np.asarray(lgb_unc.predict_proba(df_te[ALL_FEATURE_COLS]), dtype="float64")[:, 1]

    p_unc_raw = unc_te_raw
    p_unc_platt = PlattCalibrator().fit(unc_cal_raw, y_cal).predict(unc_te_raw)
    p_unc_iso = IsotonicCalibrator().fit(unc_cal_raw, y_cal).predict(unc_te_raw)
    va_unc = InductiveVennAbersCalibrator(grid_size=384).fit(unc_cal_raw, y_cal)
    p_unc_va = va_unc.predict(unc_te_raw)

    # 4. LightGBM com Restrições Monotônicas Estruturais + Venn-Abers
    lgb_mono = _fit_lgbm(df_fit, y_fit, df_cal, y_cal, ALL_FEATURE_COLS, monotonic=True)
    mono_cal_raw = np.asarray(lgb_mono.predict_proba(df_cal[ALL_FEATURE_COLS]), dtype="float64")[:, 1]
    mono_te_raw = np.asarray(lgb_mono.predict_proba(df_te[ALL_FEATURE_COLS]), dtype="float64")[:, 1]
    va_mono = InductiveVennAbersCalibrator(grid_size=384).fit(mono_cal_raw, y_cal)
    mono_va_pred = va_mono.predict_full(mono_te_raw)
    p_mono_va = mono_va_pred.prob

    # 5. CatBoost Ordenado com Setor Categórico Nativo + Venn-Abers
    cb_model, cb_cols = _fit_catboost(df_fit, y_fit, df_cal, y_cal, ALL_FEATURE_COLS)
    cal_cb_x = df_cal[cb_cols].copy()
    te_cb_x = df_te[cb_cols].copy()
    cal_cb_x["dominant_sector"] = cal_cb_x["dominant_sector"].astype(str)
    te_cb_x["dominant_sector"] = te_cb_x["dominant_sector"].astype(str)
    cb_cal_raw = cb_model.predict_proba(cal_cb_x)[:, 1]
    cb_te_raw = cb_model.predict_proba(te_cb_x)[:, 1]
    va_cb = InductiveVennAbersCalibrator(grid_size=384).fit(cb_cal_raw, y_cal)
    p_cb_va = va_cb.predict(cb_te_raw)

    # 6. Ensemble Tabular GBDT (LightGBM Monotônico + CatBoost) + Venn-Abers
    blend_cal_raw = 0.55 * mono_cal_raw + 0.45 * cb_cal_raw
    blend_te_raw = 0.55 * mono_te_raw + 0.45 * cb_te_raw
    va_blend = InductiveVennAbersCalibrator(grid_size=512).fit(blend_cal_raw, y_cal)
    blend_va_full = va_blend.predict_full(blend_te_raw)

    preds = {
        "heuristic_committee": p_heur,
        "logistic_hazard_va": p_logit,
        "lgbm_unconstrained_raw": p_unc_raw,
        "lgbm_unconstrained_platt": p_unc_platt,
        "lgbm_unconstrained_isotonic": p_unc_iso,
        "lgbm_unconstrained_va": p_unc_va,
        "lgbm_monotonic_va": p_mono_va,
        "catboost_ordered_va": p_cb_va,
        "gbdt_ensemble_va": blend_va_full.prob,
    }
    fold_metrics = {mname: compute_binary_warning_metrics(y_te, p_arr) for mname, p_arr in preds.items()}

    oof_chunk = pd.DataFrame(
        {
            "entity_id": df_te["entity_id"].values,
            "denom_social": df_te["denom_social"].values,
            "dominant_sector": df_te["dominant_sector"].values,
            "dt_comptc": df_te["dt_comptc"].values,
            "fold": split.fold.name,
            "y_true": y_te,
            "prob_heuristic": p_heur,
            "prob_logistic": p_logit,
            "prob_lgbm_unc_raw": p_unc_raw,
            "prob_lgbm_unc_platt": p_unc_platt,
            "prob_lgbm_unc_iso": p_unc_iso,
            "prob_lgbm_unc_va": p_unc_va,
            "prob_lgbm_mono_va": p_mono_va,
            "prob_catboost_va": p_cb_va,
            "raw_gbdt_blend": blend_te_raw,
            "prob_gbdt_ensemble_va": blend_va_full.prob,
            "va_p0": blend_va_full.p0,
            "va_p1": blend_va_full.p1,
            "va_width": blend_va_full.width,
        }
    )
    return oof_chunk, fold_metrics, lgb_mono, va_mono


def _run_family_ablation(panel: pd.DataFrame, splits_6m: list[TemporalSplitIndices]) -> pd.DataFrame:
    """
    Executa ablação progressiva e Leave-One-Family-Out (LOFO) nas 4 janelas walk-forward (h=6m)
    usando exatamente o mesmo protocolo LightGBM + Venn-Abers.
    """
    configs: list[tuple[str, str, list[str]]] = []
    cum_cols: list[str] = []
    family_labels = {
        "aging_structure": "Estrutura de Aging & Prazos",
        "forensic_accounting": "+ Sinais Forenses (Recompra/PDD/SCR/RJ)",
        "waterfall_liquidity": "+ Waterfall de Tranches & Liquidez",
        "originator_sector": "+ Cedentes (HHI/Churn) & Setores",
        "network_contagion": "+ Rede de Contágio LOO (Cedente/Gestor)",
        "macro_cycle": "+ Ciclo Macroeconômico BCB (Painel Completo)",
    }
    for fam_key, label in family_labels.items():
        cum_cols = cum_cols + FEATURE_FAMILIES[fam_key]
        configs.append(("progressive", label, list(cum_cols)))

    for fam_key in FEATURE_FAMILIES:
        reduced = [c for c in ALL_FEATURE_COLS if c not in FEATURE_FAMILIES[fam_key]]
        configs.append(("lofo", f"Sem {fam_key}", reduced))

    records = []
    for mode, desc, cols in configs:
        all_y = []
        all_p = []
        for sp in splits_6m:
            df_fit = panel.loc[sp.fit_mask]
            df_cal = panel.loc[sp.cal_mask]
            df_te = panel.loc[sp.test_mask]
            y_fit = df_fit["target_distress_6m"].values.astype("int32")
            y_cal = df_cal["target_distress_6m"].values.astype("int32")
            y_te = df_te["target_distress_6m"].values.astype("int32")

            m = _fit_lgbm(df_fit, y_fit, df_cal, y_cal, cols, monotonic=True)
            va = InductiveVennAbersCalibrator(grid_size=256).fit(
                np.asarray(m.predict_proba(df_cal[cols]), dtype="float64")[:, 1],
                y_cal,
            )
            p_te = va.predict(np.asarray(m.predict_proba(df_te[cols]), dtype="float64")[:, 1])
            all_y.append(y_te)
            all_p.append(p_te)

        y_cat = np.concatenate(all_y)
        p_cat = np.concatenate(all_p)
        met = compute_binary_warning_metrics(y_cat, p_cat)
        records.append(
            {
                "ablation_type": mode,
                "configuration": desc,
                "n_features": len(cols),
                **met,
            }
        )
    return pd.DataFrame(records)


def run_tabular_benchmark() -> dict[str, object]:
    ensure_directories()
    t0 = time.perf_counter()
    print("[Fase 3] Carregando painel Gold para benchmark tabular, calibração e ablações...")
    panel = pd.read_parquet(GOLD_DIR / "panel_gold.parquet")

    all_fold_rows = []
    pooled_rows = []
    bootstrap_results = {}

    for horizon in (6, 3):
        target_col = f"target_distress_{horizon}m"
        splits = build_walk_forward_splits(panel, target_col=target_col, horizon_months=horizon, eligible_only=True)
        print(f"[Fase 3] Horizonte h={horizon}m -> {len(splits)} janelas walk-forward com embargo de {horizon} meses:")

        oof_chunks = []
        last_lgbm_mono: lgb.LGBMClassifier | None = None
        last_va_mono: InductiveVennAbersCalibrator | None = None

        for sp in splits:
            print(
                f"  - {sp.fold.name} | treino até {sp.max_train_origin.date()} "
                f"(realização máx {sp.max_train_label_realization.date()}) -> teste >= {sp.min_test_origin.date()} "
                f"(n_test={int(sp.test_mask.sum())})"
            )
            oof_chunk, fmetrics, lgb_m, va_m = _evaluate_fold_models(panel, sp, target_col)
            oof_chunks.append(oof_chunk)
            last_lgbm_mono = lgb_m
            last_va_mono = va_m
            for model_name, md in fmetrics.items():
                all_fold_rows.append(
                    {
                        "horizon_months": horizon,
                        "fold": sp.fold.name,
                        "regime_label": sp.fold.regime_label,
                        "model": model_name,
                        **md,
                    }
                )

        oof_all = pd.concat(oof_chunks, ignore_index=True)
        oof_path = GOLD_DIR / f"oof_tabular_h{horizon}.parquet"
        oof_all.to_parquet(oof_path, index=False)

        model_col_map = {
            "heuristic_committee": "prob_heuristic",
            "logistic_hazard_va": "prob_logistic",
            "lgbm_unconstrained_raw": "prob_lgbm_unc_raw",
            "lgbm_unconstrained_platt": "prob_lgbm_unc_platt",
            "lgbm_unconstrained_isotonic": "prob_lgbm_unc_iso",
            "lgbm_unconstrained_va": "prob_lgbm_unc_va",
            "lgbm_monotonic_va": "prob_lgbm_mono_va",
            "catboost_ordered_va": "prob_catboost_va",
            "gbdt_ensemble_va": "prob_gbdt_ensemble_va",
        }
        for model_name, pcol in model_col_map.items():
            pm = compute_binary_warning_metrics(oof_all["y_true"].values, oof_all[pcol].values)
            pooled_rows.append({"horizon_months": horizon, "model": model_name, **pm})
            if horizon == 6 and model_name in ("lgbm_monotonic_va", "gbdt_ensemble_va"):
                print(
                    f"  ★ [Pooled h=6m] {model_name}: PR-AUC={pm['pr_auc']:.4f} | ROC-AUC={pm['roc_auc']:.4f} "
                    f"| Brier={pm['brier']:.4f} | ECE={pm['ece']:.4f} | Recall@5%={pm['recall_top5']:.3f}"
                )

        if horizon == 6:
            # Bootstrap pareado por entidade: Ensemble GBDT+VA vs Hazard Logístico e vs Heurística
            bootstrap_results["gbdt_vs_heuristic_h6"] = paired_entity_bootstrap_ci(
                oof_all["entity_id"].values,
                oof_all["y_true"].values,
                oof_all["prob_gbdt_ensemble_va"].values,
                oof_all["prob_heuristic"].values,
            )
            bootstrap_results["gbdt_vs_logistic_h6"] = paired_entity_bootstrap_ci(
                oof_all["entity_id"].values,
                oof_all["y_true"].values,
                oof_all["prob_gbdt_ensemble_va"].values,
                oof_all["prob_logistic"].values,
            )
            bootstrap_results["mono_vs_unconstrained_h6"] = paired_entity_bootstrap_ci(
                oof_all["entity_id"].values,
                oof_all["y_true"].values,
                oof_all["prob_lgbm_mono_va"].values,
                oof_all["prob_lgbm_unc_va"].values,
            )

            # Decomposição de sensibilidade do alvo (Drawdown Sênior Puro vs Salto de Inadimplência)
            te_merge = oof_all.merge(
                panel[["entity_id", "dt_comptc", "target_senior_dd_6m", "target_credit_surge_6m"]],
                on=["entity_id", "dt_comptc"],
                how="left",
            )
            bootstrap_results["subtarget_decomposition_h6"] = {
                "pure_senior_drawdown_pr_auc_gbdt": float(
                    average_precision_score(
                        te_merge["target_senior_dd_6m"].fillna(0).astype(int),
                        te_merge["prob_gbdt_ensemble_va"],
                    )
                ),
                "pure_senior_drawdown_pr_auc_heuristic": float(
                    average_precision_score(
                        te_merge["target_senior_dd_6m"].fillna(0).astype(int),
                        te_merge["prob_heuristic"],
                    )
                ),
                "credit_surge_pr_auc_gbdt": float(
                    average_precision_score(
                        te_merge["target_credit_surge_6m"].fillna(0).astype(int),
                        te_merge["prob_gbdt_ensemble_va"],
                    )
                ),
                "credit_surge_pr_auc_heuristic": float(
                    average_precision_score(
                        te_merge["target_credit_surge_6m"].fillna(0).astype(int),
                        te_merge["prob_heuristic"],
                    )
                ),
            }

            # Salvar artefato do modelo servido p/ inferência ao vivo na API FastAPI e SHAP
            if last_lgbm_mono is not None and last_va_mono is not None:
                # Reajustar um modelo final nas observações elegíveis mais recentes p/ produção ao vivo
                latest_valid = panel[(panel["is_eligible_watchlist"] == 1) & (panel["target_distress_6m"].notna())]
                ym_all = latest_valid["dt_comptc"].dt.year * 12 + latest_valid["dt_comptc"].dt.month
                max_ym_prod = int(ym_all.max())
                cal_cut = max_ym_prod - 5
                fit_cut = cal_cut - 7
                prod_fit = latest_valid[ym_all <= fit_cut]
                prod_cal = latest_valid[ym_all >= cal_cut]
                prod_lgbm = _fit_lgbm(
                    prod_fit,
                    prod_fit["target_distress_6m"].values.astype("int32"),
                    prod_cal,
                    prod_cal["target_distress_6m"].values.astype("int32"),
                    ALL_FEATURE_COLS,
                    monotonic=True,
                )
                prod_va = InductiveVennAbersCalibrator(grid_size=512).fit(
                    np.asarray(prod_lgbm.predict_proba(prod_cal[ALL_FEATURE_COLS]), dtype="float64")[:, 1],
                    prod_cal["target_distress_6m"].values.astype("int32"),
                )
                joblib.dump(
                    {
                        "lgbm_monotonic": prod_lgbm,
                        "venn_abers": prod_va,
                        "feature_cols": ALL_FEATURE_COLS,
                    },
                    MODELS_DIR / "served_tabular_h6.pkl",
                )

            print("[Fase 3] Executando estudo sistemático de ablação das 6 famílias de features (h=6m)...")
            ablation_df = _run_family_ablation(panel, splits)
            ablation_df.to_csv(METRICS_DIR / "tabular_ablation_h6.csv", index=False)

    fold_df = pd.DataFrame(all_fold_rows)
    pooled_df = pd.DataFrame(pooled_rows)
    fold_df.to_csv(METRICS_DIR / "tabular_folds_metrics.csv", index=False)
    pooled_df.to_csv(METRICS_DIR / "tabular_pooled_metrics.csv", index=False)

    summary_payload = {
        "elapsed_seconds": round(time.perf_counter() - t0, 2),
        "bootstrap_and_sensitivity": bootstrap_results,
    }
    (METRICS_DIR / "tabular_statistical_tests.json").write_text(
        json.dumps(summary_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[Fase 3] Benchmark tabular concluído em {summary_payload['elapsed_seconds']}s!")
    return summary_payload

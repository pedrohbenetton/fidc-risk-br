"""Fase 4: Matriz de Transição de Roll-Rate de Markov (Empirical Bayes) e LightGBM Quantílico + CQR."""

from __future__ import annotations

import json
import time
import warnings

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import nnls

from fidc_risk.config import GOLD_DIR, METRICS_DIR, MODELS_DIR, ensure_directories
from fidc_risk.features.gold import ALL_FEATURE_COLS
from fidc_risk.models.metrics import weighted_interval_score
from fidc_risk.models.validation import build_walk_forward_splits

QUANTILE_ALPHAS: tuple[float, ...] = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)


def _delinquency_stratum(inad90: np.ndarray) -> np.ndarray:
    """Estratifica o patamar inicial de inadimplência >90d em 4 regimes heterocedásticos."""
    s = np.zeros(len(inad90), dtype="int32")
    s[(inad90 > 1e-4) & (inad90 <= 0.02)] = 1
    s[(inad90 > 0.02) & (inad90 <= 0.08)] = 2
    s[inad90 > 0.08] = 3
    return s


def _enforce_quantile_monotonicity(q_dict: dict[float, np.ndarray]) -> dict[float, np.ndarray]:
    """Garante não-cruzamento de quantis preservando exatamente a mediana q_0.50 e limites [0, 1]."""
    med = np.clip(q_dict[0.50], 0.0, 1.0)
    out: dict[float, np.ndarray] = {0.50: med}

    # Expandir para fora a partir da mediana garantindo monotonicidade estrita
    prev_lo = med
    for a_lo in (0.25, 0.10, 0.05):
        if a_lo in q_dict:
            cur = np.clip(np.minimum(q_dict[a_lo], prev_lo), 0.0, 1.0)
            out[a_lo] = cur
            prev_lo = cur

    prev_hi = med
    for a_hi in (0.75, 0.90, 0.95):
        if a_hi in q_dict:
            cur = np.clip(np.maximum(q_dict[a_hi], prev_hi), 0.0, 1.0)
            out[a_hi] = cur
            prev_hi = cur
    return out


def _state_matrix_5(df: pd.DataFrame) -> np.ndarray:
    """
    Constrói o vetor de estado no simplex de 5 faixas de atraso por fundo-mês:
    [s0: Adimplente, s1: 1-30d, s2: 31-60d, s3: 61-90d, s4: >90d], soma = 1.
    """
    s1 = df["inad_1_30_pct"].values
    s2 = df["inad_31_60_pct"].values
    s3 = df["inad_61_90_pct"].values
    s4 = df["inad_90p_pct"].values
    inad_sum = s1 + s2 + s3 + s4
    scale = np.where(inad_sum > 1.0, 1.0 / np.maximum(inad_sum, 1e-9), 1.0)
    s1, s2, s3, s4 = s1 * scale, s2 * scale, s3 * scale, s4 * scale
    s0 = np.clip(1.0 - (s1 + s2 + s3 + s4), 0.0, 1.0)
    return np.column_stack([s0, s1, s2, s3, s4])


class EmpiricalBayesMarkovRollRate:
    """
    Estima matrizes de transição de Markov 5x5 linha-estocásticas P(S_{t+1} | S_t) via NNLS,
    combinando matriz global e matrizes por setor via encolhimento de Empirical Bayes:
        M_sec = (n_sec / (n_sec + kappa)) * M_sec_raw + (kappa / (n_sec + kappa)) * M_global.
    """

    def __init__(self, shrinkage_kappa: float = 1500.0) -> None:
        self.shrinkage_kappa = shrinkage_kappa
        self.global_matrix: np.ndarray = np.eye(5)
        self.sector_matrices: dict[str, np.ndarray] = {}

    @staticmethod
    def _estimate_transition_nnls(s_curr: np.ndarray, s_next: np.ndarray) -> np.ndarray:
        # Resolve min_{M >= 0} || s_curr @ M - s_next ||_F^2 coluna por coluna via NNLS + normaliza linhas
        mat = np.zeros((5, 5), dtype="float64")
        # Regularização leve em direção à identidade para evitar linhas degeneradas
        reg = 1e-2
        x_aug = np.vstack([s_curr, np.sqrt(reg) * np.eye(5)])
        for col_j in range(5):
            e_j = np.zeros(5)
            e_j[col_j] = np.sqrt(reg)
            y_aug = np.concatenate([s_next[:, col_j], e_j])
            sol, _ = nnls(x_aug, y_aug)
            mat[:, col_j] = sol
        row_sums = mat.sum(axis=1, keepdims=True)
        mat = np.where(row_sums > 1e-9, mat / np.maximum(row_sums, 1e-9), np.eye(5))
        return mat

    def fit(self, df_fit: pd.DataFrame) -> EmpiricalBayesMarkovRollRate:
        sub = df_fit.sort_values(["entity_id", "dt_comptc"]).copy()
        ym = sub["dt_comptc"].dt.year * 12 + sub["dt_comptc"].dt.month
        g = sub.groupby("entity_id", sort=False)
        next_ym = g["dt_comptc"].shift(-1)
        valid_1m = next_ym.notna() & ((next_ym.dt.year * 12 + next_ym.dt.month) == (ym + 1))

        s_all = _state_matrix_5(sub)
        idx_curr = np.flatnonzero(valid_1m.values)
        # Como sub está ordenado por (entity_id, dt_comptc), a linha seguinte exata é idx_curr + 1!
        s_t = s_all[idx_curr]
        s_tp1 = s_all[idx_curr + 1]
        sectors_t = sub["dominant_sector"].astype(str).values[idx_curr]

        if len(s_t) >= 50:
            self.global_matrix = self._estimate_transition_nnls(s_t, s_tp1)

        self.sector_matrices = {}
        for sec in np.unique(sectors_t):
            mask_s = sectors_t == sec
            n_s = int(mask_s.sum())
            if n_s >= 80:
                m_raw = self._estimate_transition_nnls(s_t[mask_s], s_tp1[mask_s])
                w = n_s / (n_s + self.shrinkage_kappa)
                self.sector_matrices[str(sec)] = w * m_raw + (1.0 - w) * self.global_matrix
        return self

    def predict_inad90_point(self, df: pd.DataFrame, horizon: int) -> np.ndarray:
        s_now = _state_matrix_5(df)
        sectors = df["dominant_sector"].astype(str).values
        preds = np.empty(len(df), dtype="float64")

        global_mh = np.linalg.matrix_power(self.global_matrix, horizon)
        sec_mh = {sec: np.linalg.matrix_power(m, horizon) for sec, m in self.sector_matrices.items()}

        for sec in np.unique(sectors):
            mask = sectors == sec
            mh = sec_mh.get(str(sec), global_mh)
            s_fut = s_now[mask] @ mh
            preds[mask] = np.clip(s_fut[:, 4], 0.0, 1.0)
        return preds


def _conformalize_point_forecast_stratified(
    point_cal: np.ndarray,
    y_cal: np.ndarray,
    stratum_cal: np.ndarray,
    point_te: np.ndarray,
    stratum_te: np.ndarray,
) -> dict[float, np.ndarray]:
    """Constrói intervalos empíricos calibrados por estrato de atraso em torno de uma previsão pontual."""
    res_cal = y_cal - point_cal
    q_out: dict[float, np.ndarray] = {a: np.empty(len(point_te), dtype="float64") for a in QUANTILE_ALPHAS}

    global_qs = {a: float(np.quantile(res_cal, a)) for a in QUANTILE_ALPHAS}
    for st_val in (0, 1, 2, 3):
        m_te = stratum_te == st_val
        if not m_te.any():
            continue
        m_cal = stratum_cal == st_val
        r_sub = res_cal[m_cal] if m_cal.sum() >= 50 else res_cal
        for a in QUANTILE_ALPHAS:
            offset = 0.0 if a == 0.50 else float(np.quantile(r_sub, a))
            if m_cal.sum() < 50:
                offset = 0.0 if a == 0.50 else global_qs[a]
            q_out[a][m_te] = np.clip(point_te[m_te] + offset, 0.0, 1.0)
    return _enforce_quantile_monotonicity(q_out)


class StratifiedCQRCalibrator:
    """
    Conformalized Quantile Regression (Romano, Patterson & Candès, 2019) estratificada por
    nível inicial de inadimplência, preservando a mediana e garantindo validade de cobertura finita.
    """

    def __init__(self) -> None:
        self.margins: dict[tuple[float, int], float] = {}
        self.global_margins: dict[float, float] = {}

    def fit(
        self,
        q_cal: dict[float, np.ndarray],
        y_cal: np.ndarray,
        stratum_cal: np.ndarray,
    ) -> StratifiedCQRCalibrator:
        pairs = [(0.05, 0.95, 0.10), (0.10, 0.90, 0.20), (0.25, 0.75, 0.50)]
        for lo_a, hi_a, miscov in pairs:
            scores = np.maximum(q_cal[lo_a] - y_cal, y_cal - q_cal[hi_a])
            n_all = len(scores)
            lvl_all = min(np.ceil((n_all + 1) * (1.0 - miscov)) / max(n_all, 1), 1.0)
            self.global_margins[miscov] = float(np.quantile(scores, lvl_all))

            for st_val in (0, 1, 2, 3):
                m = stratum_cal == st_val
                n_g = int(m.sum())
                if n_g >= 60:
                    lvl_g = min(np.ceil((n_g + 1) * (1.0 - miscov)) / n_g, 1.0)
                    self.margins[(miscov, st_val)] = float(np.quantile(scores[m], lvl_g))
                else:
                    self.margins[(miscov, st_val)] = self.global_margins[miscov]
        return self

    def apply(
        self,
        q_raw: dict[float, np.ndarray],
        stratum: np.ndarray,
    ) -> dict[float, np.ndarray]:
        out: dict[float, np.ndarray] = {0.50: np.clip(q_raw[0.50], 0.0, 1.0)}
        pairs = [(0.05, 0.95, 0.10), (0.10, 0.90, 0.20), (0.25, 0.75, 0.50)]
        for lo_a, hi_a, miscov in pairs:
            adj = np.array(
                [self.margins.get((miscov, int(s)), self.global_margins[miscov]) for s in stratum],
                dtype="float64",
            )
            out[lo_a] = np.clip(q_raw[lo_a] - adj, 0.0, 1.0)
            out[hi_a] = np.clip(q_raw[hi_a] + adj, 0.0, 1.0)
        return _enforce_quantile_monotonicity(out)


def _fit_quantile_lgbm_suite(
    df_fit: pd.DataFrame,
    y_fit: np.ndarray,
    df_cal: pd.DataFrame,
    y_cal: np.ndarray,
) -> dict[float, lgb.LGBMRegressor]:
    models: dict[float, lgb.LGBMRegressor] = {}
    for alpha in QUANTILE_ALPHAS:
        reg = lgb.LGBMRegressor(
            objective="quantile",
            alpha=alpha,
            n_estimators=550,
            learning_rate=0.03,
            num_leaves=31,
            max_depth=6,
            min_child_samples=100,
            subsample=0.80,
            subsample_freq=1,
            colsample_bytree=0.75,
            reg_alpha=1.0,
            reg_lambda=4.0,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            reg.fit(
                df_fit[ALL_FEATURE_COLS],
                y_fit,
                eval_set=[(df_cal[ALL_FEATURE_COLS], y_cal)],
                callbacks=[lgb.early_stopping(stopping_rounds=45, verbose=False)],
            )
        models[alpha] = reg
    return models


def run_rollrate_and_cqr_benchmark() -> dict[str, object]:
    ensure_directories()
    t0 = time.perf_counter()
    print("[Fase 4] Carregando painel Gold para benchmark de Roll-Rate Markov e LightGBM Quantílico + CQR...")
    panel = pd.read_parquet(GOLD_DIR / "panel_gold.parquet")
    # Filtramos fundos comerciais ativos (não FIDC-NP estruturais e carteira >= R$ 2M)
    panel_com = panel[(panel["is_fidc_np_structural"] == 0) & (panel["vl_carteira"] >= 2_000_000.0)].copy()

    all_rows = []
    fold_rows = []
    saved_h3_oof = []
    last_markov: EmpiricalBayesMarkovRollRate | None = None
    last_qmodels_h3: dict[float, lgb.LGBMRegressor] | None = None
    last_cqr_h3: StratifiedCQRCalibrator | None = None

    for horizon in (1, 3, 6):
        target_col = f"target_inad90p_h{horizon}"
        splits = build_walk_forward_splits(
            panel_com,
            target_col=target_col,
            horizon_months=horizon,
            eligible_only=False,
        )
        print(f"[Fase 4] Horizonte h={horizon}m -> {len(splits)} janelas walk-forward:")

        pooled_y = []
        pooled_preds: dict[str, dict[float, list[np.ndarray]]] = {
            m: {a: [] for a in QUANTILE_ALPHAS}
            for m in (
                "persistence_global",
                "persistence_stratified",
                "markov_rollrate_eb",
                "lgbm_quantile_raw",
                "lgbm_cqr_stratified",
            )
        }

        for sp in splits:
            df_fit = panel_com.loc[sp.fit_mask]
            df_cal = panel_com.loc[sp.cal_mask]
            df_te = panel_com.loc[sp.test_mask]

            y_fit = df_fit[target_col].values.astype("float64")
            y_cal = df_cal[target_col].values.astype("float64")
            y_te = df_te[target_col].values.astype("float64")

            st_cal = _delinquency_stratum(df_cal["inad_90p_pct"].values)
            st_te = _delinquency_stratum(df_te["inad_90p_pct"].values)

            # 1. Persistência Global (sem estratificação)
            q_pers_glob = _conformalize_point_forecast_stratified(
                df_cal["inad_90p_pct"].values,
                y_cal,
                np.zeros_like(st_cal),
                df_te["inad_90p_pct"].values,
                np.zeros_like(st_te),
            )

            # 2. Persistência Estratificada por Patamar de Atraso (Baseline Forte de Referência)
            q_pers_strat = _conformalize_point_forecast_stratified(
                df_cal["inad_90p_pct"].values,
                y_cal,
                st_cal,
                df_te["inad_90p_pct"].values,
                st_te,
            )

            # 3. Matriz de Roll-Rate de Markov c/ Empirical Bayes Setorial
            markov = EmpiricalBayesMarkovRollRate(shrinkage_kappa=1500.0).fit(df_fit)
            mk_cal = markov.predict_inad90_point(df_cal, horizon=horizon)
            mk_te = markov.predict_inad90_point(df_te, horizon=horizon)
            q_markov = _conformalize_point_forecast_stratified(mk_cal, y_cal, st_cal, mk_te, st_te)

            # 4. LightGBM Quantílico Global Bruto (c/ não-cruzamento)
            q_models = _fit_quantile_lgbm_suite(df_fit, y_fit, df_cal, y_cal)
            q_cal_raw = _enforce_quantile_monotonicity(
                {a: np.asarray(q_models[a].predict(df_cal[ALL_FEATURE_COLS]), dtype="float64") for a in QUANTILE_ALPHAS}
            )
            q_te_raw = _enforce_quantile_monotonicity(
                {a: np.asarray(q_models[a].predict(df_te[ALL_FEATURE_COLS]), dtype="float64") for a in QUANTILE_ALPHAS}
            )

            # 5. LightGBM + CQR Estratificado por Patamar de Inadimplência
            cqr = StratifiedCQRCalibrator().fit(q_cal_raw, y_cal, st_cal)
            q_te_cqr = cqr.apply(q_te_raw, st_te)

            if horizon == 3:
                last_markov = markov
                last_qmodels_h3 = q_models
                last_cqr_h3 = cqr
                saved_h3_oof.append(
                    pd.DataFrame(
                        {
                            "entity_id": df_te["entity_id"].values,
                            "dt_comptc": df_te["dt_comptc"].values,
                            "fold": sp.fold.name,
                            "inad_now": df_te["inad_90p_pct"].values,
                            "y_true_h3": y_te,
                            "markov_h3_q50": q_markov[0.50],
                            "cqr_h3_q05": q_te_cqr[0.05],
                            "cqr_h3_q10": q_te_cqr[0.10],
                            "cqr_h3_q25": q_te_cqr[0.25],
                            "cqr_h3_q50": q_te_cqr[0.50],
                            "cqr_h3_q75": q_te_cqr[0.75],
                            "cqr_h3_q90": q_te_cqr[0.90],
                            "cqr_h3_q95": q_te_cqr[0.95],
                        }
                    )
                )

            fold_map = {
                "persistence_global": q_pers_glob,
                "persistence_stratified": q_pers_strat,
                "markov_rollrate_eb": q_markov,
                "lgbm_quantile_raw": q_te_raw,
                "lgbm_cqr_stratified": q_te_cqr,
            }
            pooled_y.append(y_te)
            for mname, qd in fold_map.items():
                wm = weighted_interval_score(y_te, qd)
                fold_rows.append(
                    {
                        "horizon_months": horizon,
                        "fold": sp.fold.name,
                        "model": mname,
                        **wm,
                    }
                )
                for a in QUANTILE_ALPHAS:
                    pooled_preds[mname][a].append(qd[a])

        y_all = np.concatenate(pooled_y)
        for mname, q_lists in pooled_preds.items():
            q_cat = {a: np.concatenate(q_lists[a]) for a in QUANTILE_ALPHAS}
            wm = weighted_interval_score(y_all, q_cat)
            all_rows.append({"horizon_months": horizon, "model": mname, "n_obs": len(y_all), **wm})
            if mname in ("persistence_stratified", "markov_rollrate_eb", "lgbm_quantile_raw", "lgbm_cqr_stratified"):
                print(
                    f"  -> [h={horizon}m] {mname:23s}: WIS={wm['wis']*100:.3f} p.p. | MAE={wm['mae']*100:.3f} p.p. "
                    f"| Cob90={wm['coverage_90']*100:.1f}% | Larg90={wm['mean_width_90']*100:.2f} p.p."
                )

    if saved_h3_oof:
        pd.concat(saved_h3_oof, ignore_index=True).to_parquet(GOLD_DIR / "oof_cqr_h3.parquet", index=False)

    if last_markov is not None and last_qmodels_h3 is not None and last_cqr_h3 is not None:
        joblib.dump(
            {
                "markov_global_matrix": last_markov.global_matrix,
                "markov_sector_matrices": last_markov.sector_matrices,
                "qmodels_h3": last_qmodels_h3,
                "cqr_h3": last_cqr_h3,
            },
            MODELS_DIR / "served_cqr_h3.pkl",
        )
        # Exportar matriz de transição global p/ inspeção forense e web
        pd.DataFrame(
            last_markov.global_matrix,
            index=["Adimplente", "1-30d", "31-60d", "61-90d", ">90d"],
            columns=["Adimplente", "1-30d", "31-60d", "61-90d", ">90d"],
        ).to_csv(METRICS_DIR / "markov_global_transition_matrix.csv")

    pd.DataFrame(all_rows).to_csv(METRICS_DIR / "rollrate_cqr_pooled_metrics.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(METRICS_DIR / "rollrate_cqr_fold_metrics.csv", index=False)

    summary: dict[str, object] = {"elapsed_seconds": round(time.perf_counter() - t0, 2), "rows": len(all_rows)}
    (METRICS_DIR / "rollrate_cqr_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[Fase 4] Concluída em {summary['elapsed_seconds']}s!")
    return summary

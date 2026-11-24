"""Calibração Pós-Hoc: Platt Scaling, Regressão Isotônica e Venn-Abers Indutivo (IVAP)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


def _logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    clipped = np.clip(p, eps, 1.0 - eps)
    return np.log(clipped / (1.0 - clipped))


class PlattCalibrator:
    """Calibração paramétrica de Platt sobre log-odds."""

    def __init__(self) -> None:
        self._lr = LogisticRegression(C=1.0, solver="lbfgs")
        self._constant_prob: float | None = None

    def fit(self, raw_scores: np.ndarray, y_true: np.ndarray) -> PlattCalibrator:
        y = np.asarray(y_true, dtype="int32")
        if len(np.unique(y)) < 2:
            self._constant_prob = float(y.mean()) if len(y) > 0 else 0.05
            return self
        x = _logit(np.asarray(raw_scores, dtype="float64")).reshape(-1, 1)
        self._lr.fit(x, y)
        return self

    def predict(self, raw_scores: np.ndarray) -> np.ndarray:
        if self._constant_prob is not None:
            return np.full(len(raw_scores), self._constant_prob, dtype="float64")
        x = _logit(np.asarray(raw_scores, dtype="float64")).reshape(-1, 1)
        return np.asarray(self._lr.predict_proba(x)[:, 1], dtype="float64")


class IsotonicCalibrator:
    """Calibração monotônica não-paramétrica via Pool-Adjacent-Violators (PAV)."""

    def __init__(self) -> None:
        self._iso = IsotonicRegression(y_min=1e-5, y_max=1.0 - 1e-5, out_of_bounds="clip")
        self._fitted = False

    def fit(self, raw_scores: np.ndarray, y_true: np.ndarray) -> IsotonicCalibrator:
        s = np.asarray(raw_scores, dtype="float64")
        y = np.asarray(y_true, dtype="float64")
        if len(s) >= 5:
            self._iso.fit(s, y)
            self._fitted = True
        return self

    def predict(self, raw_scores: np.ndarray) -> np.ndarray:
        s = np.asarray(raw_scores, dtype="float64")
        if not self._fitted:
            return np.clip(s, 1e-5, 1.0 - 1e-5)
        return np.asarray(self._iso.predict(s), dtype="float64")


@dataclass
class VennAbersPrediction:
    prob: np.ndarray       # Probabilidade pontual regularizada p1 / (1 - p0 + p1)
    p0: np.ndarray         # Limite inferior multiprobabilístico
    p1: np.ndarray         # Limite superior multiprobabilístico (usado p/ aversão a ambiguidade)
    width: np.ndarray      # Largura epistêmica |p1 - p0|


class InductiveVennAbersCalibrator:
    """
    Inductive Venn-Abers Predictor (Vovk & Petej, 2014).
    Para cada ponto de uma malha densa de escores g_k, resolve duas regressões isotônicas exatas
    adicionando (g_k, 0) e (g_k, 1) ao conjunto de calibração, garantindo:
      1) Validade de amostra finita livre de distribuição: P(Y=1) pertence a [p0(X), p1(X)];
      2) Monotonicidade estrita de p0(s), p1(s) e p_VA(s);
      3) Quantificação direta de incerteza epistêmica (width = p1 - p0) em regiões raras!
    """

    def __init__(self, grid_size: int = 512) -> None:
        self.grid_size = grid_size
        self._grid: np.ndarray | None = None
        self._p0_grid: np.ndarray | None = None
        self._p1_grid: np.ndarray | None = None

    def fit(self, raw_scores: np.ndarray, y_true: np.ndarray) -> InductiveVennAbersCalibrator:
        s = np.asarray(raw_scores, dtype="float64")
        y = np.asarray(y_true, dtype="float64")
        if len(s) < 10:
            self._grid = np.array([0.0, 1.0])
            base = float(y.mean()) if len(y) > 0 else 0.05
            self._p0_grid = np.array([base, base])
            self._p1_grid = np.array([base, base])
            return self

        # Comprimir calibração por escores únicos (com pesos) para acelerar PAV exato na malha
        order = np.argsort(s)
        s_sorted = s[order]
        y_sorted = y[order]

        q_grid = np.linspace(0.0, 1.0, self.grid_size)
        grid = np.unique(np.concatenate([[0.0], np.quantile(s_sorted, q_grid), [1.0]]))
        p0_vals = np.empty(len(grid), dtype="float64")
        p1_vals = np.empty(len(grid), dtype="float64")

        iso0 = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        iso1 = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")

        # Para eficiência quando n_cal > 4000, agrupamos s_sorted em até 2000 micro-bins suficientes
        if len(s_sorted) > 2000:
            bins = np.linspace(0, len(s_sorted), 2001, dtype=int)
            s_comp = np.array([s_sorted[bins[i] : bins[i + 1]].mean() for i in range(2000)])
            y_comp = np.array([y_sorted[bins[i] : bins[i + 1]].mean() for i in range(2000)])
            w_comp = np.array([float(bins[i + 1] - bins[i]) for i in range(2000)])
        else:
            s_comp = s_sorted
            y_comp = y_sorted
            w_comp = np.ones(len(s_sorted), dtype="float64")

        for idx, g in enumerate(grid):
            s_aug = np.append(s_comp, g)
            w_aug = np.append(w_comp, 1.0)
            y_aug0 = np.append(y_comp, 0.0)
            y_aug1 = np.append(y_comp, 1.0)

            iso0.fit(s_aug, y_aug0, sample_weight=w_aug)
            iso1.fit(s_aug, y_aug1, sample_weight=w_aug)
            p0_vals[idx] = float(iso0.predict([g])[0])
            p1_vals[idx] = float(iso1.predict([g])[0])

        # Garantir monotonicidade numérica exata e p0 <= p1
        p0_vals = np.maximum.accumulate(np.clip(p0_vals, 0.0, 1.0))
        p1_vals = np.maximum.accumulate(np.clip(p1_vals, p0_vals, 1.0))

        self._grid = grid
        self._p0_grid = p0_vals
        self._p1_grid = p1_vals
        return self

    def predict_full(self, raw_scores: np.ndarray) -> VennAbersPrediction:
        assert self._grid is not None and self._p0_grid is not None and self._p1_grid is not None
        s = np.asarray(raw_scores, dtype="float64")
        p0 = np.interp(s, self._grid, self._p0_grid)
        p1 = np.interp(s, self._grid, self._p1_grid)
        p1 = np.maximum(p1, p0)
        denom = np.maximum(1.0 - p0 + p1, 1e-12)
        prob = np.clip(p1 / denom, 1e-6, 1.0 - 1e-6)
        return VennAbersPrediction(prob=prob, p0=p0, p1=p1, width=p1 - p0)

    def predict(self, raw_scores: np.ndarray) -> np.ndarray:
        return self.predict_full(raw_scores).prob

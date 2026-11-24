"""Métricas probabilísticas, calibração, ranking de watchlist e escores quantílicos (WIS)."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Calcula o ECE (Expected Calibration Error) em quantis de probabilidade predita."""
    y_t = np.asarray(y_true, dtype="float64")
    y_p = np.clip(np.asarray(y_prob, dtype="float64"), 0.0, 1.0)
    if len(y_t) == 0:
        return 0.0
    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.unique(np.quantile(y_p, quantiles))
    if len(edges) <= 2:
        return float(abs(y_t.mean() - y_p.mean()))
    bin_ids = np.digitize(y_p, edges[1:-1], right=True)
    ece = 0.0
    n = float(len(y_t))
    for b in range(len(edges) - 1):
        mask = bin_ids == b
        cnt = int(mask.sum())
        if cnt > 0:
            ece += (cnt / n) * abs(float(y_t[mask].mean()) - float(y_p[mask].mean()))
    return float(ece)


def watchlist_topk_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    top_frac: float = 0.05,
) -> dict[str, float]:
    """Avalia a eficácia operacional de um comitê que audita os top-k% fundos mais arriscados."""
    y_t = np.asarray(y_true, dtype="float64")
    y_s = np.asarray(y_score, dtype="float64")
    n = len(y_t)
    total_pos = float(y_t.sum())
    if n == 0 or total_pos == 0:
        return {"recall_top5": 0.0, "precision_top5": 0.0, "lift_top5": 0.0}
    k = max(1, int(np.ceil(n * top_frac)))
    top_idx = np.argpartition(-y_s, kth=min(k - 1, n - 1))[:k]
    caught = float(y_t[top_idx].sum())
    recall = caught / total_pos
    precision = caught / float(k)
    base_rate = total_pos / float(n)
    lift = precision / max(base_rate, 1e-12)
    return {
        "recall_top5": float(recall),
        "precision_top5": float(precision),
        "lift_top5": float(lift),
    }


def compute_binary_warning_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    """Pacote completo de métricas para alerta precoce binário probabilístico."""
    y_t = np.asarray(y_true, dtype="int32")
    y_p = np.clip(np.asarray(y_prob, dtype="float64"), 1e-6, 1.0 - 1e-6)
    has_both = len(np.unique(y_t)) >= 2
    pr_auc = float(average_precision_score(y_t, y_p)) if has_both else 0.0
    roc_auc = float(roc_auc_score(y_t, y_p)) if has_both else 0.5
    brier = float(brier_score_loss(y_t, y_p))
    ll = float(log_loss(y_t, y_p, labels=[0, 1]))
    ece = expected_calibration_error(y_t, y_p, n_bins=10)
    top_m = watchlist_topk_metrics(y_t, y_p, top_frac=0.05)
    return {
        "n_obs": int(len(y_t)),
        "event_rate": float(y_t.mean()),
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "brier": brier,
        "log_loss": ll,
        "ece": ece,
        **top_m,
    }


def paired_entity_bootstrap_ci(
    entity_ids: np.ndarray,
    y_true: np.ndarray,
    prob_a: np.ndarray,
    prob_b: np.ndarray,
    n_boot: int = 400,
    seed: int = 42,
) -> dict[str, float]:
    """
    Bootstrap agrupado por entidade (cluster bootstrap) para testar a diferença pareada
    de PR-AUC e Brier Score entre o modelo A e o modelo B de referência.
    """
    rng = np.random.default_rng(seed)
    uniq_entities, inv = np.unique(entity_ids, return_inverse=True)
    # Pré-indexar linhas por entidade para reamostragem O(N)
    order = np.argsort(inv)
    sorted_inv = inv[order]
    splits = np.flatnonzero(np.diff(sorted_inv)) + 1
    groups = np.split(order, splits)
    n_ent = len(uniq_entities)

    pr_diffs = []
    brier_diffs = []
    for _ in range(n_boot):
        sampled_g = rng.integers(0, n_ent, size=n_ent)
        idx = np.concatenate([groups[g] for g in sampled_g])
        yt = y_true[idx]
        if len(np.unique(yt)) < 2:
            continue
        pa = prob_a[idx]
        pb = prob_b[idx]
        pr_diffs.append(float(average_precision_score(yt, pa) - average_precision_score(yt, pb)))
        brier_diffs.append(float(brier_score_loss(yt, pa) - brier_score_loss(yt, pb)))

    arr_pr = np.asarray(pr_diffs)
    arr_br = np.asarray(brier_diffs)
    return {
        "pr_auc_diff_mean": float(arr_pr.mean()),
        "pr_auc_diff_ci95_low": float(np.quantile(arr_pr, 0.025)),
        "pr_auc_diff_ci95_high": float(np.quantile(arr_pr, 0.975)),
        "pr_auc_p_value_one_sided": float(np.mean(arr_pr <= 0.0)),
        "brier_diff_mean": float(arr_br.mean()),
        "brier_diff_ci95_low": float(np.quantile(arr_br, 0.025)),
        "brier_diff_ci95_high": float(np.quantile(arr_br, 0.975)),
    }


def pinball_loss(y_true: np.ndarray, y_q: np.ndarray, alpha: float) -> float:
    diff = y_true - y_q
    return float(np.mean(np.maximum(alpha * diff, (alpha - 1.0) * diff)))


def weighted_interval_score(
    y_true: np.ndarray,
    quantiles_dict: dict[float, np.ndarray],
) -> dict[str, float]:
    """
    Calcula o Weighted Interval Score (WIS — Bracher et al., 2021) sobre quantis simétricos
    (ex.: {0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95}) além de MAE e coberturas empíricas.
    """
    y = np.asarray(y_true, dtype="float64")
    q50 = np.asarray(quantiles_dict[0.50], dtype="float64")
    pairs = [(0.05, 0.95, 0.10), (0.10, 0.90, 0.20), (0.25, 0.75, 0.50)]
    avail_pairs = [(lo, hi, a) for lo, hi, a in pairs if lo in quantiles_dict and hi in quantiles_dict]

    wis_acc = 0.5 * np.abs(y - q50)
    for lo, hi, alpha in avail_pairs:
        l_arr = np.minimum(quantiles_dict[lo], quantiles_dict[hi])
        u_arr = np.maximum(quantiles_dict[lo], quantiles_dict[hi])
        interval_score = (
            (u_arr - l_arr)
            + (2.0 / alpha) * np.maximum(l_arr - y, 0.0)
            + (2.0 / alpha) * np.maximum(y - u_arr, 0.0)
        )
        wis_acc = wis_acc + 0.5 * alpha * interval_score

    denom = len(avail_pairs) + 0.5
    wis = float(np.mean(wis_acc / denom))
    mae = float(np.mean(np.abs(y - q50)))

    cov90 = (
        float(np.mean((y >= quantiles_dict[0.05]) & (y <= quantiles_dict[0.95])))
        if 0.05 in quantiles_dict and 0.95 in quantiles_dict
        else np.nan
    )
    cov80 = (
        float(np.mean((y >= quantiles_dict[0.10]) & (y <= quantiles_dict[0.90])))
        if 0.10 in quantiles_dict and 0.90 in quantiles_dict
        else np.nan
    )
    width90 = (
        float(np.mean(quantiles_dict[0.95] - quantiles_dict[0.05]))
        if 0.05 in quantiles_dict and 0.95 in quantiles_dict
        else np.nan
    )
    return {
        "wis": wis,
        "mae": mae,
        "coverage_80": cov80,
        "coverage_90": cov90,
        "mean_width_90": width90,
    }

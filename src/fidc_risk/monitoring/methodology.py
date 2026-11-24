"""Evidência para a página de Metodologia: validação fora da amostra por safra, calibração e erros.

Tudo é calculado sobre as previsões fora da amostra (out-of-fold) do mesmo modelo que gera o score ao
vivo (LightGBM monotônico + calibração Venn-Abers, coluna ``prob_lgbm_mono_va``). Cada safra (ano da
data da previsão) foi prevista por um modelo treinado só com dados anteriores, com intervalo de h meses
entre o fim do treino e o início do teste.

Só entram previsões cuja janela futura de h meses está toda nos dados: nos meses mais recentes o evento
ainda pode acontecer, e contá-los como "não houve estresse" infla o acerto aparente.
"""

from __future__ import annotations

import json
import math
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from fidc_risk.config import BRONZE_DIR, GOLD_DIR, METRICS_DIR
from fidc_risk.features import gold
from fidc_risk.models.metrics import expected_calibration_error

SERVED_SCORE_COL = "prob_lgbm_mono_va"
HORIZON_MONTHS = 6
MESES_ABREV = ["jan", "fev", "mar", "abr", "mai", "jun", "jul", "ago", "set", "out", "nov", "dez"]


def _ym(d: pd.Series) -> pd.Series:
    return d.dt.year * 12 + d.dt.month


def complete_window_mask(dt: pd.Series, last_competence: pd.Timestamp, horizon: int = HORIZON_MONTHS) -> pd.Series:
    """True quando os h meses seguintes a t estão todos dentro dos dados (até ``last_competence``)."""
    last = last_competence.year * 12 + last_competence.month
    return (last - _ym(pd.to_datetime(dt))) >= horizon


def top_k_mask(scores: np.ndarray, frac: float) -> np.ndarray:
    """Marca os ceil(n * frac) maiores scores (empates resolvidos pela ordem de entrada)."""
    s = np.asarray(scores, dtype="float64")
    n = len(s)
    out = np.zeros(n, dtype=bool)
    if n == 0:
        return out
    k = max(1, math.ceil(n * frac - 1e-12))
    out[np.argsort(-s, kind="stable")[:k]] = True
    return out


def calibration_bins(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> list[dict[str, float]]:
    """Faixas de score com o mesmo número de previsões: score médio previsto vs. frequência observada."""
    y_a = np.asarray(y, dtype="float64")
    p_a = np.asarray(p, dtype="float64")
    if len(p_a) == 0:
        return []
    edges = np.unique(np.quantile(p_a, np.linspace(0.0, 1.0, n_bins + 1)))
    ids = np.digitize(p_a, edges[1:-1]) if len(edges) > 2 else np.zeros(len(p_a), dtype=int)
    out = []
    for b in range(max(len(edges) - 1, 1)):
        m = ids == b
        if m.any():
            out.append({"previsto": float(p_a[m].mean()), "observado": float(y_a[m].mean()), "n": int(m.sum())})
    return out


def _safra_label(ano: int, meses: pd.Series) -> str:
    lo, hi = int(meses.min()), int(meses.max())
    if lo == 1 and hi == 12:
        return str(ano)
    return f"{ano} ({MESES_ABREV[lo - 1]}–{MESES_ABREV[hi - 1]})"


def _evaluable(oof: pd.DataFrame, last_competence: pd.Timestamp, horizon: int) -> pd.DataFrame:
    df = oof.copy()
    df["dt_comptc"] = pd.to_datetime(df["dt_comptc"])
    df = df[complete_window_mask(df["dt_comptc"], last_competence, horizon)].copy()
    df["ano"] = df["dt_comptc"].dt.year
    return df


def vintage_metrics(
    oof: pd.DataFrame,
    score_col: str,
    last_competence: pd.Timestamp,
    horizon: int = HORIZON_MONTHS,
    top_frac: float = 0.05,
) -> pd.DataFrame:
    """Uma linha por safra (ano): volume, eventos, ordenação, concentração no topo e erro de calibração."""
    df = _evaluable(oof, last_competence, horizon)
    rows = []
    for ano, g in df.groupby("ano", sort=True):
        y = g["y_true"].to_numpy(dtype="float64")
        p = g[score_col].to_numpy(dtype="float64")
        top = top_k_mask(p, top_frac)
        base = float(y.mean())
        prec_top = float(y[top].mean())
        rows.append(
            {
                "ano": int(ano),
                "rotulo": _safra_label(int(ano), g["dt_comptc"].dt.month),
                "inicio": str(g["dt_comptc"].min().date())[:7],
                "fim": str(g["dt_comptc"].max().date())[:7],
                "n_obs": int(len(g)),
                "eventos": int(y.sum()),
                "taxa_eventos": base,
                "score_medio": float(p.mean()),
                "acerto_ordenacao": float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else float("nan"),
                "concentracao_top5": prec_top / base if base > 0 else float("nan"),
                "eventos_no_top5": float(y[top].sum() / y.sum()) if y.sum() > 0 else float("nan"),
                "erro_calibracao": float(expected_calibration_error(y, p, n_bins=10)),
            }
        )
    return pd.DataFrame(rows)


def _flag_topk_by_year(df: pd.DataFrame, score_col: str, frac: float) -> pd.Series:
    flags = pd.Series(False, index=df.index)
    for _, g in df.groupby("ano"):
        flags.loc[g.index] = top_k_mask(g[score_col].to_numpy(), frac)
    return flags


def error_analysis(
    oof: pd.DataFrame,
    score_col: str,
    last_competence: pd.Timestamp,
    horizon: int = HORIZON_MONTHS,
    min_eventos_segmento: int = 30,
) -> dict[str, Any]:
    """Onde o modelo erra, só em agregados: eventos fora do radar e alarmes falsos, por segmento."""
    df = _evaluable(oof, last_competence, horizon)
    df["top5"] = _flag_topk_by_year(df, score_col, 0.05)
    df["top20"] = _flag_topk_by_year(df, score_col, 0.20)
    df["abaixo_mediana"] = df[score_col] < df.groupby("ano")[score_col].transform("median")
    ev = df[df["y_true"] == 1]

    total = {
        "eventos": int(len(ev)),
        "eventos_no_top5": float(ev["top5"].mean()) if len(ev) else float("nan"),
        "eventos_no_top20": float(ev["top20"].mean()) if len(ev) else float("nan"),
        "eventos_abaixo_da_mediana": float(ev["abaixo_mediana"].mean()) if len(ev) else float("nan"),
        "alarmes_falsos_top5": float(1.0 - df.loc[df["top5"], "y_true"].mean()) if df["top5"].any() else float("nan"),
    }

    seg = df["dominant_sector"].fillna("Não classificado").astype(str)
    n_ev = ev.groupby(seg.loc[ev.index]).size()
    pequenos = set(n_ev[n_ev < min_eventos_segmento].index) | (set(seg.unique()) - set(n_ev.index))
    df["segmento"] = np.where(seg.isin(pequenos), "Outros", seg)
    por_segmento: list[dict[str, Any]] = []
    for nome, g in df.groupby("segmento", sort=False):
        e = g[g["y_true"] == 1]
        if len(e) == 0:
            continue
        por_segmento.append(
            {
                "segmento": str(nome),
                "n_obs": int(len(g)),
                "eventos": int(len(e)),
                "taxa_eventos": float(g["y_true"].mean()),
                "eventos_no_top20": float(e["top20"].mean()),
                "eventos_abaixo_da_mediana": float(e["abaixo_mediana"].mean()),
            }
        )
    por_segmento.sort(key=lambda r: (r["segmento"] == "Outros", -r["eventos"]))
    return {"total": total, "por_segmento": por_segmento}


def error_profile(
    oof: pd.DataFrame,
    panel: pd.DataFrame,
    score_col: str,
    last_competence: pd.Timestamp,
    horizon: int = HORIZON_MONTHS,
) -> dict[str, Any]:
    """Compara eventos vistos (top 20% do ano) e não vistos: gatilho do evento e situação do fundo em t."""
    df = _evaluable(oof, last_competence, horizon)
    df["top20"] = _flag_topk_by_year(df, score_col, 0.20)
    cols = [
        "entity_id", "dt_comptc", f"target_senior_dd_{horizon}m", f"target_credit_surge_{horizon}m",
        "inad_90p_pct", "inad_early_1_60_pct", "subordinacao_pct",
    ]
    pan = panel[cols].copy()
    pan["dt_comptc"] = pd.to_datetime(pan["dt_comptc"])
    ev = df[df["y_true"] == 1].merge(pan, on=["entity_id", "dt_comptc"], how="left")
    out: dict[str, Any] = {}
    for nome, g in (("vistos", ev[ev["top20"]]), ("nao_vistos", ev[~ev["top20"]])):
        sr = g[f"target_senior_dd_{horizon}m"].fillna(0) > 0
        cs = g[f"target_credit_surge_{horizon}m"].fillna(0) > 0
        n = max(len(g), 1)
        out[nome] = {
            "eventos": int(len(g)),
            "so_queda_cota_senior": float((sr & ~cs).sum() / n),
            "so_salto_inadimplencia": float((cs & ~sr).sum() / n),
            "ambos": float((sr & cs).sum() / n),
            "mediana_vencido90_em_t": float(g["inad_90p_pct"].median()),
            "mediana_atraso_1_60_em_t": float(g["inad_early_1_60_pct"].median()),
            "mediana_subordinacao_em_t": float(g["subordinacao_pct"].median()),
        }
    return out


def _ym_label(ym: int) -> str:
    y, m = divmod(int(ym) - 1, 12)
    return f"{y:04d}-{m + 1:02d}"


def served_holdout_months(ultimo_com_rotulo: int) -> tuple[list[str], tuple[str, str]]:
    """Meses que o modelo servido não viu (entre o fim do treino e o início da calibração) e a janela de
    calibração, replicando o corte de models/tabular_suite.py: calibra nos 6 últimos meses com rótulo e
    treina até 7 meses antes do início da calibração."""
    cal_cut = ultimo_com_rotulo - 5
    fit_cut = cal_cut - 7
    fora = [_ym_label(ym) for ym in range(fit_cut + 1, cal_cut)]
    return fora, (_ym_label(cal_cut), _ym_label(ultimo_com_rotulo))


def served_level_check(df: pd.DataFrame, meses: list[str], last_competence: pd.Timestamp,
                       horizon: int = HORIZON_MONTHS) -> dict[str, Any]:
    """Score médio do modelo servido contra a frequência real, nos meses dados que já têm h meses de futuro."""
    d = df.copy()
    d["dt_comptc"] = pd.to_datetime(d["dt_comptc"])
    d["mes"] = d["dt_comptc"].dt.strftime("%Y-%m")
    d = d[d["mes"].isin(meses) & complete_window_mask(d["dt_comptc"], last_competence, horizon)]
    por_mes = [
        {"mes": m, "n_obs": int(len(g)), "score_medio": float(g["score"].mean()), "taxa_eventos": float(g["y_true"].mean())}
        for m, g in d.groupby("mes", sort=True)
    ]
    return {
        "por_mes": por_mes,
        "n_obs": int(len(d)),
        "score_medio": float(d["score"].mean()) if len(d) else float("nan"),
        "taxa_eventos": float(d["y_true"].mean()) if len(d) else float("nan"),
    }


def served_model_level() -> dict[str, Any] | None:
    """Checa o nível do score em produção nos meses que o modelo servido não viu (com 6 meses de resultado)."""
    import joblib

    from fidc_risk.config import MODELS_DIR

    model_path = MODELS_DIR / "served_tabular_h6.pkl"
    if not model_path.exists():
        return None
    served = joblib.load(model_path)
    painel = pd.read_parquet(GOLD_DIR / "panel_gold.parquet")
    elig = painel[(painel["is_eligible_watchlist"] == 1) & painel["target_distress_6m"].notna()]
    ultimo = int((elig["dt_comptc"].dt.year * 12 + elig["dt_comptc"].dt.month).max())
    fora, (cal_ini, cal_fim) = served_holdout_months(ultimo)
    sub = elig[elig["dt_comptc"].dt.strftime("%Y-%m").isin(fora)]
    raw = served["lgbm_monotonic"].predict_proba(sub[served["feature_cols"]])[:, 1]
    df = pd.DataFrame({"dt_comptc": sub["dt_comptc"].to_numpy(), "score": served["venn_abers"].predict_full(raw).prob,
                       "y_true": sub["target_distress_6m"].to_numpy()})
    r = served_level_check(df, fora, pd.Timestamp(painel["dt_comptc"].max()))
    return {**r, "meses": [fora[0], fora[-1]], "calibracao": [cal_ini, cal_fim]}


def _definicoes() -> dict[str, float]:
    return {
        "horizonte_meses": HORIZON_MONTHS,
        "queda_senior_acumulada_pct": gold.TARGET_SENIOR_CUM_PCT,
        "queda_senior_mensal_pct": gold.TARGET_SENIOR_MONTH_PCT,
        "salto_estoque_ruim_pp": gold.TARGET_BAD_JUMP * 100,
        "estoque_ruim_minimo_pct": gold.TARGET_BAD_LEVEL * 100,
        "folga_subordinacao_pp": gold.TARGET_CUSHION_MARGIN * 100,
        "elegivel_pl_minimo_mi": gold.ELIGIBLE_MIN_PL / 1e6,
        "elegivel_carteira_minima_mi": gold.ELIGIBLE_MIN_CARTEIRA / 1e6,
        "elegivel_estoque_ruim_maximo_pct": gold.ELIGIBLE_MAX_BAD * 100,
        "estresse_agudo_queda_senior_mes_pct": gold.ACUTE_SENIOR_MONTH_PCT,
        "estresse_agudo_estoque_ruim_pct": gold.ACUTE_BAD_LEVEL * 100,
    }


def _clean(o: Any) -> Any:
    """NaN -> None para gerar JSON válido."""
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_clean(v) for v in o]
    if isinstance(o, float) and math.isnan(o):
        return None
    return o


def build_summary(
    oof: pd.DataFrame,
    score_col: str,
    last_competence: pd.Timestamp,
    horizon: int = HORIZON_MONTHS,
    baixado_em: str | None = None,
    min_eventos_segmento: int = 30,
    panel: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Resumo completo (JSON) que alimenta a página e o PDF de Metodologia."""
    vint = vintage_metrics(oof, score_col, last_competence, horizon)
    ok = _evaluable(oof, last_competence, horizon)
    allp = oof.copy()
    allp["dt_comptc"] = pd.to_datetime(allp["dt_comptc"])
    fora = allp[~complete_window_mask(allp["dt_comptc"], last_competence, horizon)]

    safras = []
    for row in vint.to_dict(orient="records"):
        g = ok[ok["ano"] == row["ano"]]
        row["calibracao"] = calibration_bins(g["y_true"].to_numpy(), g[score_col].to_numpy(), n_bins=10)
        safras.append(row)

    summary: dict[str, Any] = {
        "modelo": "o mesmo que gera o score ao vivo",
        "coluna_score": score_col,
        "ultima_competencia": str(last_competence.date())[:7],
        "baixado_em": baixado_em,
        "definicoes": _definicoes(),
        "avaliacao": {
            "inicio": str(ok["dt_comptc"].min().date())[:7] if len(ok) else None,
            "fim": str(ok["dt_comptc"].max().date())[:7] if len(ok) else None,
            "n_obs": int(len(ok)),
            "eventos": int(ok["y_true"].sum()),
            "taxa_eventos": float(ok["y_true"].mean()) if len(ok) else float("nan"),
        },
        "fora_por_janela_incompleta": {
            "n_obs": int(len(fora)),
            "inicio": str(fora["dt_comptc"].min().date())[:7] if len(fora) else None,
            "fim": str(fora["dt_comptc"].max().date())[:7] if len(fora) else None,
            "taxa_eventos_parcial": float(fora["y_true"].mean()) if len(fora) else float("nan"),
        },
        "por_safra": safras,
        "erros": error_analysis(oof, score_col, last_competence, horizon, min_eventos_segmento),
    }
    if panel is not None:
        summary["perfil_erros"] = error_profile(oof, panel, score_col, last_competence, horizon)
    return _clean(summary)


def _download_date() -> str | None:
    manifest = BRONZE_DIR / "manifest.json"
    if not manifest.exists():
        return None
    ts = json.loads(manifest.read_text(encoding="utf-8")).get("generated_at_utc")
    return pd.Timestamp(ts).strftime("%d/%m/%Y") if ts else None


def compute_methodology_summary() -> dict[str, Any]:
    """Lê as previsões fora da amostra e o painel Gold e grava reports/metrics/methodology_*.{json,csv}."""
    oof_path = GOLD_DIR / "oof_master_h6.parquet"
    if not oof_path.exists():
        oof_path = GOLD_DIR / "oof_tabular_h6.parquet"
    oof = pd.read_parquet(oof_path)
    panel = pd.read_parquet(
        GOLD_DIR / "panel_gold.parquet",
        columns=[
            "entity_id", "dt_comptc", "target_senior_dd_6m", "target_credit_surge_6m",
            "inad_90p_pct", "inad_early_1_60_pct", "subordinacao_pct",
        ],
    )
    last = pd.Timestamp(panel["dt_comptc"].max())
    summary = build_summary(oof, SERVED_SCORE_COL, last, HORIZON_MONTHS, _download_date(), panel=panel)
    nivel = served_model_level()
    if nivel is not None:
        summary["nivel_modelo_servido"] = _clean(nivel)

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    (METRICS_DIR / "methodology_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    pd.DataFrame([{k: v for k, v in s.items() if k != "calibracao"} for s in summary["por_safra"]]).to_csv(
        METRICS_DIR / "methodology_vintage.csv", index=False
    )
    pd.DataFrame(
        [{"safra": s["rotulo"], **b} for s in summary["por_safra"] for b in s["calibracao"]]
    ).to_csv(METRICS_DIR / "methodology_calibration_vintage.csv", index=False)
    pd.DataFrame(summary["erros"]["por_segmento"]).to_csv(METRICS_DIR / "methodology_errors_segment.csv", index=False)
    return summary

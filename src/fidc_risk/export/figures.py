"""Geração das 8 figuras científicas de alta resolução (220 DPI) a partir dos artefatos reais."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fidc_risk.config import FIGURES_DIR, GOLD_DIR, METRICS_DIR, ensure_directories

PALETTE = {
    "navy": "#0f172a",
    "slate": "#475569",
    "blue": "#0284c7",
    "teal": "#0d9488",
    "amber": "#d97706",
    "red": "#dc2626",
    "purple": "#7c3aed",
    "light_bg": "#f8fafc",
    "grid": "#e2e8f0",
}


def _apply_pub_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
            "axes.edgecolor": "#cbd5e1",
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "grid.color": "#e2e8f0",
            "grid.linestyle": "--",
            "grid.alpha": 0.7,
            "axes.titlesize": 11.5,
            "axes.titleweight": "bold",
            "axes.labelsize": 9.8,
            "xtick.labelsize": 8.8,
            "ytick.labelsize": 8.8,
            "legend.fontsize": 8.6,
            "figure.facecolor": "white",
            "axes.facecolor": "#ffffff",
        }
    )


def _save(fig: plt.Figure, filename: str) -> Path:
    out = FIGURES_DIR / filename
    fig.savefig(out, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def plot_fig1_industry_evolution(panel: pd.DataFrame) -> Path:
    com = panel[panel["is_fidc_np_structural"] == 0].copy()
    m = (
        com.groupby("dt_comptc", as_index=False)
        .agg(
            n_funds=("entity_id", "nunique"),
            pl_bi=("vl_pl", lambda s: float(s.sum()) / 1e9),
            inad90_wt=(
                "inad_90p_pct",
                lambda s: float(np.average(s, weights=np.maximum(com.loc[s.index, "vl_carteira"], 1.0))),
            ),
            selic=("selic_meta_aa", "median"),
        )
        .sort_values("dt_comptc")
    )

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10.5, 6.4), sharex=True)
    ax1.fill_between(m["dt_comptc"], 0, m["pl_bi"], color=PALETTE["blue"], alpha=0.18)
    ax1.plot(m["dt_comptc"], m["pl_bi"], color=PALETTE["blue"], lw=2.2, label="Patrimônio Líquido Comercial (R$ bi)")
    ax1.set_ylabel("Patrimônio Líquido (R$ bilhões)", color=PALETTE["blue"])
    ax1_r = ax1.twinx()
    ax1_r.grid(False)
    ax1_r.plot(m["dt_comptc"], m["n_funds"], color=PALETTE["navy"], lw=1.8, ls="--", label="FIDCs/Classes Ativos")
    ax1_r.set_ylabel("Nº de FIDCs / Classes", color=PALETTE["navy"])
    ax1.set_title("Evolução Estrutural da Indústria de FIDCs no Brasil (CVM 2018–2026, ex-NP)")

    # Anotações de marcos históricos escalonadas verticalmente (zero colisão textual)
    for dt_str, label, y_frac in [
        ("2020-03-31", "COVID-19", 0.88),
        ("2023-01-31", "Choque Americanas/Light", 0.88),
        ("2024-04-30", "Expansão RCVM 175", 0.72),
    ]:
        dt_m = pd.Timestamp(dt_str)
        ax1.axvline(dt_m, color=PALETTE["slate"], lw=1.0, ls=":", alpha=0.8)
        ax1.text(dt_m, ax1.get_ylim()[1] * y_frac, f" {label}", fontsize=8.2, color=PALETTE["slate"], fontweight="bold")

    ax2.plot(
        m["dt_comptc"],
        m["inad90_wt"] * 100.0,
        color=PALETTE["red"],
        lw=2.2,
        label="Inadimplência >90d Ponderada (% Carteira)",
    )
    ax2.set_ylabel("Inadimplência >90d (%)", color=PALETTE["red"])
    ax2_r = ax2.twinx()
    ax2_r.grid(False)
    ax2_r.step(m["dt_comptc"], m["selic"], where="post", color=PALETTE["amber"], lw=1.9, label="Meta Selic (% a.a.)")
    ax2_r.set_ylabel("Meta Selic (% a.a.)", color=PALETTE["amber"])
    ax2.set_title("Ciclo Monetário (BCB SGS) vs. Inadimplência >90 dias da Carteira")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines1r, labels1r = ax1_r.get_legend_handles_labels()
    ax1.legend(lines1 + lines1r, labels1 + labels1r, loc="upper left", frameon=True)

    lines2, labels2 = ax2.get_legend_handles_labels()
    lines2r, labels2r = ax2_r.get_legend_handles_labels()
    ax2.legend(lines2 + lines2r, labels2 + labels2r, loc="upper left", frameon=True)
    fig.tight_layout()
    return _save(fig, "fig1_industry_evolution.png")


def plot_fig2_model_benchmark() -> Path:
    tab_f = pd.read_csv(METRICS_DIR / "tabular_folds_metrics.csv")
    tab_f = tab_f[tab_f["horizon_months"] == 6]
    deep_f_path = METRICS_DIR / "deep_folds_metrics.csv"
    if deep_f_path.exists():
        deep_f = pd.read_csv(deep_f_path)
        all_f = pd.concat([tab_f[["fold", "model", "pr_auc", "brier"]], deep_f[["fold", "model", "pr_auc", "brier"]]])
    else:
        all_f = tab_f[["fold", "model", "pr_auc", "brier"]]

    keep_models = [
        ("heuristic_committee", "Comitê Estrutural (Regra)", PALETTE["slate"]),
        ("logistic_hazard_va", "Hazard Logístico + VA", "#64748b"),
        ("lgbm_monotonic_va", "LightGBM Monotônico + VA", PALETTE["blue"]),
        ("temporal_transformer_5seed_va", "Transformer Temporal (GPU)", PALETTE["amber"]),
        ("hetero_contagion_gnn_5seed_va", "HeteroContagionGNN (GPU)", PALETTE["purple"]),
        ("gbdt_ensemble_va", "Ensemble GBDT + VA", PALETTE["teal"]),
        ("hybrid_gbdt_contagion_gnn", "Stack Híbrido GBDT + GNN", PALETTE["navy"]),
    ]
    avail = [(k, lbl, c) for k, lbl, c in keep_models if k in set(all_f["model"])]

    folds = ["fold_2022", "fold_2023", "fold_2024", "fold_2025_2026"]
    fold_labels = ["2022\nAlta Selic", "2023\nCrise Crédito", "2024\nRCVM 175", "2025–26\nJuros Restritivos"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.0, 5.0))
    x = np.arange(len(folds))
    width = 0.82 / max(len(avail), 1)

    for idx, (mkey, mlbl, col) in enumerate(avail):
        sub = all_f[all_f["model"] == mkey].set_index("fold")
        vals = [float(sub.loc[f, "pr_auc"]) if f in sub.index else 0.0 for f in folds]
        offset = (idx - (len(avail) - 1) / 2.0) * width
        ax1.bar(x + offset, vals, width=width * 0.92, label=mlbl, color=col, alpha=0.9)

    ax1.axhline(0.125, color=PALETTE["red"], ls="--", lw=1.1, label="Taxa Base (~12,5%)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(fold_labels)
    ax1.set_ylim(0.0, 0.56)
    ax1.set_ylabel("PR-AUC Out-of-Time (h = 6 meses)")
    ax1.set_title("Desempenho por Regime Macro-Crédito (Embargo = 6m)")
    ax1.legend(loc="upper right", fontsize=7.5, ncol=2, framealpha=0.92)

    # Painel direito: Pooled PR-AUC vs Brier Score
    oof_path = GOLD_DIR / "oof_master_h6.parquet"
    if not oof_path.exists():
        oof_path = GOLD_DIR / "oof_tabular_h6.parquet"
    oof = pd.read_parquet(oof_path)
    from fidc_risk.models.metrics import compute_binary_warning_metrics

    col_lookup = {
        "heuristic_committee": "prob_heuristic",
        "logistic_hazard_va": "prob_logistic",
        "lgbm_monotonic_va": "prob_lgbm_mono_va",
        "temporal_transformer_5seed_va": "prob_seq_transformer_va",
        "hetero_contagion_gnn_5seed_va": "prob_hetero_gnn_va",
        "gbdt_ensemble_va": "prob_gbdt_ensemble_va",
        "hybrid_gbdt_contagion_gnn": "prob_hybrid_gbdt_gnn",
    }
    label_offsets = {
        "Comitê Estrutural (Regra)": (8, -2),
        "Hazard Logístico + VA": (8, -4),
        "Transformer Temporal (GPU)": (8, -10),
        "HeteroContagionGNN (GPU)": (-125, -14),
        "LightGBM Monotônico + VA": (8, 4),
        "Ensemble GBDT + VA": (-120, -14),
        "Stack Híbrido GBDT + GNN": (-135, -18),
    }
    pts = []
    for mkey, mlbl, col in avail:
        pcol = col_lookup.get(mkey)
        if pcol and pcol in oof.columns:
            mm = compute_binary_warning_metrics(oof["y_true"].values, oof[pcol].values)
            pts.append((mlbl, mm["brier"], mm["pr_auc"], mm["recall_top5"], col))

    for mlbl, br, pr, rec5, col in pts:
        ax2.scatter(br, pr, s=160 + rec5 * 400, color=col, alpha=0.9, edgecolors="black", linewidth=0.8)
        dx, dy = label_offsets.get(mlbl, (7, -4))
        ax2.annotate(
            f"{mlbl}\n(Rec@5%={rec5*100:.1f}%)",
            (br, pr),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=7.9,
            fontweight="semibold",
        )
    ax2.set_xlim(0.1032, 0.0908)
    ax2.set_ylim(0.265, 0.405)
    ax2.set_xlabel("Brier Score Out-of-Time (menor à direita é melhor →)")
    ax2.set_ylabel("PR-AUC Consolidado (98.261 previsões)")
    ax2.set_title("Fronteira Precisão-Calibração (h = 6 meses)")
    fig.tight_layout()
    return _save(fig, "fig2_model_benchmark_regimes.png")


def plot_fig3_ablation_waterfall() -> Path:
    ab = pd.read_csv(METRICS_DIR / "tabular_ablation_h6.csv")
    prog = ab[ab["ablation_type"] == "progressive"].reset_index(drop=True)
    lofo = ab[ab["ablation_type"] == "lofo"].copy()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.2, 4.9))
    y_pos = np.arange(len(prog))
    colors = [
        PALETTE["red"] if i == len(prog) - 1 and prog.loc[i, "pr_auc"] < prog.loc[i - 1, "pr_auc"] else PALETTE["blue"]
        for i in range(len(prog))
    ]
    bars = ax1.barh(y_pos, prog["pr_auc"], color=colors, height=0.62, alpha=0.9)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(prog["configuration"])
    ax1.invert_yaxis()
    ax1.set_xlim(0.24, max(prog["pr_auc"]) * 1.08)
    ax1.set_xlabel("PR-AUC Out-of-Time (LightGBM Monotônico + Venn-Abers)")
    ax1.set_title("Adição Progressiva de Famílias (Descoberta do Proxy Macro)")
    for b, v in zip(bars, prog["pr_auc"], strict=False):
        ax1.text(v + 0.002, b.get_y() + b.get_height() / 2, f"{v:.4f}", va="center", fontsize=8.5, fontweight="bold")

    full_pr = float(prog.iloc[-1]["pr_auc"])
    lofo["delta_vs_full_pp"] = (lofo["pr_auc"] - full_pr) * 100.0
    lofo = lofo.sort_values("delta_vs_full_pp")
    lofo_labels = {
        "Sem aging_structure": "Sem Aging & Prazos",
        "Sem waterfall_liquidity": "Sem Waterfall & Liquidez",
        "Sem forensic_accounting": "Sem Sinais Forenses",
        "Sem originator_sector": "Sem Cedentes & Setores",
        "Sem network_contagion": "Sem Grafo de Contágio LOO",
        "Sem macro_cycle": "Sem Variáveis Macro em Nível",
    }
    yp2 = np.arange(len(lofo))
    c2 = [PALETTE["teal"] if v > 0 else PALETTE["amber"] for v in lofo["delta_vs_full_pp"]]
    ax2.barh(yp2, lofo["delta_vs_full_pp"], color=c2, height=0.60, alpha=0.9)
    ax2.axvline(0.0, color=PALETTE["navy"], lw=1.0)
    ax2.set_yticks(yp2)
    ax2.set_yticklabels([lofo_labels.get(x, x) for x in lofo["configuration"]])
    ax2.set_xlabel("Variação de PR-AUC ao Remover a Família (p.p.)")
    ax2.set_title("Impacto Leave-One-Family-Out (LOFO)")
    fig.tight_layout()
    return _save(fig, "fig3_ablation_waterfall.png")


def plot_fig4_calibration_venn_abers() -> Path:
    oof_path = GOLD_DIR / "oof_master_h6.parquet"
    if not oof_path.exists():
        oof_path = GOLD_DIR / "oof_tabular_h6.parquet"
    oof = pd.read_parquet(oof_path)
    y = oof["y_true"].values

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.4, 4.8))
    ax1.plot([0, 0.75], [0, 0.75], ls="--", color=PALETTE["slate"], lw=1.2, label="Calibração Perfeita")

    curves = [
        ("LightGBM Bruto (Sem Calibração)", oof["prob_lgbm_unc_raw"].values, PALETTE["red"], "s--"),
        ("Platt Scaling", oof["prob_lgbm_unc_platt"].values, PALETTE["amber"], "^-."),
        ("Isotônica (PAV)", oof["prob_lgbm_unc_iso"].values, PALETTE["blue"], "o-"),
        ("Ensemble + Venn-Abers Indutivo", oof["prob_gbdt_ensemble_va"].values, PALETTE["teal"], "D-"),
    ]
    for lbl, p, col, fmt in curves:
        edges = np.quantile(p, np.linspace(0, 1, 11))
        xs, ys = [], []
        for i in range(10):
            m = (p >= edges[i]) & (p <= edges[i + 1] if i == 9 else p < edges[i + 1])
            if m.sum() > 20:
                xs.append(float(p[m].mean()))
                ys.append(float(y[m].mean()))
        ax1.plot(xs, ys, fmt, color=col, lw=1.8, markersize=5, label=lbl)

    ax1.set_xlim(0, 0.75)
    ax1.set_ylim(0, 0.75)
    ax1.set_xlabel("Probabilidade Predita Média no Decil")
    ax1.set_ylabel("Frequência Empírica Realizada de Estresse (h = 6m)")
    ax1.set_title("Curvas de Confiabilidade Out-of-Time (n = 98.261)")
    ax1.legend(loc="upper left", frameon=True)

    # Painel direito: Largura Epistêmica Venn-Abers |p1 - p0| por faixa de risco
    bins = pd.qcut(oof["prob_gbdt_ensemble_va"], q=8, duplicates="drop")
    grp = oof.groupby(bins, observed=False).agg(
        p_mean=("prob_gbdt_ensemble_va", "mean"),
        p0_mean=("va_p0", "mean"),
        p1_mean=("va_p1", "mean"),
        w_p90=("va_width", lambda s: float(np.quantile(s, 0.90))),
        y_rate=("y_true", "mean"),
    )
    ax2.fill_between(
        grp["p_mean"] * 100,
        grp["p0_mean"] * 100,
        grp["p1_mean"] * 100,
        color=PALETTE["teal"],
        alpha=0.28,
        label="Banda Média Multiprobabilística [p0, p1]",
    )
    ax2.plot(grp["p_mean"] * 100, grp["y_rate"] * 100, "o-", color=PALETTE["navy"], lw=2.0, label="Taxa Realizada (%)")
    ax2.plot(
        grp["p_mean"] * 100,
        (grp["p_mean"] + grp["w_p90"]) * 100,
        "--",
        color=PALETTE["purple"],
        lw=1.5,
        label="p + Largura P90 (Cauda de Ambiguidade)",
    )
    ax2.set_xlabel("Probabilidade Venn-Abers Predita (%)")
    ax2.set_ylabel("Taxa / Limites de Probabilidade (%)")
    ax2.set_title("Quantificação de Incerteza Epistêmica (Venn-Abers)")
    ax2.legend(loc="upper left")
    fig.tight_layout()
    return _save(fig, "fig4_calibration_venn_abers.png")


def plot_fig5_forensic_event_study() -> Path:
    ev = pd.read_csv(METRICS_DIR / "forensic_event_study.csv")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.8, 4.8))

    ax1.axvline(0, color=PALETTE["red"], ls="--", lw=1.4, label="Choque Agudo (τ = 0)")
    ax1.plot(ev["tau"], ev["inad_early_1_60_mean_pct"], "o-", color=PALETTE["amber"], lw=2.1, label="Atraso Curto 1–60d (%)")
    ax1.plot(
        ev["tau"],
        ev["recompra_subst_mean_pct"] * 2.5,
        "s--",
        color=PALETTE["purple"],
        lw=1.9,
        label="Giro Recompras + Substituições (×2,5 %)",
    )
    ax1.plot(ev["tau"], ev["inad_90p_mean_pct"], "D-", color=PALETTE["red"], lw=2.2, label="Inadimplência >90d (%)")
    ax1.set_xlabel("Meses em Tempo de Evento ao Redor do 1º Estresse Agudo (τ)")
    ax1.set_ylabel("Percentual da Carteira (%)")
    ax1.set_title("Anatomia Forense Pré-Quebra: Atraso Curto e Recompras Antecedem τ=0")
    ax1.legend(loc="upper left")

    ax2.axvline(0, color=PALETTE["red"], ls="--", lw=1.4)
    ax2.plot(ev["tau"], ev["subordinacao_mean_pct"], "o-", color=PALETTE["blue"], lw=2.1, label="Colchão Subordinada (% PL)")
    ax2.plot(
        ev["tau"],
        ev["contagion_peer_inad90_mean_pct"] * 3.0,
        "^-",
        color=PALETTE["teal"],
        lw=1.9,
        label="Inad. >90d em Pares de Cedente (×3 %)",
    )
    ax2_r = ax2.twinx()
    ax2_r.grid(False)
    ax2_r.bar(ev["tau"], ev["senior_ret_mean_pct"], width=0.45, color=PALETTE["red"], alpha=0.28, label="Retorno Cota Sênior (% mês)")
    ax2_r.set_ylabel("Retorno Mensal Cota Sênior (%)", color=PALETTE["red"])
    ax2.set_xlabel("Meses em Tempo de Evento (τ)")
    ax2.set_ylabel("Subordinação / Contágio (%)")
    ax2.set_title("Erosão Estrutural da Subordinação e Queda da Cota Sênior")

    l1, lb1 = ax2.get_legend_handles_labels()
    l2, lb2 = ax2_r.get_legend_handles_labels()
    ax2.legend(l1 + l2, lb1 + lb2, loc="lower left", fontsize=8.0)
    fig.tight_layout()
    return _save(fig, "fig5_forensic_event_study.png")


def plot_fig6_markov_and_cqr() -> Path:
    mk = pd.read_csv(METRICS_DIR / "markov_global_transition_matrix.csv", index_col=0)
    cqr_m = pd.read_csv(METRICS_DIR / "rollrate_cqr_pooled_metrics.csv")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.8, 4.9))
    mat = mk.values * 100.0
    im = ax1.imshow(mat, cmap="YlGnBu", vmin=0, vmax=95)
    ax1.grid(False)
    labels = list(mk.columns)
    ax1.set_xticks(range(len(labels)))
    ax1.set_yticks(range(len(labels)))
    ax1.set_xticklabels(labels, rotation=25)
    ax1.set_yticklabels(labels)
    ax1.set_xlabel("Estado Destino em t + 1 mês")
    ax1.set_ylabel("Estado Origem em t")
    ax1.set_title("Matriz Mensal de Roll-Rate de Markov (%)")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            ax1.text(j, i, f"{v:.1f}%", ha="center", va="center", color="white" if v > 50 else PALETTE["navy"], fontsize=8.3, fontweight="bold")
    fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)

    horizons = [1, 3, 6]
    models_plot = [
        ("persistence_stratified", "Persistência Estratificada (Ref.)", PALETTE["slate"], "o--"),
        ("markov_rollrate_eb", "Roll-Rate Markov Empirical Bayes", PALETTE["amber"], "s-."),
        ("lgbm_quantile_raw", "LightGBM Quantílico Bruto", PALETTE["blue"], "^-"),
        ("lgbm_cqr_stratified", "LightGBM + CQR Estratificado", PALETTE["teal"], "D-"),
    ]
    for mkey, mlbl, col, fmt in models_plot:
        sub = cqr_m[cqr_m["model"] == mkey].sort_values("horizon_months")
        if len(sub) > 0:
            ax2.plot(sub["horizon_months"], sub["wis"] * 100.0, fmt, color=col, lw=2.1, markersize=6, label=mlbl)

    ax2.set_xticks(horizons)
    ax2.set_xticklabels(["h = 1 mês", "h = 3 meses", "h = 6 meses"])
    ax2.set_ylabel("Weighted Interval Score — WIS (p.p. da carteira, menor é melhor)")
    ax2.set_title("Previsão Probabilística da Inadimplência >90d")
    ax2.legend(loc="upper left")
    fig.tight_layout()
    return _save(fig, "fig6_markov_and_cqr.png")


def plot_fig7_portfolio_wealth() -> Path:
    bt = pd.read_csv(METRICS_DIR / "portfolio_monthly_returns.csv")
    bt["realization_dt"] = pd.to_datetime(bt["realization_dt"])
    cdi_wealth = 100.0 * np.cumprod(1.0 + bt["cdi_ret_pct"].values / 100.0)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10.6, 6.4), sharex=True, gridspec_kw={"height_ratios": [1.85, 1.15]})

    series_cfg = [
        ("cdi_ret_pct", "100% CDI (BCB SGS 4391)", PALETTE["slate"], "--", 1.8),
        ("equal_weight_lagged", "Equal-Weight Cotas Sênior", "#64748b", ":", 1.8),
        ("traditional_committee_lagged", "Filtro Tradicional Comitê", PALETTE["amber"], "-", 1.9),
        ("yield_chasing_lagged", "Yield-Chasing Top-25 (Sem Filtro)", PALETTE["red"], "-.", 1.9),
        ("ml_watchlist_lagged", "Watchlist ML Cost-Sensitive", PALETTE["blue"], "-", 2.2),
        ("cvar_venn_abers_lagged", "Alocação CVaR + Venn-Abers (c/ Lag)", PALETTE["teal"], "-", 2.5),
    ]

    for col, lbl, color, ls, lw in series_cfg:
        r = bt[col].values / 100.0
        wealth = 100.0 * np.cumprod(1.0 + r)
        ax1.plot(bt["realization_dt"], wealth, ls=ls, lw=lw, color=color, label=f"{lbl} ({wealth[-1]:.1f})")

        if col != "cdi_ret_pct":
            rel_excess_pct = (wealth / cdi_wealth - 1.0) * 100.0
            ax2.plot(bt["realization_dt"], rel_excess_pct, ls=ls, lw=1.8, color=color, label=lbl)

    ax1.set_ylabel("Patrimônio Acumulado (Base 100)")
    ax1.set_title("Backtest Institucional Walk-Forward de Cotas Sênior c/ Defasagem Real de Resgate (2022–2026)")
    ax1.legend(loc="upper left", frameon=True, fontsize=8.3)

    ax2.axhline(0, color=PALETTE["navy"], lw=0.9, ls="--")
    ax2.set_ylabel("Alfa Acumulado vs. CDI (%)")
    ax2.set_xlabel("Data de Realização Mensal")
    ax2.set_title("Riqueza Relativa Acumulada Acima do CDI (Wealth Ratio − 1)")
    fig.tight_layout()
    return _save(fig, "fig7_portfolio_wealth_vs_cdi.png")


def plot_fig8_shap_and_drift() -> Path:
    shap_df = pd.read_csv(METRICS_DIR / "shap_global_importance.csv").head(14)
    drift_df = pd.read_csv(METRICS_DIR / "covariate_drift_psi.csv")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.6, 5.2))
    fam_colors = {
        "aging_structure": PALETTE["blue"],
        "forensic_accounting": PALETTE["red"],
        "waterfall_liquidity": PALETTE["teal"],
        "originator_sector": PALETTE["amber"],
        "network_contagion": PALETTE["purple"],
        "macro_cycle": PALETTE["slate"],
    }
    y_pos = np.arange(len(shap_df))
    bar_cols = [fam_colors.get(f, PALETTE["navy"]) for f in shap_df["family"]]
    ax1.barh(y_pos, shap_df["mean_abs_shap"], color=bar_cols, height=0.66, alpha=0.9)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(shap_df["label_ptbr"], fontsize=8.2)
    ax1.invert_yaxis()
    ax1.set_xlabel("Impacto Médio Absoluto |SHAP| (log-odds)")
    ax1.set_title("Top-14 Determinantes Globais de Risco (TreeSHAP)")

    piv = drift_df.pivot(index="label_ptbr", columns="semester", values="psi")
    top8 = shap_df["label_ptbr"].head(8).tolist()
    piv = piv.loc[[r for r in top8 if r in piv.index]]
    mat = np.clip(piv.values, 0.0, 0.60)
    im = ax2.imshow(mat, aspect="auto", cmap="OrRd", vmin=0.0, vmax=0.50)
    ax2.grid(False)
    ax2.set_xticks(range(len(piv.columns)))
    ax2.set_xticklabels(list(piv.columns), rotation=35, fontsize=8.0)
    ax2.set_yticks(range(len(piv.index)))
    ax2.set_yticklabels(list(piv.index), fontsize=8.0)
    ax2.set_title("Monitoramento Semestral de Drift (PSI vs. Pré-2022)")
    fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04, label="PSI (>0,25 = Crítico)")
    fig.tight_layout()
    return _save(fig, "fig8_shap_and_drift.png")


def generate_all_figures() -> list[str]:
    ensure_directories()
    _apply_pub_style()
    panel = pd.read_parquet(GOLD_DIR / "panel_gold.parquet")
    generated = [
        plot_fig1_industry_evolution(panel),
        plot_fig2_model_benchmark(),
        plot_fig3_ablation_waterfall(),
        plot_fig4_calibration_venn_abers(),
        plot_fig5_forensic_event_study(),
        plot_fig6_markov_and_cqr(),
        plot_fig7_portfolio_wealth(),
        plot_fig8_shap_and_drift(),
    ]
    names = [p.name for p in generated]
    print(f"[Export] {len(names)} figuras de alta resolução geradas em {FIGURES_DIR}: {names}")
    return names

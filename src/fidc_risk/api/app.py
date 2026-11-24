"""Aplicação FastAPI servindo o Lakehouse DuckDB, Watchlist Calibrada e Motor Contrafactual."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import duckdb
import joblib
import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from fidc_risk import __version__, config
from fidc_risk.api.schemas import (
    CounterfactualRequest,
    CounterfactualResponse,
    HealthResponse,
    ShapDriverOut,
    WatchlistItemOut,
)
from fidc_risk.config import GOLD_DIR, LAKEHOUSE_DB_PATH, MODELS_DIR, WEB_DATA_DIR, WEB_DIR
from fidc_risk.models.rollrate_cqr import QUANTILE_ALPHAS, _delinquency_stratum
from fidc_risk.monitoring.shap_and_drift import human_feature_label


class RiskServiceState:
    """Gerencia artefatos em memória para inferência sub-15ms e consultas analíticas."""

    def __init__(
        self,
        live_df: pd.DataFrame | None = None,
        served_tab: dict[str, Any] | None = None,
        served_cqr: dict[str, Any] | None = None,
    ) -> None:
        self.live_df = live_df
        self.served_tab = served_tab
        self.served_cqr = served_cqr
        self._explainer: Any | None = None

    def ensure_loaded(self) -> None:
        if self.live_df is None:
            live_path = GOLD_DIR / "live_watchlist_scored.parquet"
            if live_path.exists():
                self.live_df = pd.read_parquet(live_path)
        if self.served_tab is None:
            tab_path = MODELS_DIR / "served_tabular_h6.pkl"
            if tab_path.exists():
                self.served_tab = joblib.load(tab_path)
        if self.served_cqr is None:
            cqr_path = MODELS_DIR / "served_cqr_h3.pkl"
            if cqr_path.exists():
                self.served_cqr = joblib.load(cqr_path)

    def get_explainer(self) -> Any | None:
        self.ensure_loaded()
        if self._explainer is None and self.served_tab is not None:
            self._explainer = shap.TreeExplainer(self.served_tab["lgbm_monotonic"])
        return self._explainer


def create_app(custom_state: RiskServiceState | None = None) -> FastAPI:
    state = custom_state or RiskServiceState()

    app = FastAPI(
        title="fidc-risk-br Analytical & Scoring API",
        version=__version__,
        description=(
            "API institucional de alerta precoce, calibração Venn-Abers, previsão quantílica CQR "
            "e simulação contrafactual para Fundos de Investimento em Direitos Creditórios (FIDCs)."
        ),
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health", response_model=HealthResponse)
    def health_check() -> HealthResponse:
        state.ensure_loaded()
        n_live = len(state.live_df) if state.live_df is not None else 0
        latest = (
            str(pd.to_datetime(state.live_df["dt_comptc"].max()).date())
            if state.live_df is not None and n_live > 0
            else None
        )
        return HealthResponse(
            status="ok",
            version=__version__,
            model_loaded=state.served_tab is not None,
            live_watchlist_count=n_live,
            latest_competence=latest,
        )

    @app.get("/api/v1/overview")
    def get_overview() -> dict[str, Any]:
        bundle_path = WEB_DATA_DIR / "observatory_bundle.json"
        if bundle_path.exists():
            data = json.loads(bundle_path.read_text(encoding="utf-8"))
            return {
                "meta": data.get("meta", {}),
                "kpis": data.get("kpis", {}),
                "benchmarks_h6": data.get("benchmarks_h6", []),
                "forensic": data.get("forensic", {}),
                "portfolio_summary": data.get("portfolio_summary", []),
            }
        state.ensure_loaded()
        return {
            "meta": {"version": __version__},
            "kpis": {"live_funds": len(state.live_df) if state.live_df is not None else 0},
        }

    @app.get("/api/v1/watchlist", response_model=list[WatchlistItemOut])
    def list_watchlist(
        sector: str | None = Query(default=None),
        only_open_end: bool = Query(default=False),
        min_pl_millions: float = Query(default=5.0, ge=0.0),
        search: str | None = Query(default=None),
        limit: int = Query(default=50, ge=1, le=300),
    ) -> list[WatchlistItemOut]:
        state.ensure_loaded()
        if state.live_df is None or len(state.live_df) == 0:
            return []

        df = state.live_df[state.live_df["vl_pl"] >= min_pl_millions * 1_000_000.0]
        if sector:
            df = df[df["dominant_sector"].astype(str).str.lower() == sector.strip().lower()]
        if only_open_end:
            df = df[df["is_aberto"] == 1]
        if search:
            q = search.strip().upper()
            mask = df["denom_social"].astype(str).str.upper().str.contains(q, regex=False) | df[
                "entity_id"
            ].astype(str).str.contains(q, regex=False)
            df = df[mask]

        df = df.sort_values("prob_distress_6m", ascending=False).head(limit)
        items: list[WatchlistItemOut] = []
        for row in df.to_dict(orient="records"):
            raw_drv = row.get("shap_drivers_json", "[]")
            drivers = [ShapDriverOut(**d) for d in json.loads(raw_drv)] if isinstance(raw_drv, str) else []
            items.append(
                WatchlistItemOut(
                    entity_id=str(row["entity_id"]),
                    cnpj_raw=str(row.get("cnpj_raw", row["entity_id"])),
                    denom_social=str(row["denom_social"]),
                    dominant_sector=str(row["dominant_sector"]),
                    nome_gestor=str(row.get("nome_gestor", "")),
                    is_aberto=int(row["is_aberto"]),
                    vl_pl=float(row["vl_pl"]),
                    prob_distress_6m=float(row["prob_distress_6m"]),
                    va_p0=float(row["va_p0"]),
                    va_p1=float(row["va_p1"]),
                    va_width=float(row["va_width"]),
                    inad_90p_pct=float(row["inad_90p_pct"]),
                    inad_early_1_60_pct=float(row["inad_early_1_60_pct"]),
                    subordinacao_pct=float(row["subordinacao_pct"]),
                    recompra_masking_index=float(row["recompra_masking_index"]),
                    scr_provision_gap_pct=float(row["scr_provision_gap_pct"]),
                    contagion_cedente_inad90p_wt=float(row["contagion_cedente_inad90p_wt"]),
                    cqr_inad90_h3_q10=float(row["cqr_inad90_h3_q10"]),
                    cqr_inad90_h3_q50=float(row["cqr_inad90_h3_q50"]),
                    cqr_inad90_h3_q90=float(row["cqr_inad90_h3_q90"]),
                    shap_drivers=drivers,
                )
            )
        return items

    @app.get("/api/v1/funds/{entity_id}")
    def get_fund_history(entity_id: str) -> dict[str, Any]:
        clean_id = "".join(ch for ch in entity_id if ch.isdigit())
        state.ensure_loaded()

        current_snapshot = None
        if state.live_df is not None:
            sub = state.live_df[state.live_df["entity_id"] == clean_id]
            if len(sub) > 0:
                r = sub.iloc[0].to_dict()
                current_snapshot = {
                    "entity_id": clean_id,
                    "denom_social": r.get("denom_social"),
                    "dominant_sector": r.get("dominant_sector"),
                    "nome_gestor": r.get("nome_gestor"),
                    "vl_pl": float(r.get("vl_pl", 0.0)),
                    "prob_distress_6m": float(r.get("prob_distress_6m", 0.0)),
                    "va_p0": float(r.get("va_p0", 0.0)),
                    "va_p1": float(r.get("va_p1", 0.0)),
                    "shap_drivers": json.loads(r.get("shap_drivers_json", "[]")),
                }

        history_rows = []
        if LAKEHOUSE_DB_PATH.exists():
            con = duckdb.connect(str(LAKEHOUSE_DB_PATH), read_only=True)
            try:
                q = """
                    SELECT
                        CAST(dt_comptc AS VARCHAR) AS dt_comptc,
                        vl_pl,
                        inad_90p_pct,
                        inad_early_1_60_pct,
                        pdd_carteira_pct,
                        subordinacao_pct,
                        recompra_carteira_pct,
                        eff_senior_ret_mes,
                        contagion_cedente_inad90p_wt
                    FROM gold.panel
                    WHERE entity_id = ?
                    ORDER BY dt_comptc ASC
                """
                hdf = con.execute(q, [clean_id]).fetchdf()
                hdf = hdf.replace({np.nan: None})
                history_rows = hdf.to_dict(orient="records")
            finally:
                con.close()

        if current_snapshot is None and not history_rows:
            raise HTTPException(status_code=404, detail=f"Fundo {clean_id} não encontrado no Lakehouse.")

        return {"entity_id": clean_id, "current": current_snapshot, "history": history_rows}

    @app.post("/api/v1/score", response_model=CounterfactualResponse)
    def score_counterfactual(req: CounterfactualRequest) -> CounterfactualResponse:
        state.ensure_loaded()
        if state.served_tab is None:
            raise HTTPException(status_code=503, detail="Modelo servido não carregado.")

        feat_cols: list[str] = state.served_tab["feature_cols"]
        lgbm_m = state.served_tab["lgbm_monotonic"]
        va_cal = state.served_tab["venn_abers"]

        base_vec = dict.fromkeys(feat_cols, 0.0)
        resolved_id: str | None = None
        if req.entity_id and state.live_df is not None:
            cid = "".join(ch for ch in req.entity_id if ch.isdigit())
            match = state.live_df[state.live_df["entity_id"] == cid]
            if len(match) == 0:
                raise HTTPException(status_code=404, detail=f"Fundo {cid} não encontrado na competência ativa.")
            row0 = match.iloc[0]
            for c in feat_cols:
                base_vec[c] = float(row0[c])
            resolved_id = cid

        shocked_vec = dict(base_vec)
        for k, v in req.feature_overrides.items():
            if k not in shocked_vec:
                raise HTTPException(status_code=422, detail=f"Feature desconhecida: {k}")
            shocked_vec[k] = float(v)

        x_pair = pd.DataFrame([base_vec, shocked_vec], columns=feat_cols)
        raw_pair = lgbm_m.predict_proba(x_pair)[:, 1]
        va_pair = va_cal.predict_full(raw_pair)

        q50_val, q90_val = 0.0, 0.0
        if state.served_cqr is not None:
            qm = state.served_cqr["qmodels_h3"]
            cqr = state.served_cqr["cqr_h3"]
            x_sh = x_pair.iloc[[1]]
            q_raw = {a: qm[a].predict(x_sh) for a in QUANTILE_ALPHAS}
            q_adj = cqr.apply(q_raw, _delinquency_stratum(np.array([shocked_vec["inad_90p_pct"]])))
            q50_val = float(q_adj[0.50][0])
            q90_val = float(q_adj[0.90][0])

        drivers: list[ShapDriverOut] = []
        explainer = state.get_explainer()
        if explainer is not None:
            sv_raw = explainer.shap_values(x_pair.iloc[[1]])
            sv = np.asarray(sv_raw[1] if isinstance(sv_raw, list) else sv_raw, dtype="float64")[0]
            top_idx = np.argsort(-np.abs(sv))[:6]
            for fi in top_idx:
                cname = feat_cols[int(fi)]
                drivers.append(
                    ShapDriverOut(
                        feature=cname,
                        label=human_feature_label(cname),
                        shap_log_odds=round(float(sv[fi]), 4),
                        feature_value=round(float(shocked_vec[cname]), 4),
                    )
                )

        return CounterfactualResponse(
            entity_id=resolved_id,
            base_prob_6m=round(float(va_pair.prob[0]), 5),
            shocked_prob_6m=round(float(va_pair.prob[1]), 5),
            delta_prob_6m=round(float(va_pair.prob[1] - va_pair.prob[0]), 5),
            shocked_va_p0=round(float(va_pair.p0[1]), 5),
            shocked_va_p1=round(float(va_pair.p1[1]), 5),
            shocked_va_width=round(float(va_pair.width[1]), 5),
            shocked_cqr_inad90_h3_q50=round(q50_val, 5),
            shocked_cqr_inad90_h3_q90=round(q90_val, 5),
            top_shap_drivers=drivers,
        )

    @app.get("/data/private/observatory_bundle.json", include_in_schema=False)
    def private_bundle() -> FileResponse:
        """Pacote nominal (fundos identificados) para uso local; mora em data/private/, fora de web/."""
        path = config.PRIVATE_BUNDLE_PATH
        if not path.exists():
            raise HTTPException(status_code=404, detail="Pacote nominal não gerado nesta máquina.")
        return FileResponse(path, media_type="application/json")

    if WEB_DIR.exists() and (Path(WEB_DIR) / "index.html").exists():
        app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")

    return app


app = create_app()

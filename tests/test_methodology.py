"""Métricas da página de Metodologia: validação por safra, calibração e erros agregados (dados sintéticos)."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pymupdf
import pytest

from fidc_risk.models.metrics import watchlist_topk_metrics
from fidc_risk.monitoring.methodology import (
    calibration_bins,
    complete_window_mask,
    error_analysis,
    top_k_mask,
    vintage_metrics,
)


def test_janela_completa_exige_h_meses_observados_depois_de_t() -> None:
    dt = pd.Series(pd.to_datetime(["2026-03-31", "2026-04-30", "2026-05-31", "2026-10-31"]))
    got = complete_window_mask(dt, pd.Timestamp("2026-10-31"), horizon=6).tolist()
    assert got == [True, True, False, False]


def test_top_k_pega_os_maiores_scores_com_k_arredondado_para_cima() -> None:
    s = np.array([0.1, 0.9, 0.5, 0.7, 0.2])
    assert top_k_mask(s, 0.2).tolist() == [False, True, False, False, False]  # ceil(5*0.2)=1
    assert top_k_mask(s, 0.5).tolist() == [False, True, True, True, False]  # ceil(2.5)=3


def test_top_k_concorda_com_a_metrica_do_pipeline_sem_empates() -> None:
    rng = np.random.default_rng(0)
    p = rng.random(400)
    y = (rng.random(400) < p * 0.5).astype(int)
    mask = top_k_mask(p, 0.05)
    recall = y[mask].sum() / y.sum()
    assert recall == pytest.approx(watchlist_topk_metrics(y, p, 0.05)["recall_top5"])


def test_calibracao_em_faixas_de_score_soma_todas_as_observacoes() -> None:
    p = np.repeat([0.05, 0.15, 0.25, 0.35], 50)
    y = np.concatenate([np.r_[np.ones(k), np.zeros(50 - k)] for k in (2, 8, 12, 18)])
    bins = calibration_bins(y, p, n_bins=4)
    assert [b["n"] for b in bins] == [50, 50, 50, 50]
    assert [b["previsto"] for b in bins] == pytest.approx([0.05, 0.15, 0.25, 0.35])
    assert [b["observado"] for b in bins] == pytest.approx([0.04, 0.16, 0.24, 0.36])


def _oof() -> pd.DataFrame:
    """Dois anos: 2023 com ordenação perfeita, 2024 com ordenação invertida; e meses sem janela completa."""
    rows = []
    for ano, invert in ((2023, False), (2024, True)):
        for i in range(100):
            y = int(i >= 80)  # 20 eventos por ano
            score = (i + 1) / 100.0
            rows.append({
                "entity_id": f"{ano}{i:03d}",
                "dt_comptc": pd.Timestamp(f"{ano}-06-30"),
                "dominant_sector": "Consignado" if i % 2 else "Factoring",
                "y_true": y,
                "prob": (1.01 - score) if invert else score,
            })
    for i in range(10):  # 2026-07: janela de 6 meses incompleta (dados até 2026-10)
        rows.append({"entity_id": f"x{i}", "dt_comptc": pd.Timestamp("2026-07-31"),
                     "dominant_sector": "Factoring", "y_true": 0, "prob": 0.5})
    return pd.DataFrame(rows)


def test_metricas_por_safra_separam_anos_e_excluem_janela_incompleta() -> None:
    res = vintage_metrics(_oof(), "prob", pd.Timestamp("2026-10-31"), horizon=6)
    v = res.set_index("ano")
    assert list(v.index) == [2023, 2024]
    assert v.loc[2023, "n_obs"] == 100 and v.loc[2023, "eventos"] == 20
    assert v.loc[2023, "acerto_ordenacao"] == pytest.approx(1.0)
    assert v.loc[2024, "acerto_ordenacao"] == pytest.approx(0.0)
    # top 5% de 100 = 5 fundos; em 2023 todos são eventos: 5/5 contra base de 20% -> 5x a média
    assert v.loc[2023, "concentracao_top5"] == pytest.approx(5.0)
    assert v.loc[2023, "eventos_no_top5"] == pytest.approx(5 / 20)
    assert v.loc[2024, "concentracao_top5"] == pytest.approx(0.0)
    assert v.loc[2023, "taxa_eventos"] == pytest.approx(0.2)
    assert v.loc[2023, "score_medio"] == pytest.approx(np.mean(np.arange(1, 101) / 100))


def test_analise_de_erros_agrega_sem_identificar_fundos() -> None:
    err = error_analysis(_oof(), "prob", pd.Timestamp("2026-10-31"), horizon=6, min_eventos_segmento=1)
    tot = err["total"]
    assert tot["eventos"] == 40
    # 2023: os 20 eventos estão no top 20%; 2024: nenhum (ordem invertida)
    assert tot["eventos_no_top20"] == pytest.approx(0.5)
    assert tot["eventos_abaixo_da_mediana"] == pytest.approx(0.5)
    # top 5%: 5 por ano; 2023 só eventos, 2024 só não-eventos -> metade são alarmes falsos
    assert tot["alarmes_falsos_top5"] == pytest.approx(0.5)
    segs = {s["segmento"]: s for s in err["por_segmento"]}
    assert set(segs) == {"Consignado", "Factoring"}
    assert segs["Consignado"]["eventos"] + segs["Factoring"]["eventos"] == 40
    payload = json.dumps(err, ensure_ascii=False)
    assert "entity_id" not in payload and "2023000" not in payload


def test_segmentos_com_poucos_eventos_viram_outros() -> None:
    err = error_analysis(_oof(), "prob", pd.Timestamp("2026-10-31"), horizon=6, min_eventos_segmento=25)
    assert [s["segmento"] for s in err["por_segmento"]] == ["Outros"]


# ---------------------------------------------------------------- página e PDF

def _summary() -> dict:
    from fidc_risk.monitoring.methodology import build_summary

    return build_summary(_oof(), "prob", pd.Timestamp("2026-10-31"), horizon=6, baixado_em="18/11/2026",
                         min_eventos_segmento=1)


JARGAO = ["roc", "auc", "lift", "venn", "abers", "lightgbm", "shap", "walk-forward", "embargo", "ece", "pr-auc", "gbdt"]


def test_pagina_de_metodologia_responde_as_perguntas_sem_jargao() -> None:
    from fidc_risk.export.methodology_page import render_methodology_html

    html = render_methodology_html(_summary())
    for sec in ("De onde vêm os dados", "O que o score significa", "Como testamos", "Onde o modelo erra", "Limitações"):
        assert sec in html
    # Jargão só no glossário final de nomes técnicos.
    corpo = re.sub(r'<section id="nomes-tecnicos".*?</section>', "", html, flags=re.S)
    texto = re.sub(r"<[^>]+>", " ", corpo).lower()
    for termo in JARGAO:
        assert not re.search(rf"\b{re.escape(termo)}\b", texto), termo
    assert 'id="nomes-tecnicos"' in html
    assert "<svg" in html  # calibração por safra desenhada na própria página
    assert "Fundo A" not in html and not re.search(r"\d{2}\.\d{3}\.\d{3}/\d{4}-\d{2}", html)


def test_pdf_de_metodologia_tem_ate_duas_paginas(tmp_path: Path) -> None:
    from fidc_risk.export.methodology_page import render_methodology_pdf

    out = render_methodology_pdf(_summary(), tmp_path / "metodologia.pdf")
    with pymupdf.open(out) as doc:
        assert 1 <= len(doc) <= 2
        texto = "".join(p.get_text() for p in doc)
    assert "O que o score significa" in texto
    assert "Onde o modelo erra" in texto


def test_pagina_versionada_corresponde_ao_resumo_versionado() -> None:
    """web/metodologia.html é gerada a partir de reports/metrics/methodology_summary.json."""
    from fidc_risk.config import METRICS_DIR, WEB_DIR
    from fidc_risk.export.methodology_page import render_methodology_html

    summary = json.loads((METRICS_DIR / "methodology_summary.json").read_text(encoding="utf-8"))
    assert (WEB_DIR / "metodologia.html").read_text(encoding="utf-8") == render_methodology_html(summary)


# ---------------------------------------------------------------- nível do score em produção

def test_meses_que_o_modelo_servido_nao_viu() -> None:
    """Replica o corte do modelo servido (tabular_suite): calibra nos 6 últimos meses com rótulo e treina até
    7 meses antes disso; os meses entre os dois cortes ficam fora do treino e da calibração."""
    from fidc_risk.monitoring.methodology import served_holdout_months

    ultimo_com_rotulo = 2026 * 12 + 9  # set/2026
    assert served_holdout_months(ultimo_com_rotulo) == (
        ["2025-10", "2025-11", "2025-12", "2026-01", "2026-02", "2026-03"],
        ("2026-04", "2026-09"),
    )


def test_checagem_de_nivel_compara_score_medio_e_eventos_por_mes() -> None:
    from fidc_risk.monitoring.methodology import served_level_check

    df = pd.DataFrame({
        "dt_comptc": pd.to_datetime(["2025-10-31"] * 4 + ["2025-11-30"] * 4 + ["2026-07-31"] * 2),
        "score": [0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.2, 0.5, 0.5],
        "y_true": [1, 0, 0, 0, 1, 1, 0, 0, 0, 0],
    })
    r = served_level_check(df, ["2025-10", "2025-11"], pd.Timestamp("2026-10-31"))
    assert [m["mes"] for m in r["por_mes"]] == ["2025-10", "2025-11"]  # 2026-07 fora (e sem janela completa)
    assert r["score_medio"] == pytest.approx(0.15) and r["taxa_eventos"] == pytest.approx(3 / 8)
    assert r["n_obs"] == 8


def test_pagina_avisa_quando_o_score_ao_vivo_esta_baixo() -> None:
    from fidc_risk.export.methodology_page import render_methodology_html

    s = _summary()
    s["nivel_modelo_servido"] = {"meses": ["2025-10", "2026-03"], "score_medio": 0.09, "taxa_eventos": 0.16,
                                 "n_obs": 15000, "calibracao": ["2026-04", "2026-09"],
                                 "por_mes": [{"mes": "2025-10", "score_medio": 0.075, "taxa_eventos": 0.154, "n_obs": 2545}]}
    html = render_methodology_html(s)
    assert "score do observatório está abaixo" in html
    corpo = re.sub(r'<section id="nomes-tecnicos".*?</section>', "", html, flags=re.S)
    texto = re.sub(r"<[^>]+>", " ", corpo).lower()
    for termo in JARGAO:
        assert not re.search(rf"\b{re.escape(termo)}\b", texto), termo

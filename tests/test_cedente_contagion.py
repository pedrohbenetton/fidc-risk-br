"""Estudo de contágio por cedente: eventos, exposição, controles, desfechos e efeito médio (dados sintéticos)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fidc_risk.studies.cedente_contagion import (
    attach_strata,
    control_events,
    exposure_events,
    matched_effect,
    onset_events,
    outcomes,
    verdict,
)


def _panel(rows: list[tuple[str, int, float]], seg: str = "Consignado", pl: float = 100e6) -> pd.DataFrame:
    return pd.DataFrame(
        [{"entity_id": e, "ym": ym, "v": v, "stress": v > 0.10, "pl": pl, "segmento": seg} for e, ym, v in rows]
    )


def _edges(rows: list[tuple[str, int, str]]) -> pd.DataFrame:
    return pd.DataFrame([{"entity_id": e, "ym": ym, "ced": c} for e, ym, c in rows])


def test_entrada_em_estresse_exige_mes_anterior_saudavel_e_observado() -> None:
    p = _panel([("A", 1, 0.05), ("A", 2, 0.12), ("A", 3, 0.13), ("B", 1, 0.20), ("B", 2, 0.30),
                ("C", 1, 0.02), ("C", 3, 0.15)])  # C pula o mês 2: sem evento
    ev = onset_events(p)
    assert list(map(tuple, ev[["entity_id", "ym"]].to_numpy())) == [("A", 2)]


def test_exposicao_e_por_cedente_compartilhado_com_classe_que_entrou_em_estresse() -> None:
    p = _panel([("A", 1, 0.05), ("A", 2, 0.12),  # A entra em estresse no mês 2
                ("B", 1, 0.01), ("B", 2, 0.02),  # B divide X com A: exposta
                ("C", 1, 0.01), ("C", 2, 0.02),  # C só tem Y: não exposta
                ("D", 1, 0.30), ("D", 2, 0.30)])  # D divide X, mas já está em estresse: fora
    e = _edges([("A", 2, "X"), ("B", 2, "X"), ("C", 2, "Y"), ("D", 2, "X")])
    exp = exposure_events(p, e, onset_events(p))
    assert list(map(tuple, exp[["entity_id", "ym"]].to_numpy())) == [("B", 2)]
    assert exp.iloc[0]["n_cedentes"] == 1


def test_so_conta_exposicao_nova_dentro_da_janela() -> None:
    rows = [("A", 1, 0.05), ("A", 2, 0.12), ("A2", 3, 0.05), ("A2", 4, 0.12)]
    rows += [("B", m, 0.01) for m in range(1, 6)]
    p = _panel(rows)
    e = _edges([("A", 2, "X"), ("B", 2, "X"), ("A2", 4, "X"), ("B", 4, "X")])
    exp = exposure_events(p, e, onset_events(p), fresh_window=6)
    assert list(exp["ym"]) == [2]  # a segunda exposição (mês 4) cai dentro de 6 meses da primeira


def test_controles_sao_saudaveis_com_cedente_informado_e_sem_exposicao() -> None:
    p = _panel([("A", 2, 0.12), ("A", 1, 0.05), ("B", 2, 0.01), ("C", 2, 0.01), ("E", 2, 0.01), ("F", 2, 0.40)])
    e = _edges([("A", 2, "X"), ("B", 2, "X"), ("C", 2, "Y"), ("F", 2, "Z")])  # E não informa cedente
    exp = exposure_events(p, e, onset_events(p))
    ctrl = control_events(p, e, exp)
    assert sorted(ctrl.loc[ctrl["ym"] == 2, "entity_id"]) == ["C"]


def test_desfechos_em_h_meses_e_censura_sem_observacao() -> None:
    p = _panel([("B", 2, 0.01), ("B", 3, 0.05), ("B", 4, 0.15), ("B", 5, 0.08),
                ("G", 2, 0.02), ("G", 3, 0.02)])  # G some depois do mês 3
    ev = pd.DataFrame({"entity_id": ["B", "G"], "ym": [2, 2]})
    o = outcomes(p, ev, horizons=(1, 3)).set_index("entity_id")
    assert o.loc["B", "y_h1"] == 0 and o.loc["B", "y_h3"] == 1
    assert o.loc["B", "dv_h3"] == pytest.approx(0.08 - 0.01)
    assert o.loc["G", "y_h1"] == 0 and np.isnan(o.loc["G", "y_h3"])


def _sint(n_por_estrato: int, taxa_exp: float, taxa_ctrl: float, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    exp, ctrl = [], []
    for ym in range(10):
        for i in range(n_por_estrato):
            exp.append({"entity_id": f"e{ym}_{i}", "ym": ym, "segmento": "S", "porte": 1, "y_h6": float(rng.random() < taxa_exp)})
            for j in range(4):
                ctrl.append({"entity_id": f"c{ym}_{i}_{j}", "ym": ym, "segmento": "S", "porte": 1,
                             "y_h6": float(rng.random() < taxa_ctrl)})
    return pd.DataFrame(exp), pd.DataFrame(ctrl)


def test_efeito_pareado_identifica_diferenca_e_ausencia_de_efeito() -> None:
    exp, ctrl = _sint(30, 1.0, 0.0)
    r = matched_effect(exp, ctrl, "y_h6", n_boot=50, seed=1)
    assert r["efeito"] == pytest.approx(1.0) and r["ic_baixo"] == pytest.approx(1.0)
    exp, ctrl = _sint(40, 0.2, 0.2, seed=3)
    r = matched_effect(exp, ctrl, "y_h6", n_boot=200, seed=1)
    assert r["ic_baixo"] < 0 < r["ic_alto"]


def test_estrato_sem_controles_suficientes_fica_de_fora() -> None:
    exp, ctrl = _sint(5, 0.5, 0.1)
    ctrl = ctrl[ctrl["ym"] != 0]
    r = matched_effect(exp, ctrl, "y_h6", n_boot=20, seed=1, min_controles=3)
    assert r["n_expostas"] == 45  # o mês 0 (5 eventos) sai por falta de controles


def test_porte_em_quintis_dentro_do_mes() -> None:
    p = pd.DataFrame({"entity_id": list("abcdefghij"), "ym": [1] * 10, "pl": np.arange(1, 11) * 1e6,
                      "segmento": ["S"] * 10, "v": [0.0] * 10, "stress": [False] * 10})
    s = attach_strata(p)
    assert sorted(s["porte"].unique()) == [1, 2, 3, 4, 5]
    assert s.loc[s["entity_id"] == "a", "porte"].item() == 1 and s.loc[s["entity_id"] == "j", "porte"].item() == 5


def test_veredito_so_e_robusto_com_efeito_positivo_estavel_e_sem_pre_tendencia() -> None:
    base = {"efeito_h6": {"efeito": 0.05, "ic_baixo": 0.02, "ic_alto": 0.08},
            "pre_tendencia": {"efeito": 0.001, "ic_baixo": -0.002, "ic_alto": 0.004},
            "sem_hubs_h6": {"efeito": 0.04, "ic_baixo": 0.01, "ic_alto": 0.07},
            "por_ano_h6": [{"efeito": 0.03}, {"efeito": 0.06}, {"efeito": 0.02}, {"efeito": -0.01}, {"efeito": 0.04}]}
    assert verdict(base)["robusto"] is True
    fraco = {**base, "efeito_h6": {"efeito": 0.01, "ic_baixo": -0.01, "ic_alto": 0.03}}
    assert verdict(fraco)["robusto"] is False
    pre = {**base, "pre_tendencia": {"efeito": 0.02, "ic_baixo": 0.01, "ic_alto": 0.03}}
    assert verdict(pre)["robusto"] is False


def test_pre_tendencia_negativa_nao_reprova_mas_nivel_de_atraso_sim() -> None:
    base = {"efeito_h6": {"efeito": 0.05, "ic_baixo": 0.02, "ic_alto": 0.08},
            "pre_tendencia": {"efeito": -0.005, "ic_baixo": -0.009, "ic_alto": -0.001},
            "sem_hubs_h6": {"efeito": 0.04, "ic_baixo": 0.01, "ic_alto": 0.07},
            "pareado_atraso_h6": {"efeito": 0.03, "ic_baixo": 0.01, "ic_alto": 0.05},
            "por_ano_h6": [{"efeito": 0.03}, {"efeito": 0.06}, {"efeito": 0.02}]}
    v = verdict(base)
    assert v["robusto"] is True and v["notas"]  # pré-tendência negativa vira nota, não reprovação
    sem_nivel = {**base, "pareado_atraso_h6": {"efeito": 0.01, "ic_baixo": -0.01, "ic_alto": 0.03}}
    assert verdict(sem_nivel)["robusto"] is False


def test_faixa_de_vencido_em_t_vira_estrato() -> None:
    p = pd.DataFrame({"entity_id": list("abcd"), "ym": [1] * 4, "pl": [1e6] * 4, "segmento": ["S"] * 4,
                      "v": [0.0, 0.01, 0.03, 0.08], "stress": [False] * 4})
    assert attach_strata(p)["faixa_v"].tolist() == [1, 2, 3, 4]


def test_faixa_do_atraso_medio_nos_6_meses_anteriores() -> None:
    from fidc_risk.studies.cedente_contagion import attach_prior_level

    rows = [("A", m, 0.0) for m in range(1, 8)] + [("B", m, 0.06) for m in range(1, 7)] + [("B", 7, 0.0)]
    rows += [("C", 6, 0.03), ("C", 7, 0.01)]  # só 1 mês de histórico: sem faixa
    p = attach_prior_level(_panel(rows))
    f = p.set_index(["entity_id", "ym"])["faixa_v_previa"]
    assert f[("A", 7)] == 1 and f[("B", 7)] == 4 and f[("C", 7)] == 0


def test_veredito_exige_resistir_ao_historico_de_atraso() -> None:
    base = {"efeito_h6": {"efeito": 0.05, "ic_baixo": 0.02, "ic_alto": 0.08},
            "pre_tendencia": {"efeito": 0.0, "ic_baixo": -0.01, "ic_alto": 0.01},
            "sem_hubs_h6": {"efeito": 0.04, "ic_baixo": 0.01, "ic_alto": 0.07},
            "pareado_atraso_h6": {"efeito": 0.03, "ic_baixo": 0.01, "ic_alto": 0.05},
            "pareado_historico_h6": {"efeito": 0.01, "ic_baixo": -0.01, "ic_alto": 0.03},
            "por_ano_h6": [{"efeito": 0.03}, {"efeito": 0.06}]}
    v = verdict(base)
    assert v["robusto"] is False and any("histórico" in m for m in v["motivos"])

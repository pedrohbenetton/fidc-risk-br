"""Pseudonimização do pacote público: códigos estáveis, sem nome/CNPJ/gestor e sem PL exato.

Todos os fundos, gestoras e cedentes deste arquivo são fictícios (CNPJs com dígitos repetidos).
"""

from __future__ import annotations

import copy
import json
import random
import re
from pathlib import Path

import pytest

from fidc_risk.export.anonymize import (
    SENSITIVE_SHAP_FEATURES,
    CodeBook,
    anonymize_bundle,
    faixa_pl,
)

CNPJ_FMT = re.compile(r"\d{2}\.\d{3}\.\d{3}/\d{4}-\d{2}")
CNPJ_RAW = re.compile(r"(?<![\d.])\d{14}(?![\d.])")
NOMES_REAIS = ["FIDC FICTICIO ALFA", "FIDC FICTICIO BETA", "FIDC FORA DA LISTA", "GESTORA FICTICIA UM", "GESTORA FICTICIA DOIS"]


def _fund(eid: str, name: str, gestor: str, pl_m: float, prob: float) -> dict:
    cnpj = f"{eid[:2]}.{eid[2:5]}.{eid[5:8]}/{eid[8:12]}-{eid[12:]}"
    return {
        "entity_id": eid,
        "cnpj": cnpj,
        "name": name,
        "gestor": gestor,
        "sector": "Consignado",
        "is_aberto": 1,
        "pl_m": pl_m,
        "prob_6m": prob,
        "va_p0": prob - 0.01,
        "va_p1": prob + 0.01,
        "inad90_pct": 4.2,
        "shap": [
            {"feature": "pl_log", "label": "Porte (log do PL)", "shap_log_odds": 0.1, "feature_value": 18.37},
            {"feature": "gestor_other_funds_cnt", "label": "Outros fundos do gestor", "shap_log_odds": 0.05, "feature_value": 37.0},
            {"feature": "pdd_carteira_pct", "label": "PDD / carteira", "shap_log_odds": 0.3, "feature_value": 0.12},
        ],
        "history": [{"dt": "2026-09", "inad90": 4.0, "early": 1.0, "pdd": 3.0, "subord": 20.0, "pl_m": pl_m * 0.98}],
    }


def _bundle() -> dict:
    return {
        "meta": {"project": "fidc-risk-br", "latest_competence": "2026-10-31"},
        "kpis": {"unique_gestores": 2},
        "watchlist_live": [
            _fund("11111111000111", "FIDC FICTICIO ALFA", "GESTORA FICTICIA UM LTDA", 35.04, 0.6923),
            _fund("22222222000122", "FIDC FICTICIO BETA", "GESTORA FICTICIA DOIS LTDA", 1234.56, 0.4),
        ],
        "contagion_graph": {
            "nodes": [
                {"id": "C_99999999000199", "kind": "cedente", "label": "Cedente 99.999.999/0001-99", "degree": 2,
                 "exposure_m": 7.3, "risk": 0.47},
                {"id": "F_11111111000111", "kind": "fidc", "entity_id": "11111111000111", "cnpj": "11.111.111/0001-11",
                 "label": "FIDC FICTICIO ALFA", "full_name": "FIDC FICTICIO ALFA", "sector": "Consignado",
                 "gestor": "GESTORA FICTICIA UM LTDA", "pl_m": 35.04, "risk": 0.69, "inad90": 4.2, "subord": 20.0},
                {"id": "F_33333333000133", "kind": "fidc", "entity_id": "33333333000133", "cnpj": "33.333.333/0001-33",
                 "label": "FIDC FORA DA LISTA", "full_name": "FIDC FORA DA LISTA", "sector": "Factoring",
                 "gestor": "GESTORA FICTICIA UM LTDA", "pl_m": 10.0, "risk": 0.1, "inad90": 1.0, "subord": 30.0},
            ],
            "links": [
                {"source": "C_99999999000199", "target": "F_11111111000111", "share_pct": 15.0, "exp_m": 5.26},
                {"source": "C_99999999000199", "target": "F_33333333000133", "share_pct": 20.0, "exp_m": 2.0},
            ],
        },
    }


@pytest.fixture
def book() -> CodeBook:
    return CodeBook(rng=random.Random(7))


def test_nenhum_nome_cnpj_ou_documento_sobrevive(book: CodeBook) -> None:
    out = json.dumps(anonymize_bundle(_bundle(), book), ensure_ascii=False)
    assert not CNPJ_FMT.search(out)
    assert not CNPJ_RAW.search(out)
    for nome in NOMES_REAIS:
        assert nome not in out.upper()


def test_codigos_no_formato_fundo_a001_e_gestora_g01(book: CodeBook) -> None:
    out = anonymize_bundle(_bundle(), book)
    for f in out["watchlist_live"]:
        assert re.fullmatch(r"A\d{3,}", f["entity_id"])
        assert f["name"] == f"Fundo {f['entity_id']}"
        assert re.fullmatch(r"Gestora G\d{2,}", f["gestor"])
        assert "cnpj" not in f


def test_codigos_estaveis_entre_execucoes_com_mapeamento_persistido(tmp_path: Path) -> None:
    path = tmp_path / "private" / "demo_codes.json"
    first = CodeBook(rng=random.Random(1))
    out1 = anonymize_bundle(_bundle(), first)
    first.save(path)

    # Outra execução, outro gerador aleatório: os códigos vêm do arquivo, não da ordem nem do acaso.
    second = CodeBook.load(path, rng=random.Random(999))
    out2 = anonymize_bundle(_bundle(), second)
    assert out1 == out2


def test_fundo_novo_recebe_codigo_novo_sem_mudar_os_antigos(book: CodeBook) -> None:
    antes = {f["name"]: f["entity_id"] for f in anonymize_bundle(_bundle(), book)["watchlist_live"]}
    b = _bundle()
    b["watchlist_live"].insert(0, _fund("44444444000144", "FIDC NOVO GAMA", "GESTORA FICTICIA UM LTDA", 50.0, 0.9))
    depois = anonymize_bundle(b, book)["watchlist_live"]
    codigos = [f["entity_id"] for f in depois]
    assert codigos[1:] == list(antes.values())
    assert codigos[0] not in antes.values()


def test_mesma_gestora_mesmo_codigo_na_lista_e_no_grafo(book: CodeBook) -> None:
    out = anonymize_bundle(_bundle(), book)
    alfa = out["watchlist_live"][0]
    node = next(n for n in out["contagion_graph"]["nodes"] if n.get("entity_id") == alfa["entity_id"])
    assert node["gestor"] == alfa["gestor"]
    fora = next(n for n in out["contagion_graph"]["nodes"] if n["kind"] == "fidc" and n["entity_id"] != alfa["entity_id"])
    assert fora["gestor"] == alfa["gestor"]  # mesma gestora fictícia
    assert alfa["gestor"] != out["watchlist_live"][1]["gestor"]


def test_ids_do_grafo_consistentes_com_a_lista(book: CodeBook) -> None:
    out = anonymize_bundle(_bundle(), book)
    graph = out["contagion_graph"]
    node_ids = {n["id"] for n in graph["nodes"]}
    assert f"F_{out['watchlist_live'][0]['entity_id']}" in node_ids
    for link in graph["links"]:
        assert link["source"] in node_ids and link["target"] in node_ids
    ced = next(n for n in graph["nodes"] if n["kind"] == "cedente")
    assert re.fullmatch(r"Cedente C\d{3,}", ced["label"])


def test_pl_exato_sai_e_entra_faixa(book: CodeBook) -> None:
    out = anonymize_bundle(_bundle(), book)
    alfa, beta = out["watchlist_live"]
    assert "pl_m" not in alfa and "pl_m" not in beta
    assert alfa["pl_faixa"] == "R$ 25–50 mi"
    assert beta["pl_faixa"] == "R$ 1–2,5 bi"
    assert alfa["pl_ordem"] < beta["pl_ordem"]
    assert all("pl_m" not in h for h in alfa["history"])
    for n in out["contagion_graph"]["nodes"]:
        assert "pl_m" not in n and "exposure_m" not in n
    # exp_m / share_pct devolveria o PL exato do fundo
    assert all("exp_m" not in link for link in out["contagion_graph"]["links"])


def test_valores_shap_que_reidentificam_sao_removidos(book: CodeBook) -> None:
    out = anonymize_bundle(_bundle(), book)
    for d in out["watchlist_live"][0]["shap"]:
        if d["feature"] in SENSITIVE_SHAP_FEATURES:
            assert d["feature_value"] is None
        else:
            assert d["feature_value"] == 0.12
    assert {"pl_log", "gestor_other_funds_cnt"} <= SENSITIVE_SHAP_FEATURES


def test_metricas_de_risco_e_setor_continuam_reais(book: CodeBook) -> None:
    out = anonymize_bundle(_bundle(), book)
    alfa = out["watchlist_live"][0]
    assert alfa["prob_6m"] == 0.6923
    assert alfa["sector"] == "Consignado"
    assert alfa["inad90_pct"] == 4.2
    assert alfa["history"] == [{"dt": "2026-09", "inad90": 4.0, "early": 1.0, "pdd": 3.0, "subord": 20.0}]
    assert out["meta"]["anonymized"] is True and out["meta"]["demo"] is True


def test_entrada_nao_e_alterada(book: CodeBook) -> None:
    original = _bundle()
    snapshot = copy.deepcopy(original)
    anonymize_bundle(original, book)
    assert original == snapshot


@pytest.mark.parametrize(
    ("pl_m", "faixa"),
    [
        (0.0, "até R$ 10 mi"),
        (9.99, "até R$ 10 mi"),
        (10.0, "R$ 10–25 mi"),
        (99.9, "R$ 50–100 mi"),
        (100.0, "R$ 100–250 mi"),
        (250.0, "R$ 250–500 mi"),
        (999.0, "R$ 500 mi–1 bi"),
        (1000.0, "R$ 1–2,5 bi"),
        (2500.0, "acima de R$ 2,5 bi"),
        (90_000.0, "acima de R$ 2,5 bi"),
    ],
)
def test_faixas_de_pl(pl_m: float, faixa: str) -> None:
    assert faixa_pl(pl_m)[1] == faixa


def test_mapeamento_salvo_nao_vai_para_o_git() -> None:
    import subprocess

    from fidc_risk.config import DEMO_CODES_PATH, PRIVATE_BUNDLE_PATH, PROJECT_ROOT

    for p in (DEMO_CODES_PATH, PRIVATE_BUNDLE_PATH):
        rel = p.relative_to(PROJECT_ROOT).as_posix()
        try:
            r = subprocess.run(["git", "check-ignore", "-q", rel], cwd=PROJECT_ROOT, check=False)
        except FileNotFoundError:
            pytest.skip("git indisponível")
        assert r.returncode == 0, f"{rel} precisa estar no .gitignore"

"""Relatório de carteira para uma gestora (relatorio.carteira): validação da entrada, números e PDF.

Fundos fictícios do mini-informe (tests/cvm_mini.py) com CNPJs de dígitos verificadores válidos.
F2 está em estresse (40% vencido 90+) e fica fora da carteira na maior parte dos testes: o relatório
pode contar classes em estresse de fora da carteira, mas nunca identificá-las.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pymupdf
import pytest

from relatorio import carteira as cart
from relatorio import indicadores
from tests import cvm_mini as mini


def _cnpj_valido(base12: str) -> str:
    return base12 + cart.digitos_verificadores(base12)


# CNPJs fictícios com DV válido no lugar dos do mini-informe.
V1, V2, V3, V4 = (_cnpj_valido(b) for b in ("111111110001", "222222220001", "333333330001", "444444440001"))
REMAP = {mini.F1: cart.formatar_cnpj(V1), mini.F2: cart.formatar_cnpj(V2), mini.F3: cart.formatar_cnpj(V3),
         mini.F4: cart.formatar_cnpj(V4)}
FORA_DA_BASE = _cnpj_valido("555555550001")


@pytest.fixture
def dados(tmp_path: Path) -> cart.DadosMes:
    cur = mini.write_mini_competencia(tmp_path / "202610", "202610", REMAP)
    prev = mini.write_mini_competencia(tmp_path / "202609", "202609", REMAP)
    scores = pd.DataFrame({
        "k": [V1, V2, V3],
        "score": [0.12, 0.55, 0.04],
        "p0": [0.10, 0.50, 0.03],
        "p1": [0.15, 0.60, 0.05],
        "validado": [True, False, True],
    }).set_index("k")
    prev_scores = scores.assign(score=[0.05, 0.50, 0.04])
    return cart.DadosMes(
        competencia="2026-10",
        competencia_anterior="2026-09",
        cur=indicadores.por_classe(str(cur)),
        prev=indicadores.por_classe(str(prev)),
        ced_cur=indicadores.cedentes_pj(str(cur)),
        ced_prev=indicadores.cedentes_pj(str(prev)),
        scores=scores,
        scores_prev=prev_scores,
        baixado_em="22/11/2026",
    )


def _csv(tmp_path: Path, linhas: list[str], header: str = "cnpj") -> Path:
    p = tmp_path / "carteira.csv"
    p.write_text("\n".join([header, *linhas]) + "\n", encoding="utf-8")
    return p


# ------------------------------------------------------------------ CNPJ

def test_digitos_verificadores_numericos_e_alfanumericos() -> None:
    assert cart.cnpj_valido("11.222.333/0001-81")
    assert not cart.cnpj_valido("11.222.333/0001-82")
    assert not cart.cnpj_valido("11.111.111/1111-11")  # sequência repetida
    assert cart.cnpj_valido("12.ABC.345/01DE-35")  # exemplo oficial do CNPJ alfanumérico (Receita Federal)
    assert not cart.cnpj_valido("12.ABC.345/01DE-36")
    assert not cart.cnpj_valido("1234")


def test_cnpj_invalido_na_carteira_falha_apontando_a_linha(tmp_path: Path) -> None:
    p = _csv(tmp_path, [cart.formatar_cnpj(V1), "11.222.333/0001-82"])
    with pytest.raises(cart.CarteiraError, match=r"linha 3.*11\.222\.333/0001-82"):
        cart.ler_carteira(p)


def test_carteira_vazia_falha(tmp_path: Path) -> None:
    with pytest.raises(cart.CarteiraError, match="vazia"):
        cart.ler_carteira(_csv(tmp_path, []))


def test_carteira_aceita_csv_sem_cabecalho_e_remove_repetidos(tmp_path: Path) -> None:
    p = tmp_path / "c.csv"
    p.write_text(f"{V1}\n{cart.formatar_cnpj(V1)}\n{V3};x\n", encoding="utf-8")
    assert cart.ler_carteira(p) == [V1, V3]


def test_cli_devolve_erro_claro_para_cnpj_invalido(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    p = _csv(tmp_path, ["00.000.000/0001-00"])
    code = cart.main(["--cnpjs", str(p), "--competencia", "2026-10", "--out", str(tmp_path / "r.pdf")])
    assert code == 2
    assert "inválido" in capsys.readouterr().err
    assert not (tmp_path / "r.pdf").exists()


# ------------------------------------------------------------------ números

def test_resumo_e_linhas_conferidos_a_mao(dados: cart.DadosMes) -> None:
    rel = cart.montar_relatorio([V1, V3], dados)
    r = rel["resumo"]
    assert r["fundos_informados"] == 2 and r["fundos_encontrados"] == 2
    assert r["pl_total"] == pytest.approx(1000 + 2100)
    assert r["carteira_total"] == pytest.approx(1000 + 2000)
    assert r["vencido90_ponderado"] == pytest.approx((50 + 0) / 3000)  # ponderado pela carteira
    assert r["em_estresse"] == 0
    linhas = {f["cnpj"]: f for f in rel["fundos"]}
    assert linhas[V1]["vencido90"] == pytest.approx(0.05)
    assert linhas[V1]["pdd"] == pytest.approx(30 / 1000)
    assert linhas[V1]["score"] == pytest.approx(0.12) and linhas[V1]["faixa"] == pytest.approx((0.10, 0.15))
    assert linhas[V1]["delta_score"] == pytest.approx(0.07)
    assert [f["cnpj"] for f in rel["fundos"]] == [V1, V3]  # maior score primeiro


def test_contagio_conta_classes_em_estresse_de_fora_sem_identificar(dados: cart.DadosMes) -> None:
    rel = cart.montar_relatorio([V1, V3], dados)
    por_cedente = {c["cedente"]: c for c in rel["contagio"]}
    # X liga F1 a F2 (em estresse, fora); Y liga F3 a F2; Z liga F3 a F4 (sem estresse) e não entra.
    x, y = cart.formatar_cnpj("99999999000191"), cart.formatar_cnpj("88888888000188")
    assert set(por_cedente) == {x, y}
    assert por_cedente[x]["fundos_carteira"] == [mini.NOMES[mini.F1]]
    assert por_cedente[x]["classes_estresse_fora"] == 1
    texto = repr(rel)
    assert V2 not in texto and cart.formatar_cnpj(V2) not in texto and mini.NOMES[mini.F2] not in texto


def test_fundo_em_estresse_na_carteira_sem_score_validado(dados: cart.DadosMes) -> None:
    rel = cart.montar_relatorio([V2], dados)
    (linha,) = rel["fundos"]
    assert linha["em_estresse"] is True
    assert linha["status_score"] == "fora_do_dominio"
    assert rel["resumo"]["em_estresse"] == 1
    assert rel["contagio"] == []  # o único cedente em estresse é o próprio fundo da carteira


def test_cnpj_fora_da_base_fica_listado_e_fora_das_somas(dados: cart.DadosMes) -> None:
    rel = cart.montar_relatorio([V1, FORA_DA_BASE], dados)
    assert rel["nao_encontrados"] == [FORA_DA_BASE]
    assert rel["resumo"]["fundos_encontrados"] == 1
    assert rel["resumo"]["pl_total"] == pytest.approx(1000)


def test_nenhum_fundo_na_base_falha(dados: cart.DadosMes) -> None:
    with pytest.raises(cart.CarteiraError, match="nenhum"):
        cart.montar_relatorio([FORA_DA_BASE], dados)


def test_fundo_sem_score_aparece_com_status(dados: cart.DadosMes) -> None:
    rel = cart.montar_relatorio([V4], dados)
    (linha,) = rel["fundos"]
    assert linha["status_score"] == "sem_score" and linha["score"] is None


def test_mudancas_no_mes(dados: cart.DadosMes) -> None:
    prev = dados.prev.copy()
    prev.loc[V3, ["vencido90_pct", "em_estresse"]] = [0.25, True]  # F3 saiu do estresse
    prev.loc[V1, "vencido90_pct"] = 0.01  # F1 subiu 4 p.p.
    d = cart.DadosMes(**{**dados.__dict__, "prev": prev})
    m = cart.montar_relatorio([V1, V3], d)["mudancas"]
    assert m["sairam_estresse"] == [mini.NOMES[mini.F3]]
    assert m["entraram_estresse"] == []
    assert [x["nome"] for x in m["vencido_subiu"]] == [mini.NOMES[mini.F1]]
    assert [x["nome"] for x in m["score_subiu"]] == [mini.NOMES[mini.F1]]  # 5% -> 12%


# ------------------------------------------------------------------ PDF

def _pdf_texto(p: Path) -> tuple[int, str]:
    with pymupdf.open(p) as doc:
        return len(doc), "".join(pg.get_text() for pg in doc)


def test_pdf_da_carteira_com_um_fundo_so(dados: cart.DadosMes, tmp_path: Path) -> None:
    out = cart.gerar_pdf(cart.montar_relatorio([V1], dados), tmp_path / "relatorio.pdf", marca="Marca Teste")
    paginas, texto = _pdf_texto(out)
    assert paginas == 2
    for trecho in ("Resumo da carteira", "Contágio", "O que mudou", mini.NOMES[mini.F1], "Confidencial"):
        assert trecho in texto


def test_pdf_lista_cnpj_fora_da_base(dados: cart.DadosMes, tmp_path: Path) -> None:
    out = cart.gerar_pdf(cart.montar_relatorio([V1, V3, FORA_DA_BASE], dados), tmp_path / "r.pdf")
    paginas, texto = _pdf_texto(out)
    assert 2 <= paginas <= 3
    assert cart.formatar_cnpj(FORA_DA_BASE) in texto
    assert mini.NOMES[mini.F2] not in texto and cart.formatar_cnpj(V2) not in texto


def test_exemplo_de_carteira_tem_10_cnpjs_validos() -> None:
    from fidc_risk.config import PROJECT_ROOT

    cnpjs = cart.ler_carteira(PROJECT_ROOT / "exemplos" / "carteira_exemplo.csv")
    assert len(cnpjs) == 10


def test_nome_curto_abrevia_e_cabe_na_coluna() -> None:
    from reportlab.pdfbase.pdfmetrics import stringWidth

    nome = "FUNDO DE INVESTIMENTO EM DIREITOS CREDITÓRIOS NÃO PADRONIZADOS ALFA - RESPONSABILIDADE LIMITADA"
    assert cart.nome_curto(nome) == "FIDC NP ALFA - RL"
    assert cart.nome_curto("FUNDO DE  INVESTIMENTO EM DIREITOS CREDITÓRIOS ECO") == "FIDC ECO"
    curto = cart.nome_curto(nome * 2, largura_pt=100)
    assert curto.endswith("…") and stringWidth(curto, "Helvetica", 6.6) <= 100


def test_carteira_grande_cabe_em_tres_paginas(dados: cart.DadosMes, tmp_path: Path) -> None:
    """Com mais de 15 fundos a tabela vira uma linha por fundo; 60 fundos ainda cabem em 3 páginas."""
    import pandas as pd

    base = dados.cur.loc[[V1]]
    clones = pd.concat([base.rename(index={V1: _cnpj_valido(f"{n:08d}0001")}) for n in range(10, 70)])
    clones["nome"] = [f"FUNDO DE INVESTIMENTO EM DIREITOS CREDITÓRIOS EXEMPLO {i} - RESPONSABILIDADE LIMITADA"
                      for i in range(len(clones))]
    d = cart.DadosMes(**{**dados.__dict__, "cur": pd.concat([dados.cur, clones])})
    out = cart.gerar_pdf(cart.montar_relatorio(list(clones.index), d), tmp_path / "grande.pdf")
    paginas, _ = _pdf_texto(out)
    assert 2 <= paginas <= 3

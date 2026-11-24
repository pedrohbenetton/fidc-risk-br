"""Testes do boletim mensal (relatorio/): indicadores conferidos à mão e regressão do PDF de outubro/2026."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import pymupdf
import pytest

from relatorio import gerar_pdf, indicadores
from tests.cvm_mini import EXPECTED_MONTH, write_mini_competencia

FIXTURES = Path(__file__).parent / "fixtures" / "relatorio"


def _assert_close(got: object, exp: object, path: str = "") -> None:
    if isinstance(exp, dict):
        assert isinstance(got, dict) and set(got) == set(exp), f"{path}: chaves {got!r} != {exp!r}"
        for k in exp:
            _assert_close(got[k], exp[k], f"{path}.{k}")
    elif isinstance(exp, list):
        assert isinstance(got, list) and len(got) == len(exp), f"{path}: {got!r} != {exp!r}"
        for i, (g, e) in enumerate(zip(got, exp, strict=True)):
            _assert_close(g, e, f"{path}[{i}]")
    elif isinstance(exp, float):
        assert math.isclose(float(got), exp, rel_tol=1e-12, abs_tol=1e-15), f"{path}: {got!r} != {exp!r}"  # type: ignore[arg-type]
    else:
        assert got == exp, f"{path}: {got!r} != {exp!r}"


@pytest.fixture
def mini_dir(tmp_path: Path) -> Path:
    return write_mini_competencia(tmp_path / "202610", "202610")


def test_month_confere_com_calculo_manual(mini_dir: Path) -> None:
    _assert_close(indicadores.month(str(mini_dir)), EXPECTED_MONTH)


def test_competencia_e_lida_do_nome_dos_arquivos(mini_dir: Path) -> None:
    assert indicadores.competencia_do_dir(mini_dir) == "2026-10"


def test_competencia_sem_arquivos_da_cvm_falha_com_mensagem(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="inf_mensal_fidc_tab_I"):
        indicadores.competencia_do_dir(tmp_path)


def test_main_grava_json_rotulado_pelas_competencias(tmp_path: Path) -> None:
    cur = write_mini_competencia(tmp_path / "a", "202610")
    prev = write_mini_competencia(tmp_path / "b", "202609")
    yoy = write_mini_competencia(tmp_path / "c", "202510")
    out = tmp_path / "indicadores.json"

    res = indicadores.main(str(cur), str(prev), str(yoy), str(out))

    assert list(res) == ["2026-10", "2026-09", "2025-10"]
    saved = json.loads(out.read_text(encoding="utf-8"))
    assert list(saved) == ["2026-10", "2026-09", "2025-10"]
    _assert_close(saved["2025-10"], EXPECTED_MONTH)


def _pdf_text(path: Path) -> list[str]:
    with pymupdf.open(path) as doc:
        return [page.get_text() for page in doc]


def _numeros(paginas: list[str]) -> list[str]:
    return re.findall(r"[−+-]?\d[\d.,]*", "\n".join(paginas))


def _trecho(paginas: list[str], inicio: str, fim: str) -> str:
    texto = "\n".join(paginas)
    return texto[texto.index(inicio):texto.index(fim)]


def test_pdf_de_outubro_mantem_os_numeros_e_o_texto_do_original(tmp_path: Path) -> None:
    """Mesmos indicadores do boletim enviado em novembro/2026: manchete, destaques, painel e todos os números
    iguais. Mudam as cores dos gráficos, as legendas que as descrevem e a tabela por segmento na página 2."""
    out = gerar_pdf.gerar(
        FIXTURES / "indicadores_2026-10.json",
        marca="@pedrohbenetton",
        out_dir=tmp_path,
        baixado_em="22/11/2026",
    )
    assert out == tmp_path / "boletim-fidc-2026-10.pdf"
    novo = _pdf_text(out)
    # Texto extraído do PDF enviado em novembro/2026, com a marca trocada pelo @ do GitHub.
    original = (FIXTURES / "boletim-fidc-2026-10_original.txt").read_text(encoding="utf-8").split("")
    assert len(novo) == 2
    assert _trecho(novo, "@pedrohbenetton · Boletim de", "Painel do mês") == _trecho(original, "@pedrohbenetton · Boletim de", "Painel do mês")
    assert _trecho(novo, "Painel do mês", "Vencido há mais de 90 dias, por segmento") == _trecho(
        original, "Painel do mês", "Vencido há mais de 90 dias, por segmento")
    # Todos os números do original aparecem, na mesma ordem (a página 2 ganhou uma tabela por segmento).
    restantes = iter(_numeros(novo))
    faltando = [n for n in _numeros(original) if not any(n == m for m in restantes)]
    assert faltando == []


def test_pdf_nao_deixa_arquivos_fora_do_diretorio_de_saida(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    gerar_pdf.gerar(FIXTURES / "indicadores_2026-10.json", marca="X", out_dir=tmp_path / "saida", baixado_em="22/11/2026")
    assert list(cwd.iterdir()) == []
    assert (tmp_path / "saida" / "boletim-fidc-2026-10.pdf").exists()

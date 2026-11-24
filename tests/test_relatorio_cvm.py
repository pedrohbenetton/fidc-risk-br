"""Arquivos do Informe Mensal da CVM por competência: datas, extração do zip e cache local."""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from relatorio import cvm


@pytest.mark.parametrize(
    ("comp", "meses", "esperado"),
    [("2026-10", 1, "2026-09"), ("2026-01", 1, "2025-12"), ("2026-10", 12, "2025-10"), ("2026-03", 15, "2024-12")],
)
def test_competencia_anterior(comp: str, meses: int, esperado: str) -> None:
    assert cvm.competencia_anterior(comp, meses) == esperado


@pytest.mark.parametrize("ruim", ["2026-13", "2026-1", "202610", "", "out/2026"])
def test_competencia_invalida(ruim: str) -> None:
    with pytest.raises(ValueError, match="competência inválida"):
        cvm.validar_competencia(ruim)


def _zip(path: Path, ym: str, extras: dict[str, str] | None = None) -> Path:
    with zipfile.ZipFile(path, "w") as z:
        z.writestr(f"inf_mensal_fidc_tab_I_{ym}.csv", "CNPJ_FUNDO_CLASSE;DENOM_SOCIAL\n")
        z.writestr(f"inf_mensal_fidc_tab_IV_{ym}.csv", "CNPJ_FUNDO_CLASSE;TAB_IV_A_VL_PL\n")
        for nome, conteudo in (extras or {}).items():
            z.writestr(nome, conteudo)
    return path


def test_extrai_so_os_csvs_da_competencia_e_ignora_caminhos(tmp_path: Path) -> None:
    z = _zip(tmp_path / "inf.zip", "202610", {
        "../../fora.csv": "x",
        "sub/inf_mensal_fidc_tab_V_202610.csv": "ok",
        "inf_mensal_fidc_tab_I_202609.csv": "outro mês",
        "leiame.txt": "x",
    })
    destino = cvm.extrair_zip(z, tmp_path / "out", "202610")
    assert sorted(p.name for p in destino.iterdir()) == [
        "inf_mensal_fidc_tab_IV_202610.csv", "inf_mensal_fidc_tab_I_202610.csv", "inf_mensal_fidc_tab_V_202610.csv",
    ]
    assert not (tmp_path / "fora.csv").exists()


def test_diretorio_vem_do_cache_ou_do_zip_baixado(tmp_path: Path) -> None:
    bronze, cache, hist = tmp_path / "bronze", tmp_path / "cache", tmp_path / "hist"
    bronze.mkdir()
    kw = {"cache_dir": cache, "bronze_dir": bronze, "hist_dir": hist}
    with pytest.raises(FileNotFoundError, match="2026-10"):
        cvm.diretorio_competencia("2026-10", **kw)
    _zip(bronze / "inf_mensal_fidc_202610.zip", "202610")
    d = cvm.diretorio_competencia("2026-10", **kw)
    assert d == cache / "202610" and (d / "inf_mensal_fidc_tab_I_202610.csv").exists()
    (bronze / "inf_mensal_fidc_202610.zip").unlink()
    assert cvm.diretorio_competencia("2026-10", **kw) == d  # cache basta
    assert cvm.data_extracao("2026-10", **kw) is not None

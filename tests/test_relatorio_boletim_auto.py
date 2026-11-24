"""Boletim mensal automático (relatorio.boletim): download da CVM simulado, texto gerado pelos dados e PDF."""

from __future__ import annotations

import datetime as dt
import io
import json
import zipfile
from pathlib import Path

import httpx
import pymupdf
import pytest

from relatorio import boletim, cvm, gerar_pdf
from tests.cvm_mini import write_mini_competencia

HOJE = dt.date(2026, 11, 22)


def _zip_bytes(tmp: Path, aaaamm: str) -> bytes:
    d = write_mini_competencia(tmp / f"src_{aaaamm}", aaaamm)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for f in d.iterdir():
            z.write(f, arcname=f.name)
    return buf.getvalue()


class FakeCVM:
    """Servidor da CVM simulado: devolve o zip mensal das competências conhecidas e 404 para o resto."""

    def __init__(self, tmp: Path, mensais: list[str], anuais: dict[str, list[str]] | None = None) -> None:
        self.mensais = {m: _zip_bytes(tmp, m) for m in mensais}
        self.anuais = {}
        for ano, meses in (anuais or {}).items():
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w") as z:
                for m in meses:
                    with zipfile.ZipFile(io.BytesIO(_zip_bytes(tmp, m))) as zm:
                        for n in zm.namelist():
                            z.writestr(n, zm.read(n))
            self.anuais[ano] = buf.getvalue()
        self.pedidos: list[str] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        self.pedidos.append(url)
        nome = url.rsplit("/", 1)[-1]
        if "/HIST/" in url:
            ano = nome.removeprefix("inf_mensal_fidc_").removesuffix(".zip")
            return httpx.Response(200, content=self.anuais[ano]) if ano in self.anuais else httpx.Response(404)
        ym = nome.removeprefix("inf_mensal_fidc_").removesuffix(".zip")
        return httpx.Response(200, content=self.mensais[ym]) if ym in self.mensais else httpx.Response(404)

    def client(self) -> httpx.Client:
        return httpx.Client(transport=httpx.MockTransport(self.handler))


@pytest.fixture
def dirs(tmp_path: Path) -> dict[str, Path]:
    return {"bronze": tmp_path / "bronze", "hist": tmp_path / "hist", "cache": tmp_path / "cache", "out": tmp_path / "out"}


def test_baixa_so_o_que_falta(tmp_path: Path, dirs: dict[str, Path]) -> None:
    cvm_fake = FakeCVM(tmp_path, ["202610"])
    with cvm_fake.client() as c:
        p = cvm.baixar_competencia("2026-10", bronze_dir=dirs["bronze"], hist_dir=dirs["hist"], client=c, hoje=HOJE)
        assert p == dirs["bronze"] / "inf_mensal_fidc_202610.zip" and zipfile.is_zipfile(p)
        cvm.baixar_competencia("2026-10", bronze_dir=dirs["bronze"], hist_dir=dirs["hist"], client=c, hoje=HOJE)
    assert len(cvm_fake.pedidos) == 1  # a segunda chamada usa o arquivo já baixado


def test_atualizar_baixa_de_novo(tmp_path: Path, dirs: dict[str, Path]) -> None:
    cvm_fake = FakeCVM(tmp_path, ["202610"])
    with cvm_fake.client() as c:
        for _ in range(2):
            cvm.baixar_competencia("2026-10", bronze_dir=dirs["bronze"], hist_dir=dirs["hist"], client=c, hoje=HOJE,
                                   forcar=True)
    assert len(cvm_fake.pedidos) == 2


def test_mes_antigo_vem_do_arquivo_anual(tmp_path: Path, dirs: dict[str, Path]) -> None:
    cvm_fake = FakeCVM(tmp_path, [], anuais={"2024": ["202403", "202404"]})
    with cvm_fake.client() as c:
        p = cvm.baixar_competencia("2024-03", bronze_dir=dirs["bronze"], hist_dir=dirs["hist"], client=c, hoje=HOJE)
    assert p == dirs["hist"] / "inf_mensal_fidc_2024.zip"
    d = cvm.diretorio_competencia("2024-03", cache_dir=dirs["cache"], bronze_dir=dirs["bronze"], hist_dir=dirs["hist"])
    assert (d / "inf_mensal_fidc_tab_I_202403.csv").exists()


def test_competencia_nao_publicada_falha_com_mensagem_clara(tmp_path: Path, dirs: dict[str, Path]) -> None:
    cvm_fake = FakeCVM(tmp_path, ["202609"])
    with cvm_fake.client() as c, pytest.raises(cvm.CompetenciaNaoPublicada, match="ainda não publicou.*outubro/2026"):
        cvm.baixar_competencia("2026-10", bronze_dir=dirs["bronze"], hist_dir=dirs["hist"], client=c, hoje=HOJE)
    assert not (dirs["bronze"] / "inf_mensal_fidc_202610.zip").exists()


def test_mes_que_ainda_nao_terminou_falha_sem_ir_a_cvm(tmp_path: Path, dirs: dict[str, Path]) -> None:
    cvm_fake = FakeCVM(tmp_path, ["202611"])
    with cvm_fake.client() as c, pytest.raises(cvm.CompetenciaNaoPublicada, match="novembro/2026 ainda não terminou"):
        cvm.baixar_competencia("2026-11", bronze_dir=dirs["bronze"], hist_dir=dirs["hist"], client=c, hoje=HOJE)
    assert cvm_fake.pedidos == []


def test_resposta_que_nao_e_zip_e_rejeitada(dirs: dict[str, Path]) -> None:
    transport = httpx.MockTransport(lambda r: httpx.Response(200, content=b"<html>manutencao</html>"))
    with httpx.Client(transport=transport) as c, pytest.raises(RuntimeError, match="não é um zip"):
        cvm.baixar_competencia("2026-10", bronze_dir=dirs["bronze"], hist_dir=dirs["hist"], client=c, hoje=HOJE)
    assert not (dirs["bronze"] / "inf_mensal_fidc_202610.zip").exists()


# ------------------------------------------------------------------ comando completo

def _rodar(tmp_path: Path, dirs: dict[str, Path], cvm_fake: FakeCVM, comp: str, bloco: Path | None = None) -> int:
    with cvm_fake.client() as c:
        return boletim.main(
            [comp, "--marca", "Marca Teste", "--out-dir", str(dirs["out"])],
            client=c, bronze_dir=dirs["bronze"], hist_dir=dirs["hist"], cache_dir=dirs["cache"], hoje=HOJE,
            bloco_path=bloco or tmp_path / "sem_bloco.json",
        )


def test_comando_baixa_tres_competencias_calcula_e_gera_pdf(tmp_path: Path, dirs: dict[str, Path]) -> None:
    cvm_fake = FakeCVM(tmp_path, ["202610", "202609", "202510"])
    assert _rodar(tmp_path, dirs, cvm_fake, "2026-10") == 0
    assert sorted(p.rsplit("_", 1)[-1] for p in cvm_fake.pedidos) == ["202510.zip", "202609.zip", "202610.zip"]
    ind = json.loads((dirs["out"] / "indicadores-2026-10.json").read_text(encoding="utf-8"))
    assert list(ind) == ["2026-10", "2026-09", "2025-10"]
    with pymupdf.open(dirs["out"] / "boletim-fidc-2026-10.pdf") as doc:
        assert len(doc) == 2
        texto = "".join(p.get_text() for p in doc)
    assert "Marca Teste" in texto and "Outubro de 2026" in texto


def test_comando_inclui_o_bloco_do_estudo_de_contagio(tmp_path: Path, dirs: dict[str, Path]) -> None:
    bloco = tmp_path / "bloco.json"
    bloco.write_text(json.dumps({
        "titulo": "Cedente em comum: o risco viaja",
        "texto": "Texto do estudo com 12,0% contra 9,8%.",
        "rotulos": ["Expostas ao cedente", "Classes parecidas sem a ligação"],
        "taxas_6m": [0.12, 0.098], "efeito_6m": 0.022, "ic_6m": [0.002, 0.044], "forca": "pequeno",
    }), encoding="utf-8")
    cvm_fake = FakeCVM(tmp_path, ["202610", "202609", "202510"])
    assert _rodar(tmp_path, dirs, cvm_fake, "2026-10", bloco) == 0
    with pymupdf.open(dirs["out"] / "boletim-fidc-2026-10.pdf") as doc:
        assert len(doc) == 2
        texto = "".join(p.get_text() for p in doc)
    assert "Cedente em comum: o risco viaja" in texto


def test_comando_falha_claro_quando_a_cvm_nao_publicou(tmp_path: Path, dirs: dict[str, Path],
                                                      capsys: pytest.CaptureFixture[str]) -> None:
    cvm_fake = FakeCVM(tmp_path, ["202609", "202510"])
    assert _rodar(tmp_path, dirs, cvm_fake, "2026-10") == 2
    err = capsys.readouterr().err
    assert "ainda não publicou" in err and "outubro/2026" in err
    assert not (dirs["out"] / "boletim-fidc-2026-10.pdf").exists()


# ------------------------------------------------------------------ texto e cores

def _ind(inad30: float, inad90: float, inad30_prev: float, inad90_prev: float, fin: float = 9.0) -> dict:
    base = {"pl_bi": 900.0, "n_classes": 4000, "inad180_pct": 1.0, "pdd_pct": 6.0, "recompra_mi": 100.0,
            "recompra_pct": 0.1, "classes_estresse": 500, "classes_estresse_pct": 15.0, "carteira_dc_bi": 800.0,
            "segmentos": [{"seg": "Financeiro", "carteira_bi": 300.0, "inad90_pct": fin, "n": 900},
                          {"seg": "Industrial", "carteira_bi": 100.0, "inad90_pct": 2.0, "n": 300}]}
    return {"2026-10": {**base, "inad30_pct": inad30, "inad90_pct": inad90},
            "2026-09": {**base, "inad30_pct": inad30_prev, "inad90_pct": inad90_prev},
            "2025-10": {**base, "inad30_pct": inad30_prev, "inad90_pct": inad90_prev}}


def test_manchete_segue_os_dados() -> None:
    assert gerar_pdf.manchete(_ind(8.0, 4.0, 7.5, 4.0)) == (
        "Outubro de 2026: o atraso curto subiu, e o segmento financeiro concentra o problema")
    assert gerar_pdf.manchete(_ind(7.0, 4.0, 7.5, 4.0)).startswith("Outubro de 2026: o atraso curto caiu")
    estavel = gerar_pdf.manchete(_ind(7.51, 4.3, 7.5, 4.0, fin=4.4))
    assert estavel == "Outubro de 2026: o atraso curto ficou estável, e o vencido acima de 90 dias subiu"


def test_cores_vermelho_so_para_estresse() -> None:
    cores = gerar_pdf.cores_distribuicao(["0", "0–2", "2–5", "5–10", "10–20", "20–50", ">50"])
    assert cores[:4] == [gerar_pdf.NAVY] * 4
    assert cores[4:] == [gerar_pdf.VERMELHO] * 3
    assert gerar_pdf.VERMELHO == "#8e2323" and gerar_pdf.NAVY == "#1f3a5f"


def test_manchete_junta_quando_os_dois_atrasos_vao_no_mesmo_sentido() -> None:
    assert gerar_pdf.manchete(_ind(8.0, 4.4, 7.5, 4.0, fin=4.5)) == (
        "Outubro de 2026: o atraso curto e o vencido acima de 90 dias subiram")


def test_destaques_nao_repetem_sinal_nem_somam_verbo_a_numero_com_sinal() -> None:
    r = _ind(7.0, 4.0, 7.22, 4.17, fin=4.1)
    for comp in r.values():
        comp.update({"pdd_pct": 6.0, "ops_bi": {}, "dist_inad90": []})
    texto = gerar_pdf.destaques(r)
    assert "caiu 0,22 p.p." in texto and "caiu −" not in texto
    assert "em queda no mês (−0,17 p.p.)" in texto

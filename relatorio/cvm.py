"""Informe Mensal de FIDC da CVM por competência: CSVs extraídos num cache local (data/cvm_csv/AAAAMM/)."""
from __future__ import annotations

import datetime as dt
import re
import zipfile
from pathlib import Path

import httpx

from fidc_risk.config import BRONZE_DIR, CVM_FIDC_HIST_BASE, CVM_FIDC_MONTHLY_BASE, DATA_DIR

CACHE_DIR = DATA_DIR / "cvm_csv"
BRONZE_MONTHLY_DIR = BRONZE_DIR / "cvm_fidc_monthly"
BRONZE_HIST_DIR = BRONZE_DIR / "cvm_fidc_hist"
MESES = ["janeiro", "fevereiro", "março", "abril", "maio", "junho", "julho", "agosto", "setembro", "outubro",
         "novembro", "dezembro"]
USER_AGENT = "fidc-risk-br/1.0 (boletim mensal)"


class CompetenciaNaoPublicada(RuntimeError):
    """A CVM ainda não disponibilizou o informe da competência pedida."""


def mes_longo(comp: str) -> str:
    """'2026-10' -> 'outubro/2026'."""
    validar_competencia(comp)
    return f"{MESES[int(comp[5:7]) - 1]}/{comp[:4]}"
_COMP_RE = re.compile(r"^(\d{4})-(0[1-9]|1[0-2])$")


def validar_competencia(comp: str) -> str:
    if not _COMP_RE.match(comp or ""):
        raise ValueError(f"competência inválida: {comp!r} (use AAAA-MM, ex.: 2026-10)")
    return comp


def aaaamm(comp: str) -> str:
    """'2026-10' -> '202610'."""
    validar_competencia(comp)
    return comp.replace("-", "")


def competencia_anterior(comp: str, meses: int = 1) -> str:
    """'2026-10' -> '2026-09' (meses=1) ou '2025-10' (meses=12)."""
    validar_competencia(comp)
    n = int(comp[:4]) * 12 + int(comp[5:7]) - 1 - meses
    return f"{n // 12:04d}-{n % 12 + 1:02d}"


def _tem_csvs(d: Path, ym: str) -> bool:
    return (d / f"inf_mensal_fidc_tab_I_{ym}.csv").exists()


def extrair_zip(zip_path: Path, destino: Path, ym: str) -> Path:
    """Extrai só os CSVs inf_mensal_fidc_tab_*_AAAAMM.csv, pelo nome do arquivo (sem seguir caminhos do zip)."""
    alvo = re.compile(rf"^inf_mensal_fidc_tab_[A-Z0-9_]+_{ym}\.csv$", re.IGNORECASE)
    destino.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        for info in z.infolist():
            nome = Path(info.filename).name
            if info.is_dir() or not alvo.match(nome):
                continue
            (destino / nome).write_bytes(z.read(info))
    if not _tem_csvs(destino, ym):
        raise FileNotFoundError(f"{zip_path.name} não contém a tabela I da competência {ym}")
    return destino


def _zip_local(comp: str, bronze_dir: Path, hist_dir: Path) -> Path | None:
    """Zip já baixado que contém a competência: o mensal ou, para meses antigos, o anual (HIST)."""
    ym = aaaamm(comp)
    mensal = bronze_dir / f"inf_mensal_fidc_{ym}.zip"
    if mensal.exists():
        return mensal
    anual = hist_dir / f"inf_mensal_fidc_{comp[:4]}.zip"
    if anual.exists():
        with zipfile.ZipFile(anual) as z:
            if any(Path(n).name.lower() == f"inf_mensal_fidc_tab_i_{ym}.csv" for n in z.namelist()):
                return anual
    return None


def diretorio_competencia(
    comp: str, cache_dir: Path = CACHE_DIR, bronze_dir: Path = BRONZE_MONTHLY_DIR, hist_dir: Path = BRONZE_HIST_DIR
) -> Path:
    """Diretório com os CSVs da competência; extrai do zip já baixado quando preciso."""
    ym = aaaamm(comp)
    d = cache_dir / ym
    if _tem_csvs(d, ym):
        return d
    z = _zip_local(comp, bronze_dir, hist_dir)
    if z is not None:
        return extrair_zip(z, d, ym)
    raise FileNotFoundError(
        f"informe da competência {comp} não encontrado em {bronze_dir}; rode `python -m relatorio.boletim` "
        "(baixa da CVM) ou `fidc-risk ingest`"
    )


def _baixar(client: httpx.Client, url: str, destino: Path) -> bool:
    """Baixa ``url`` para ``destino`` (gravação atômica). Devolve False em 404; erro claro para o resto."""
    resp = client.get(url, headers={"User-Agent": USER_AGENT})
    if resp.status_code == 404:
        return False
    if resp.status_code != 200:
        raise RuntimeError(f"a CVM respondeu {resp.status_code} para {url}")
    conteudo = resp.content
    if not conteudo.startswith(b"PK"):
        raise RuntimeError(f"o arquivo recebido de {url} não é um zip (a CVM pode estar em manutenção)")
    destino.parent.mkdir(parents=True, exist_ok=True)
    tmp = destino.with_suffix(".part")
    tmp.write_bytes(conteudo)
    if not zipfile.is_zipfile(tmp):
        tmp.unlink()
        raise RuntimeError(f"o arquivo recebido de {url} não é um zip válido")
    tmp.replace(destino)
    return True


def baixar_competencia(
    comp: str,
    bronze_dir: Path = BRONZE_MONTHLY_DIR,
    hist_dir: Path = BRONZE_HIST_DIR,
    client: httpx.Client | None = None,
    hoje: dt.date | None = None,
    forcar: bool = False,
) -> Path:
    """Garante o zip da CVM com a competência; baixa só se faltar (ou se ``forcar``).

    Tenta o arquivo mensal (DADOS/) e, se não existir, o anual (DADOS/HIST/). Falha com
    CompetenciaNaoPublicada se o mês ainda não terminou ou se a CVM ainda não publicou o informe.
    """
    validar_competencia(comp)
    hoje = hoje or dt.date.today()
    ano, mes = int(comp[:4]), int(comp[5:7])
    if (ano, mes) >= (hoje.year, hoje.month):
        raise CompetenciaNaoPublicada(
            f"a competência {mes_longo(comp)} ainda não terminou: o informe só existe depois do fim do mês "
            "(e a CVM dá até 15 dias para o envio)"
        )
    if not forcar:
        local = _zip_local(comp, bronze_dir, hist_dir)
        if local is not None:
            return local
    ym = aaaamm(comp)
    proprio = client is None
    c = client or httpx.Client(timeout=httpx.Timeout(60.0, connect=20.0), follow_redirects=True)
    try:
        url_mensal = f"{CVM_FIDC_MONTHLY_BASE}inf_mensal_fidc_{ym}.zip"
        mensal = bronze_dir / f"inf_mensal_fidc_{ym}.zip"
        if _baixar(c, url_mensal, mensal):
            return mensal
        url_anual = f"{CVM_FIDC_HIST_BASE}inf_mensal_fidc_{ano}.zip"
        anual = hist_dir / f"inf_mensal_fidc_{ano}.zip"
        if _baixar(c, url_anual, anual):
            local = _zip_local(comp, bronze_dir, hist_dir)
            if local is not None:
                return local
    finally:
        if proprio:
            c.close()
    raise CompetenciaNaoPublicada(
        f"a CVM ainda não publicou o informe de {mes_longo(comp)} ({url_mensal} respondeu 404). O prazo de envio vai "
        "até 15 dias depois do fim do mês e o arquivo aberto aparece em seguida; tente de novo mais tarde"
    )


def data_extracao(
    comp: str, cache_dir: Path = CACHE_DIR, bronze_dir: Path = BRONZE_MONTHLY_DIR, hist_dir: Path = BRONZE_HIST_DIR
) -> str | None:
    """Data (DD/MM/AAAA) em que o arquivo da CVM da competência foi baixado: a do zip, senão a do CSV."""
    ym = aaaamm(comp)
    z = _zip_local(comp, bronze_dir, hist_dir)
    for f in (z, cache_dir / ym / f"inf_mensal_fidc_tab_I_{ym}.csv"):
        if f is None:
            continue
        if f.exists():
            return dt.datetime.fromtimestamp(f.stat().st_mtime).strftime("%d/%m/%Y")
    return None

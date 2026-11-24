"""Boletim mensal automático: baixa o que falta da CVM, calcula os indicadores e gera o PDF.

Uso: python -m relatorio.boletim 2026-11 --marca "Sua marca" [--out-dir .] [--atualizar]

Usa a competência pedida, a anterior e a de 12 meses antes. Arquivos já baixados são reaproveitados
(``--atualizar`` baixa de novo a competência pedida, que a CVM atualiza com reenvios). Se a CVM ainda
não publicou a competência, o comando falha com uma mensagem clara e não gera nada.
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

import httpx

from fidc_risk.config import PROJECT_ROOT
from relatorio import cvm, gerar_pdf, indicadores

BLOCO_CONTAGIO = PROJECT_ROOT / "relatorio" / "blocos" / "contagio_cedente.json"
COMPLETUDE_MINIMA = 0.85  # menos classes que isso em relação ao mês anterior sugere arquivo ainda incompleto


def main(
    argv: list[str] | None = None,
    client: httpx.Client | None = None,
    bronze_dir: Path = cvm.BRONZE_MONTHLY_DIR,
    hist_dir: Path = cvm.BRONZE_HIST_DIR,
    cache_dir: Path = cvm.CACHE_DIR,
    hoje: dt.date | None = None,
    bloco_path: Path = BLOCO_CONTAGIO,
) -> int:
    ap = argparse.ArgumentParser(description="Boletim mensal de crédito em FIDCs (PDF de 2 páginas).")
    ap.add_argument("competencia", help="AAAA-MM, ex.: 2026-10")
    ap.add_argument("--marca", default="[Sua marca]")
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--atualizar", action="store_true", help="baixa de novo o informe da competência pedida")
    a = ap.parse_args(argv)

    try:
        comp = cvm.validar_competencia(a.competencia)
        comps = [comp, cvm.competencia_anterior(comp, 1), cvm.competencia_anterior(comp, 12)]
        dirs = []
        for c in comps:
            cvm.baixar_competencia(c, bronze_dir=bronze_dir, hist_dir=hist_dir, client=client, hoje=hoje,
                                   forcar=a.atualizar and c == comp)
            if a.atualizar and c == comp:
                # descarta CSVs antigos do cache para extrair a versão recém-baixada
                for f in (cache_dir / cvm.aaaamm(c)).glob("*.csv"):
                    f.unlink()
            dirs.append(str(cvm.diretorio_competencia(c, cache_dir=cache_dir, bronze_dir=bronze_dir, hist_dir=hist_dir)))
    except cvm.CompetenciaNaoPublicada as exc:
        print(f"erro: {exc}", file=sys.stderr)
        return 2
    except (ValueError, FileNotFoundError, RuntimeError, httpx.HTTPError) as exc:
        print(f"erro: {exc}", file=sys.stderr)
        return 2

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    res = indicadores.main(dirs[0], dirs[1], dirs[2], str(out / f"indicadores-{comp}.json"))

    avisos = []
    n_cur, n_prev = res[comp]["n_classes"], res[comps[1]]["n_classes"]
    if n_prev and n_cur < COMPLETUDE_MINIMA * n_prev:
        avisos.append(
            f"o informe de {cvm.mes_longo(comp)} tem {n_cur} classes contra {n_prev} no mês anterior: o arquivo da CVM "
            "pode ainda estar incompleto (fundos que não enviaram no prazo). Os números podem mudar."
        )
        print(f"aviso: {avisos[-1]}", file=sys.stderr)

    pdf = gerar_pdf.gerar(
        res,
        marca=a.marca,
        out_dir=out,
        baixado_em=cvm.data_extracao(comp, cache_dir=cache_dir, bronze_dir=bronze_dir, hist_dir=hist_dir),
        bloco=gerar_pdf.carregar_bloco(bloco_path),
        avisos=avisos,
    )
    print(f"ok: {pdf}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

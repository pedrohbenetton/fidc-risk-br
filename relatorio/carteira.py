"""Relatório da carteira de uma gestora: score de estresse, atraso, PDD, contágio por cedente e mudanças no mês.

Uso: python -m relatorio.carteira --cnpjs carteira.csv --competencia 2026-10 --out relatorio.pdf [--marca "..."]

O relatório é privado para a gestora cliente e cobre só os fundos da carteira dela. Classes de fora da
carteira aparecem apenas como contagem (ex.: "2 classes em estresse usam este cedente"), nunca por nome
ou CNPJ. Vencido 90+, PDD e estresse seguem as definições do boletim mensal (tabelas V+VI da CVM).
"""
from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from relatorio import cvm, indicadores

# Paleta do observatório (web/styles.css)
PAPEL, FOLHA, TINTA, TINTA2, TINTA3, NAVY, AMBAR, VERMELHO, FIO = (
    "#f4f2ec", "#fbfaf7", "#1b2433", "#4a5160", "#7a7f88", "#1f3a5f", "#d9902b", "#8e2323", "#d9d8d2",
)
MESES = ["janeiro", "fevereiro", "março", "abril", "maio", "junho", "julho", "agosto", "setembro", "outubro",
         "novembro", "dezembro"]
LIMIAR_SCORE_PP = 0.05  # variação de score que entra em "o que mudou"
LIMIAR_VENCIDO_PP = 0.01  # variação de vencido 90+ que entra em "o que mudou"
COMPACTO_ACIMA_DE = 15  # a partir daqui, uma linha por fundo


class CarteiraError(ValueError):
    """Entrada inválida (CNPJ inválido, carteira vazia, nenhum fundo na base)."""


# ------------------------------------------------------------------ CNPJ (numérico e alfanumérico)

def normalizar_cnpj(s: str) -> str:
    return "".join(ch for ch in str(s).upper() if ch.isascii() and ch.isalnum())


def digitos_verificadores(base12: str) -> str:
    """DVs do CNPJ (módulo 11). Vale também para o CNPJ alfanumérico: cada caractere vale ord(c) - 48."""
    vals = [ord(c) - 48 for c in base12]
    dvs = ""
    for pesos in ([5, 4, 3, 2, 9, 8, 7, 6, 5, 4, 3, 2], [6, 5, 4, 3, 2, 9, 8, 7, 6, 5, 4, 3, 2]):
        r = sum(v * p for v, p in zip(vals, pesos, strict=True)) % 11
        dv = 0 if r < 2 else 11 - r
        dvs += str(dv)
        vals.append(dv)
    return dvs


def cnpj_valido(s: str) -> bool:
    c = normalizar_cnpj(s)
    if len(c) != 14 or not c[12:].isdigit() or not all(ch.isdigit() or "A" <= ch <= "Z" for ch in c[:12]):
        return False
    if len(set(c)) == 1:
        return False
    return digitos_verificadores(c[:12]) == c[12:]


def formatar_cnpj(c: str) -> str:
    c = normalizar_cnpj(c)
    return f"{c[:2]}.{c[2:5]}.{c[5:8]}/{c[8:12]}-{c[12:]}" if len(c) == 14 else c


def ler_carteira(path: str | Path) -> list[str]:
    """Lê o CSV da carteira (uma coluna de CNPJs, com ou sem cabeçalho); falha se algum CNPJ for inválido."""
    p = Path(path)
    if not p.exists():
        raise CarteiraError(f"arquivo da carteira não encontrado: {p}")
    texto = p.read_text(encoding="utf-8-sig", errors="replace")
    cnpjs: list[str] = []
    invalidos: list[str] = []
    for n, linha in enumerate(texto.splitlines(), start=1):
        valor = re.split(r"[;,\t]", linha, maxsplit=1)[0].strip().strip('"').strip()
        if not valor:
            continue
        if n == 1 and not any(ch.isdigit() for ch in valor):
            continue  # cabeçalho
        if not cnpj_valido(valor):
            invalidos.append(f"linha {n}: {valor}")
            continue
        c = normalizar_cnpj(valor)
        if c not in cnpjs:
            cnpjs.append(c)
    if invalidos:
        raise CarteiraError("CNPJ inválido na carteira (" + "; ".join(invalidos) + ")")
    if not cnpjs:
        raise CarteiraError(f"carteira vazia: nenhum CNPJ em {p}")
    return cnpjs


# ------------------------------------------------------------------ dados do mês

@dataclass
class DadosMes:
    competencia: str
    competencia_anterior: str
    cur: pd.DataFrame  # indicadores.por_classe da competência
    prev: pd.DataFrame  # idem, mês anterior
    ced_cur: pd.DataFrame  # indicadores.cedentes_pj
    ced_prev: pd.DataFrame
    scores: pd.DataFrame  # índice k; colunas score, p0, p1, validado
    scores_prev: pd.DataFrame
    baixado_em: str | None = None
    avisos: list[str] = field(default_factory=list)


def pontuar(competencia: str) -> pd.DataFrame:
    """Score do modelo servido (o mesmo do observatório) para todas as classes da competência no painel Gold."""
    import joblib

    from fidc_risk.config import GOLD_DIR, MODELS_DIR

    vazio = pd.DataFrame(columns=["score", "p0", "p1", "validado"], index=pd.Index([], name="k"))
    painel_path, modelo_path = GOLD_DIR / "panel_gold.parquet", MODELS_DIR / "served_tabular_h6.pkl"
    if not painel_path.exists() or not modelo_path.exists():
        return vazio
    fim_mes = pd.Timestamp(f"{competencia}-01") + pd.offsets.MonthEnd(0)
    painel = pd.read_parquet(painel_path, filters=[("dt_comptc", "==", fim_mes)])
    if painel.empty:
        return vazio
    served = joblib.load(modelo_path)
    feats: list[str] = served["feature_cols"]
    elig = (painel["is_fidc_np_structural"] == 0) & (painel["vl_pl"] >= 5_000_000.0) & (painel["vl_carteira"] >= 2_000_000.0)
    sub = painel[elig]
    if sub.empty:
        return vazio
    raw = served["lgbm_monotonic"].predict_proba(sub[feats])[:, 1]
    va = served["venn_abers"].predict_full(raw)
    return pd.DataFrame(
        {"score": va.prob, "p0": va.p0, "p1": va.p1, "validado": (sub["is_eligible_watchlist"] == 1).to_numpy()},
        index=pd.Index(sub["entity_id"].astype(str).to_numpy(), name="k"),
    )


def carregar_mes(competencia: str) -> DadosMes:
    """Lê os CSVs da CVM da competência e do mês anterior e calcula os scores pelo painel Gold."""
    anterior = cvm.competencia_anterior(competencia)
    d_cur = str(cvm.diretorio_competencia(competencia))
    avisos = []
    try:
        d_prev: str | None = str(cvm.diretorio_competencia(anterior))
    except FileNotFoundError:
        d_prev = None
        avisos.append(f"informe de {anterior} indisponível: sem comparação com o mês anterior")
    vazio = pd.DataFrame(columns=["em_estresse", "vencido90_pct"])
    scores = pontuar(competencia)
    if scores.empty:
        avisos.append(f"painel do modelo sem a competência {competencia}: scores indisponíveis (rode o pipeline)")
    return DadosMes(
        competencia=competencia,
        competencia_anterior=anterior,
        cur=indicadores.por_classe(d_cur),
        prev=indicadores.por_classe(d_prev) if d_prev else vazio,
        ced_cur=indicadores.cedentes_pj(d_cur),
        ced_prev=indicadores.cedentes_pj(d_prev) if d_prev else pd.DataFrame(columns=["k", "ced", "pct"]),
        scores=scores,
        scores_prev=pontuar(anterior),
        baixado_em=cvm.data_extracao(competencia),
        avisos=avisos,
    )


# ------------------------------------------------------------------ relatório (dict)

def _f(x: Any) -> float | None:
    return None if x is None or pd.isna(x) else float(x)


def _contagio(carteira: list[str], tabela: pd.DataFrame, ced: pd.DataFrame) -> list[dict[str, Any]]:
    if ced.empty or tabela.empty:
        return []
    em_estresse = set(tabela.index[tabela["em_estresse"].astype(bool)])
    na_carteira = set(carteira)
    fora_em_estresse = em_estresse - na_carteira
    linhas: list[dict[str, Any]] = []
    for c, g in ced.groupby("ced"):
        meus = g[g["k"].isin(na_carteira)]
        if meus.empty:
            continue
        n_fora = int(g["k"].isin(fora_em_estresse).sum())
        if n_fora == 0:
            continue
        nomes = [str(tabela.at[k, "nome"]) if k in tabela.index else formatar_cnpj(k) for k in meus["k"]]
        pct_ok = meus["pct"].where((meus["pct"] > 0) & (meus["pct"] <= 100))  # >100% é erro de preenchimento
        linhas.append({
            "cedente": formatar_cnpj(str(c)),
            "fundos_carteira": nomes,
            "pct_max": _f(pct_ok.max()),
            "classes_estresse_fora": n_fora,
        })
    linhas.sort(key=lambda r: (-r["classes_estresse_fora"], -(r["pct_max"] or 0.0), r["cedente"]))
    return linhas


def montar_relatorio(cnpjs: list[str], dados: DadosMes) -> dict[str, Any]:
    """Números do relatório (sem formatação). Falha se nenhum CNPJ da carteira estiver na base do mês."""
    cur, prev = dados.cur, dados.prev
    achados = [k for k in cnpjs if k in cur.index]
    nao_encontrados = [k for k in cnpjs if k not in cur.index]
    if not achados:
        raise CarteiraError(
            f"nenhum fundo da carteira tem informe na competência {dados.competencia} ({len(cnpjs)} CNPJs informados)"
        )
    t = cur.loc[achados]
    carteira_total = float(t["carteira"].sum())
    fundos: list[dict[str, Any]] = []
    for k in achados:
        r = cur.loc[k]
        sc = dados.scores.loc[k] if k in dados.scores.index else None
        sp = dados.scores_prev.loc[k] if k in dados.scores_prev.index else None
        vp = _f(prev.at[k, "vencido90_pct"]) if k in prev.index else None
        v = _f(r["vencido90_pct"])
        status = "sem_score" if sc is None else ("ok" if bool(sc["validado"]) else "fora_do_dominio")
        fundos.append({
            "cnpj": k,
            "nome": str(r["nome"]),
            "pl": float(r["pl"]),
            "segmento": None if pd.isna(r["segmento"]) else str(r["segmento"]),
            "score": None if sc is None else float(sc["score"]),
            "faixa": None if sc is None else (float(sc["p0"]), float(sc["p1"])),
            "status_score": status,
            "delta_score": None if sc is None or sp is None else float(sc["score"]) - float(sp["score"]),
            "vencido90": v,
            "atraso_1_60": _f(r["atraso_1_60_pct"]),
            "pdd": _f(r["pdd_pct"]),
            "delta_vencido90": None if v is None or vp is None else v - vp,
            "em_estresse": bool(r["em_estresse"]),
        })
    fundos.sort(key=lambda f: (f["score"] is None, -(f["score"] or 0.0), f["nome"]))

    nomes = {f["cnpj"]: f["nome"] for f in fundos}
    mudancas: dict[str, Any] = {
        "entraram_estresse": [nomes[k] for k in achados if k in prev.index and bool(cur.at[k, "em_estresse"])
                              and not bool(prev.at[k, "em_estresse"])],
        "sairam_estresse": [nomes[k] for k in achados if k in prev.index and not bool(cur.at[k, "em_estresse"])
                            and bool(prev.at[k, "em_estresse"])],
        "score_subiu": [], "score_caiu": [], "vencido_subiu": [],
        "sem_informe": [formatar_cnpj(k) for k in nao_encontrados if k in prev.index],
    }
    for f in fundos:
        ds, dv = f["delta_score"], f["delta_vencido90"]
        if ds is not None and ds >= LIMIAR_SCORE_PP:
            mudancas["score_subiu"].append({"nome": f["nome"], "de": f["score"] - ds, "para": f["score"]})
        if ds is not None and ds <= -LIMIAR_SCORE_PP:
            mudancas["score_caiu"].append({"nome": f["nome"], "de": f["score"] - ds, "para": f["score"]})
        if dv is not None and dv >= LIMIAR_VENCIDO_PP:
            mudancas["vencido_subiu"].append({"nome": f["nome"], "de": f["vencido90"] - dv, "para": f["vencido90"]})
    for chave in ("score_subiu", "score_caiu", "vencido_subiu"):
        mudancas[chave].sort(key=lambda x: -abs(x["para"] - x["de"]))

    contagio = _contagio(achados, cur, dados.ced_cur)
    antes = {c["cedente"] for c in _contagio(achados, prev, dados.ced_prev)} if not prev.empty else set()
    mudancas["novos_cedentes_risco"] = [c["cedente"] for c in contagio if c["cedente"] not in antes] if not prev.empty else []

    return {
        "competencia": dados.competencia,
        "competencia_anterior": dados.competencia_anterior,
        "baixado_em": dados.baixado_em,
        "avisos": list(dados.avisos),
        "resumo": {
            "fundos_informados": len(cnpjs),
            "fundos_encontrados": len(achados),
            "pl_total": float(t["pl"].sum()),
            "carteira_total": carteira_total,
            "vencido90_ponderado": float(t["inad90"].sum()) / carteira_total if carteira_total > 0 else None,
            "em_estresse": int(t["em_estresse"].sum()),
        },
        "fundos": fundos,
        "nao_encontrados": nao_encontrados,
        "contagio": contagio,
        "mudancas": mudancas,
    }


# ------------------------------------------------------------------ PDF

def _br(v: float, d: int = 1) -> str:
    return f"{v:,.{d}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def _pct(v: float | None, d: int = 1) -> str:
    return "—" if v is None else f"{_br(v * 100, d)}%"


def _pp(v: float | None) -> str:
    if v is None:
        return "—"
    return ("+" if v >= 0 else "−") + _br(abs(v) * 100, 1) + " p.p."


def _reais(v: float) -> str:
    return f"R$ {_br(v / 1e9, 2)} bi" if abs(v) >= 1e9 else f"R$ {_br(v / 1e6, 1)} mi"


_ABREVIACOES = [
    (r"\s+", " "),  # antes de tudo: espaços duplos e não separáveis atrapalham as trocas abaixo
    (r"FUNDO DE INVESTIMENTO EM DIREITOS CREDIT[OÓ]RIOS", "FIDC"),
    (r"FUNDO DE INVESTIMENTO", "FI"),
    (r"DIREITOS CREDIT[OÓ]RIOS", "DC"),
    (r"N[AÃ]O[ -]PADRONIZADOS?", "NP"),
    (r"RESPONSABILIDADE LIMITADA", "RL"),
    (r"\s*-\s*", " - "),
    (r"\s+", " "),
]


def nome_curto(nome: str, largura_pt: float | None = None, fonte: str = "Helvetica", tamanho: float = 6.6) -> str:
    """Abrevia termos padrão do nome do fundo e, se preciso, corta para caber em ``largura_pt``."""
    s = str(nome).upper()
    for padrao, troca in _ABREVIACOES:
        s = re.sub(padrao, troca, s)
    s = s.strip(" -")
    if largura_pt is None:
        return s
    from reportlab.pdfbase.pdfmetrics import stringWidth

    if stringWidth(s, fonte, tamanho) <= largura_pt:
        return s
    while s and stringWidth(s + "…", fonte, tamanho) > largura_pt:
        s = s[:-1]
    return s.rstrip() + "…"


def _mes(comp: str) -> str:
    return f"{MESES[int(comp[5:7]) - 1]}/{comp[:4]}"


def gerar_pdf(rel: dict[str, Any], out: str | Path, marca: str = "Observatório de crédito em FIDCs") -> Path:
    """PDF A4 de 2 a 3 páginas na paleta do observatório."""
    from xml.sax.saxutils import escape

    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.platypus import (
        PageBreak,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )

    out_p = Path(out)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    C = colors.HexColor
    H1 = ParagraphStyle("h1", fontName="Times-Bold", fontSize=18, leading=22, textColor=C(TINTA))
    SUB = ParagraphStyle("sub", fontName="Helvetica", fontSize=8.5, leading=11.5, textColor=C(TINTA2))
    H2 = ParagraphStyle("h2", fontName="Times-Bold", fontSize=12, leading=15, textColor=C(TINTA), spaceBefore=10, spaceAfter=4)
    P = ParagraphStyle("p", fontName="Helvetica", fontSize=8.4, leading=11.4, textColor=C(TINTA), spaceAfter=3)
    SM = ParagraphStyle("sm", fontName="Helvetica", fontSize=6.8, leading=8.8, textColor=C(TINTA2))
    CELL = ParagraphStyle("cell", fontName="Helvetica", fontSize=7.6, leading=9.4, textColor=C(TINTA))

    r, comp = rel["resumo"], rel["competencia"]
    e = escape

    def base_style() -> list[Any]:
        return [
            ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 7.2), ("FONT", (0, 1), (-1, -1), "Helvetica", 7.6),
            ("TEXTCOLOR", (0, 0), (-1, 0), C(TINTA2)), ("TEXTCOLOR", (0, 1), (-1, -1), C(TINTA)),
            ("LINEBELOW", (0, 0), (-1, 0), 0.6, C(TINTA2)), ("LINEBELOW", (0, 1), (-1, -1), 0.3, C(FIO)),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ]

    # 1. Resumo
    resumo_rows = [
        ["Fundos da carteira", f"{r['fundos_encontrados']} com informe em {_mes(comp)}"
         + (f" ({r['fundos_informados'] - r['fundos_encontrados']} não encontrados)" if r["fundos_informados"] > r["fundos_encontrados"] else "")],
        ["Patrimônio somado", _reais(r["pl_total"])],
        ["Carteira de direitos creditórios", _reais(r["carteira_total"])],
        ["Vencido há mais de 90 dias (ponderado pela carteira)", _pct(r["vencido90_ponderado"], 2)],
        ["Fundos em estresse (vencido 90+ acima de 10%)", f"{r['em_estresse']} de {r['fundos_encontrados']}"],
    ]
    t_res = Table(resumo_rows, colWidths=[95 * mm, 83 * mm])
    t_res.setStyle(TableStyle([
        ("FONT", (0, 0), (0, -1), "Helvetica", 8.4), ("FONT", (1, 0), (1, -1), "Helvetica-Bold", 8.4),
        ("TEXTCOLOR", (0, 0), (0, -1), C(TINTA2)), ("TEXTCOLOR", (1, 0), (1, -1), C(TINTA)),
        ("LINEBELOW", (0, 0), (-1, -1), 0.3, C(FIO)), ("TOPPADDING", (0, 0), (-1, -1), 3.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5),
        *([("TEXTCOLOR", (1, 4), (1, 4), C(VERMELHO))] if r["em_estresse"] else []),
    ]))

    # 2. Uma linha por fundo (carteiras grandes: linha única, CNPJ em coluna própria, para caber em 2–3 páginas)
    compacto = len(rel["fundos"]) > COMPACTO_ACIMA_DE
    fonte = 6.6 if compacto else 7.6
    if compacto:
        head = ["Fundo", "CNPJ", "Score 6m", "Faixa", "Venc. 90+", "PDD", "Var. venc.", "Var. score"]
        larguras = [50 * mm, 25 * mm, 15 * mm, 19 * mm, 15 * mm, 13 * mm, 18 * mm, 17 * mm]
        col_venc = 4
    else:
        head = ["Fundo", "Score 6m", "Faixa", "Vencido 90+", "PDD", "Var. vencido", "Var. score"]
        larguras = [66 * mm, 17 * mm, 19 * mm, 19 * mm, 15 * mm, 22 * mm, 20 * mm]
        col_venc = 3
    rows: list[list[Any]] = [head]
    estilo_extra: list[Any] = []
    for i, f in enumerate(rel["fundos"], start=1):
        marca_score = {"ok": "", "fora_do_dominio": " †", "sem_score": ""}[f["status_score"]]
        faixa = "—" if f["faixa"] is None else f"{_pct(f['faixa'][0])}–{_pct(f['faixa'][1])}"
        numeros = [_pct(f["score"]) + marca_score, faixa, _pct(f["vencido90"]), _pct(f["pdd"]),
                   _pp(f["delta_vencido90"]), _pp(f["delta_score"])]
        if compacto:
            rows.append([e(nome_curto(f["nome"], larguras[0] - 6, tamanho=fonte)), formatar_cnpj(f["cnpj"]), *numeros])
        else:
            nome = Paragraph(
                f"{e(f['nome'][:70])}<br/><font size=6.4 color='{TINTA3}'>{formatar_cnpj(f['cnpj'])}"
                f"{' · ' + e(f['segmento']) if f['segmento'] else ''}</font>", CELL)
            rows.append([nome, *numeros])
        if f["em_estresse"]:
            estilo_extra.append(("TEXTCOLOR", (col_venc, i), (col_venc, i), C(VERMELHO)))
            estilo_extra.append(("FONT", (col_venc, i), (col_venc, i), "Helvetica-Bold", fonte))
    t_f = Table(rows, colWidths=larguras, repeatRows=1)
    estilo = base_style() + [("ALIGN", (2 if compacto else 1, 0), (-1, -1), "RIGHT")]
    if compacto:
        estilo += [("FONT", (0, 1), (-1, -1), "Helvetica", fonte), ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 6.6),
                   ("TOPPADDING", (0, 0), (-1, -1), 1.4), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.4),
                   ("TEXTCOLOR", (1, 1), (1, -1), C(TINTA3))]
    t_f.setStyle(TableStyle(estilo + estilo_extra))

    story: list[Any] = [
        Paragraph(f"{e(marca)} · Relatório de carteira · {_mes(comp)}", SUB),
        Paragraph(f"Sua carteira de FIDCs em {_mes(comp)}", H1),
        Paragraph("Confidencial: preparado para a gestora e restrito aos fundos da carteira informada. "
                  "Classes de fora da carteira aparecem só como contagem.", SUB),
        Paragraph("1. Resumo da carteira", H2), t_res,
        Paragraph("2. Fundo a fundo", H2),
        Paragraph("Ordenado pelo score: probabilidade estimada de o fundo entrar em estresse nos próximos 6 meses "
                  "(definição na Metodologia). Faixa: mínimo e máximo do score. Var.: variação contra "
                  f"{_mes(rel['competencia_anterior'])}. Em vermelho, vencido 90+ acima de 10% da carteira.", SM),
        Spacer(1, 3), t_f,
    ]
    notas = []
    if any(f["status_score"] == "fora_do_dominio" for f in rel["fundos"]):
        notas.append("† Score fora do domínio testado: na data, o fundo já tinha vencido 90+ ou PDD acima de 20% da "
                     "carteira, ou queda forte da cota sênior. O número aparece, mas não foi validado para esse caso.")
    if any(f["status_score"] == "sem_score" for f in rel["fundos"]):
        notas.append("— Sem score: fundo abaixo do porte mínimo (PL de R$ 5 mi e carteira de R$ 2 mi), FIDC não "
                     "padronizado estrutural ou fora do painel do modelo.")
    if rel["nao_encontrados"]:
        notas.append("Não encontrados no informe de " + _mes(comp) + ": "
                     + ", ".join(formatar_cnpj(c) for c in rel["nao_encontrados"]) + ".")
    story += [Spacer(1, 3), *[Paragraph(e(n), SM) for n in notas], PageBreak()]

    # 3. Contágio
    story.append(Paragraph("3. Contágio por cedente", H2))
    if rel["contagio"]:
        story.append(Paragraph(
            "Cedentes (pessoas jurídicas) que seus fundos dividem com classes em estresse de fora da carteira. "
            "Se o problema está no cedente, e não num fundo só, ele pode chegar aos seus.", P))
        crows: list[list[Any]] = [["Cedente", "Fundos da carteira expostos", "% da carteira (máx.)", "Classes em estresse fora"]]
        for c in rel["contagio"][:25]:
            crows.append([c["cedente"], Paragraph(e("; ".join(n[:60] for n in c["fundos_carteira"])), CELL),
                          _pct(None if c["pct_max"] is None else c["pct_max"] / 100), str(c["classes_estresse_fora"])])
        tc = Table(crows, colWidths=[34 * mm, 88 * mm, 26 * mm, 30 * mm], repeatRows=1)
        tc.setStyle(TableStyle(base_style() + [("ALIGN", (2, 0), (-1, -1), "RIGHT")]))
        story.append(tc)
        if len(rel["contagio"]) > 25:
            story.append(Paragraph(f"Mais {len(rel['contagio']) - 25} cedentes omitidos.", SM))
    else:
        story.append(Paragraph("Nenhum cedente da carteira aparece em classes em estresse de fora dela neste mês.", P))

    # 4. Mudanças
    m = rel["mudancas"]
    story.append(Paragraph(f"4. O que mudou desde {_mes(rel['competencia_anterior'])}", H2))
    itens = []
    if m["entraram_estresse"]:
        itens.append("<b>Entraram em estresse:</b> " + e("; ".join(m["entraram_estresse"])))
    if m["sairam_estresse"]:
        itens.append("<b>Saíram do estresse:</b> " + e("; ".join(m["sairam_estresse"])))
    for chave, rotulo, fmt in (("score_subiu", "Score subiu 5 p.p. ou mais", _pct), ("score_caiu", "Score caiu 5 p.p. ou mais", _pct),
                               ("vencido_subiu", "Vencido 90+ subiu 1 p.p. ou mais", _pct)):
        if m[chave]:
            itens.append(f"<b>{rotulo}:</b> " + "; ".join(f"{e(x['nome'][:60])} (de {fmt(x['de'])} para {fmt(x['para'])})" for x in m[chave][:8]))
    if m.get("novos_cedentes_risco"):
        itens.append("<b>Cedentes que passaram a ligar a carteira a classes em estresse:</b> " + ", ".join(m["novos_cedentes_risco"][:10]))
    if m["sem_informe"]:
        itens.append("<b>Sem informe neste mês (tinham no anterior):</b> " + ", ".join(m["sem_informe"]))
    if not itens:
        itens.append("Nenhuma mudança relevante: ninguém entrou ou saiu do estresse, e nenhum score ou vencido 90+ "
                     "variou além dos limites (5 p.p. e 1 p.p.).")
    story += [Paragraph("• " + i, P) for i in itens]

    # Notas
    baixado = f" Dados baixados em {rel['baixado_em']}." if rel.get("baixado_em") else ""
    story += [
        Paragraph("Como ler", H2),
        Paragraph(
            "Fonte: Informe Mensal de FIDC (CVM, dados abertos), competências " + _mes(comp) + " e "
            + _mes(rel["competencia_anterior"]) + "." + baixado + " Vencido 90+: faixas de atraso acima de 90 dias das "
            "tabelas V e VI sobre a carteira (a vencer + vencida). PDD: redução ao valor recuperável (tabela I) sobre a "
            "mesma carteira. Estresse: vencido 90+ acima de 10% da carteira. Cedentes: até 9 maiores por grupo (tabela I), "
            "só pessoas jurídicas. Score: probabilidade de estresse em 6 meses do modelo do observatório; a validação "
            "fora da amostra, os erros conhecidos e as limitações estão na Metodologia. O nível do score em produção está "
            "abaixo da frequência real de estresse (ajuste feito com meses ainda sem resultado completo): use-o para ordenar "
            "e priorizar os fundos, não como probabilidade exata.", SM),
        *[Paragraph(e("Aviso: " + a), SM) for a in rel.get("avisos", [])],
        Spacer(1, 4),
        Paragraph("<b>Confidencial.</b> Material de monitoramento para uso interno da gestora. Não é recomendação de "
                  "investimento e não substitui a análise de crédito.", SM),
    ]

    rodape = f"{marca} · Relatório de carteira · {_mes(comp)} · Confidencial"

    def footer(c: Any, doc: Any) -> None:
        c.setFont("Helvetica", 7)
        c.setFillColor(C(TINTA3))
        c.drawString(18 * mm, 10 * mm, rodape)
        c.drawRightString(A4[0] - 18 * mm, 10 * mm, str(doc.page))

    doc = SimpleDocTemplate(str(out_p), pagesize=A4, leftMargin=16 * mm, rightMargin=16 * mm, topMargin=14 * mm,
                            bottomMargin=16 * mm, title=f"Relatório de carteira · {_mes(comp)}")
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return out_p


# ------------------------------------------------------------------ CLI

def gerar_relatorio(cnpjs_csv: str | Path, competencia: str, out: str | Path, marca: str,
                    dados: DadosMes | None = None) -> Path:
    cvm.validar_competencia(competencia)
    cnpjs = ler_carteira(cnpjs_csv)
    dados = dados or carregar_mes(competencia)
    return gerar_pdf(montar_relatorio(cnpjs, dados), out, marca=marca)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Relatório de carteira de FIDCs para uma gestora (PDF).")
    ap.add_argument("--cnpjs", required=True, help="CSV com os CNPJs das classes de FIDC da carteira")
    ap.add_argument("--competencia", required=True, help="AAAA-MM, ex.: 2026-10")
    ap.add_argument("--out", required=True, help="caminho do PDF de saída")
    ap.add_argument("--marca", default="Observatório de crédito em FIDCs")
    a = ap.parse_args(argv)
    try:
        p = gerar_relatorio(a.cnpjs, a.competencia, a.out, a.marca)
    except (CarteiraError, ValueError, FileNotFoundError) as exc:
        print(f"erro: {exc}", file=sys.stderr)
        return 2
    print(f"ok: {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

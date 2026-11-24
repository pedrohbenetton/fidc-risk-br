"""Gera o boletim agregado (2 páginas A4) a partir de indicadores.json.

Uso: python -m relatorio.gerar_pdf [marca] [--indicadores indicadores.json] [--out-dir .] [--baixado-em DD/MM/AAAA]
     [--bloco relatorio/blocos/contagio_cedente.json]

A manchete e os verbos dos destaques ("subiu", "caiu", "estável") saem dos próprios números, então o
gerador serve para qualquer competência. Para baixar os dados e gerar tudo de uma vez, use
``python -m relatorio.boletim AAAA-MM``.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

# Paleta do observatório (web/styles.css): navy para o dado principal, cinza para o secundário,
# vermelho só para estresse (vencido 90+ acima de 10%).
NAVY, CINZA, VERMELHO = "#1f3a5f", "#a9acb2", "#8e2323"
INK, INK2, GRID, SURF = "#1b2433", "#4a5160", "#e4e2db", "#fbfaf7"

MESES = ["janeiro", "fevereiro", "março", "abril", "maio", "junho", "julho", "agosto", "setembro", "outubro", "novembro", "dezembro"]
ESTAVEL_PP = 0.05  # variação mensal abaixo disso (em p.p.) é "estável"
SEGMENTO_DESTAQUE = 1.5  # segmento grande com vencido 90+ >= 1,5x a média "concentra o problema"
SEGMENTO_MIN_BI = 3.0
DESCRICAO_SEGMENTO = {
    "Financeiro": "crédito pessoal, consignado, corporativo, veículos",
    "Comercial/varejo": "duplicatas e recebíveis de varejo",
    "Cartão de crédito": "recebíveis de cartão",
    "Setor público": "precatórios e créditos tributários",
}
BINS_DISTRIBUICAO = [0, 0.001, 2, 5, 10, 20, 50, 100.01]
LABELS_DISTRIBUICAO = ["0", "0–2", "2–5", "5–10", "10–20", "20–50", ">50"]


def br(v: float, d: int = 1) -> str:
    return f"{v:,.{d}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def pp(a: float, b: float) -> str:
    x = a - b
    return ("+" if x >= 0 else "−") + br(abs(x), 2) + " p.p."


def pct(a: float, b: float) -> str:
    x = (a / b - 1) * 100
    return ("+" if x >= 0 else "−") + br(abs(x), 1) + "%"


def mes_curto(comp: str) -> str:
    """'2026-10' -> 'out/26'."""
    ano, mes = int(comp[:4]), int(comp[5:7])
    return f"{MESES[mes - 1][:3]}/{ano % 100:02d}"


def mes_longo(comp: str) -> str:
    """'2026-10' -> 'outubro/2026'."""
    ano, mes = int(comp[:4]), int(comp[5:7])
    return f"{MESES[mes - 1]}/{ano}"


def mes_seguinte(comp: str) -> str:
    """'2026-10' -> 'novembro'."""
    return MESES[int(comp[5:7]) % 12]


def mm_aaaa(comp: str) -> str:
    """'2026-10' -> '10/2026'."""
    return f"{comp[5:7]}/{comp[:4]}"


# ------------------------------------------------------------------ texto gerado pelos dados

def _direcao(delta: float, limiar: float = ESTAVEL_PP) -> str:
    if delta >= limiar:
        return "subiu"
    if delta <= -limiar:
        return "caiu"
    return "ficou estável"


def segmento_destaque(cur: dict[str, Any]) -> tuple[dict[str, Any], float] | None:
    """Segmento grande com o maior vencido 90+ relativo à média da indústria."""
    grandes = [s for s in cur["segmentos"] if s["carteira_bi"] >= SEGMENTO_MIN_BI]
    if not grandes or cur["inad90_pct"] <= 0:
        return None
    s = max(grandes, key=lambda x: x["inad90_pct"])
    return s, s["inad90_pct"] / cur["inad90_pct"]


def manchete(R: dict[str, Any]) -> str:
    comp_cur, comp_prev = list(R)[:2]
    cur, prev = R[comp_cur], R[comp_prev]
    mes = MESES[int(comp_cur[5:7]) - 1].capitalize()
    d30 = _direcao(cur["inad30_pct"] - prev["inad30_pct"])
    d90 = _direcao(cur["inad90_pct"] - prev["inad90_pct"])
    dest = segmento_destaque(cur)
    if dest and dest[1] >= SEGMENTO_DESTAQUE:
        return f"{mes} de {comp_cur[:4]}: o atraso curto {d30}, e o segmento {dest[0]['seg'].lower()} concentra o problema"
    if d30 == d90:
        plural = {"subiu": "subiram", "caiu": "caíram", "ficou estável": "ficaram estáveis"}[d30]
        return f"{mes} de {comp_cur[:4]}: o atraso curto e o vencido acima de 90 dias {plural}"
    return f"{mes} de {comp_cur[:4]}: o atraso curto {d30}, e o vencido acima de 90 dias {d90}"


def destaques(R: dict[str, Any]) -> str:
    comp_cur, comp_prev, comp_yoy = list(R)[:3]
    cur, prev, yoy = R[comp_cur], R[comp_prev], R[comp_yoy]
    verbo_pl = "chegou a" if cur["pl_bi"] >= yoy["pl_bi"] else "recuou para"
    d90 = cur["inad90_pct"] - prev["inad90_pct"]
    est90 = {"subiu": "em alta no mês", "caiu": "em queda no mês", "ficou estável": "estável no mês"}[_direcao(d90)]
    d30 = cur["inad30_pct"] - prev["inad30_pct"]
    var30 = {
        "subiu": f"subiu {pp(cur['inad30_pct'], prev['inad30_pct'])}",
        "caiu": f"caiu {br(abs(d30), 2)} p.p.",
        "ficou estável": f"ficou estável ({pp(cur['inad30_pct'], prev['inad30_pct'])})",
    }[_direcao(d30)]
    partes = [
        f"• O <b>patrimônio da indústria {verbo_pl} R$ {br(cur['pl_bi'])} bi</b> ({pct(cur['pl_bi'], yoy['pl_bi'])} em 12 meses), "
        f"com {br(cur['n_classes'], 0)} classes reportando.",
        f"• O <b>vencido acima de 90 dias ficou em {br(cur['inad90_pct'], 2)}%</b> da carteira, {est90} "
        f"({pp(cur['inad90_pct'], prev['inad90_pct'])}) e {pp(cur['inad90_pct'], yoy['inad90_pct'])} em 12 meses. "
        f"O <b>atraso acima de 30 dias {var30}</b> no mês: é a faixa que costuma "
        "anteceder a de 90 dias.",
    ]
    verbo_pdd = "subiu para" if cur["pdd_pct"] >= yoy["pdd_pct"] else "caiu para"
    rel_pdd = "acima do" if cur["pdd_pct"] >= cur["inad90_pct"] else "abaixo do"
    partes.append(
        f"• A <b>provisão {verbo_pdd} {br(cur['pdd_pct'], 2)}%</b> da carteira ({pp(cur['pdd_pct'], yoy['pdd_pct'])} em 12 meses), "
        f"{rel_pdd} próprio vencido de 90+ dias."
    )
    dest = segmento_destaque(cur)
    if dest:
        s, ratio = dest
        desc = DESCRICAO_SEGMENTO.get(s["seg"])
        nome = f"segmento {s['seg'].lower()}" + (f"</b> ({desc})<b>" if desc else "")
        if ratio >= SEGMENTO_DESTAQUE:
            partes.append(
                f"• O <b>{nome}</b>, com R$ {br(s['carteira_bi'], 0)} bi, tem {br(s['inad90_pct'], 1)}% vencido há mais de "
                f"90 dias, cerca de {br(ratio, 1)} vezes a média."
            )
        else:
            partes.append(
                f"• Nenhum segmento grande se destaca: o maior vencido 90+ é o do <b>{nome}</b>, {br(s['inad90_pct'], 1)}% "
                f"({br(ratio, 1)} vez a média)."
            )
    verbo_rec = "caíram para" if cur["recompra_mi"] < yoy["recompra_mi"] else "subiram para"
    frase_rec = (
        f"• As <b>recompras pelo cedente</b> {verbo_rec} R$ {br(cur['recompra_mi'], 0)} mi ({br(cur['recompra_pct'], 2)}% da carteira), "
        f"contra R$ {br(yoy['recompra_mi'], 0)} mi há um ano."
    )
    atraso_subiu = cur["inad90_pct"] > yoy["inad90_pct"] or cur["inad30_pct"] > yoy["inad30_pct"]
    if cur["recompra_mi"] < yoy["recompra_mi"] and atraso_subiu:
        frase_rec += " Menos recompra com mais atraso pode indicar cedentes com menos fôlego para honrar coobrigação."
    partes.append(frase_rec)
    return "<br/>".join(partes)


def acompanhar(R: dict[str, Any]) -> str:
    comp_cur, _, comp_yoy = list(R)[:3]
    cur, yoy = R[comp_cur], R[comp_yoy]
    dest = segmento_destaque(cur)
    onde = f", principalmente no segmento {dest[0]['seg'].lower()}" if dest and dest[1] >= SEGMENTO_DESTAQUE else ""
    itens = [f"• Se o atraso de 31 a 90 dias migra para a faixa de 90+{onde}."]
    atraso_subiu = cur["inad90_pct"] > yoy["inad90_pct"] or cur["inad30_pct"] > yoy["inad30_pct"]
    if cur["recompra_mi"] < yoy["recompra_mi"]:
        itens.append("• Se as recompras seguem caindo enquanto o atraso sobe." if atraso_subiu
                     else "• Se a queda das recompras começa a aparecer no atraso.")
    else:
        itens.append("• Se a alta das recompras está segurando o atraso aparente das carteiras.")
    itens.append("• Quantas das classes expostas por cedente comum passam a mostrar atraso próprio.")
    return "<br/>".join(itens)


# ------------------------------------------------------------------ gráficos

def cores_distribuicao(labels: list[str]) -> list[str]:
    """Navy até 10% vencido 90+; vermelho nas faixas de estresse (acima de 10%)."""
    return [NAVY if i < 4 else VERMELHO for i in range(len(labels))]


def style_ax(ax: Any) -> None:
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8, length=0)
    ax.set_facecolor(SURF)


def _graficos(cur: dict[str, Any], out_dir: Path, bloco: dict[str, Any] | None) -> dict[str, Path]:
    paths = {k: out_dir / f"g_{k}.png" for k in ("segmentos", "distribuicao", "fluxos", "contagio")}

    # Gráfico 1: vencido 90+ por segmento (barras navy; média da indústria como referência em cinza)
    segs = [s for s in cur["segmentos"] if s["carteira_bi"] >= SEGMENTO_MIN_BI]
    segs = (segs or sorted(cur["segmentos"], key=lambda x: x["carteira_bi"])[-8:] or [{"seg": "—", "carteira_bi": 0.0, "inad90_pct": 0.0}])[::-1]
    fig, ax = plt.subplots(figsize=(6.6, 2.9), dpi=200)
    ax.barh([s["seg"] for s in segs], [s["inad90_pct"] for s in segs], color=NAVY, height=0.6)
    ax.axvline(cur["inad90_pct"], color=CINZA, lw=1.0, zorder=0)
    ax.text(cur["inad90_pct"] + 0.1, -0.85, f"indústria {br(cur['inad90_pct'], 1)}%", color=INK2, fontsize=7.5)
    xmax = max([s["inad90_pct"] for s in segs] + [cur["inad90_pct"], 1e-6]) * 1.08
    for i, s in enumerate(segs):
        ax.text(xmax * 1.03, i, f"{br(s['inad90_pct'], 1)}%", va="center", ha="right", fontsize=7.5, color=INK, fontweight="bold")
        ax.text(xmax * 1.05, i, f"R$ {br(s['carteira_bi'], 0)} bi", va="center", ha="left", fontsize=7.5, color=INK2)
    ax.text(xmax * 1.05, len(segs) - 0.35, "carteira", fontsize=7, color=INK2)
    ax.text(xmax * 1.03, len(segs) - 0.35, "vencido 90+", fontsize=7, color=INK2, ha="right")
    ax.set_xlim(0, xmax * 1.25)
    ax.set_xticks([])
    style_ax(ax)
    ax.tick_params(axis="y", labelcolor=INK)
    fig.tight_layout()
    fig.savefig(paths["segmentos"])
    plt.close(fig)

    # Gráfico 2: classes por faixa de vencido 90+ (vermelho só nas faixas de estresse)
    fig, ax = plt.subplots(figsize=(3.2, 2.4), dpi=200)
    d = cur["dist_inad90"]
    cnt = [sum(1 for v in d if lo <= v < hi) for lo, hi in zip(BINS_DISTRIBUICAO[:-1], BINS_DISTRIBUICAO[1:], strict=True)]
    ax.bar(LABELS_DISTRIBUICAO, cnt, color=cores_distribuicao(LABELS_DISTRIBUICAO), width=0.7)
    for i, c in enumerate(cnt):
        ax.text(i, c + max(cnt + [1]) * 0.02, str(c), ha="center", fontsize=7, color=INK)
    ax.set_yticks([])
    ax.set_xlabel("% da carteira vencida há mais de 90 dias", fontsize=7.5, color=INK2)
    style_ax(ax)
    fig.tight_layout()
    fig.savefig(paths["distribuicao"])
    plt.close(fig)

    # Gráfico 3: fluxos do mês (captações em navy; saídas em cinza)
    ops = cur["ops_bi"]
    fl = [("Captações", ops.get("Captações no Mês", 0)), ("Resgates", ops.get("Resgates no Mês", 0)), ("Amortizações", ops.get("Amortizações", 0))]
    fig, ax = plt.subplots(figsize=(3.2, 2.4), dpi=200)
    ax.bar([f[0] for f in fl], [f[1] for f in fl], color=[NAVY, CINZA, CINZA], width=0.6)
    topo = max([f[1] for f in fl] + [1e-9])
    for i, (_, v) in enumerate(fl):
        ax.text(i, v + topo * 0.03, f"R$ {br(v, 1)} bi", ha="center", fontsize=7.5, color=INK)
    ax.set_yticks([])
    style_ax(ax)
    fig.tight_layout()
    fig.savefig(paths["fluxos"])
    plt.close(fig)

    # Gráfico 4 (opcional): estudo de contágio por cedente, expostas (navy) contra controles (cinza)
    if bloco:
        fig, ax = plt.subplots(figsize=(2.6, 1.9), dpi=200)
        taxas = [t * 100 for t in bloco["taxas_6m"]]
        ax.bar([0, 1], taxas, color=[NAVY, CINZA], width=0.62)
        for i, t in enumerate(taxas):
            ax.text(i, t + max(taxas) * 0.03, f"{br(t, 1)}%", ha="center", fontsize=7.5, color=INK)
        ax.set_xticks([0, 1], [r.replace(" ", "\n", 1) for r in bloco["rotulos"]], fontsize=6.6)
        ax.set_ylim(0, max(taxas) * 1.25)
        ax.set_yticks([])
        ax.set_title("Entraram em estresse em 6 meses", fontsize=7.2, color=INK, loc="left")
        style_ax(ax)
        ax.tick_params(axis="x", labelsize=6.6)
        fig.tight_layout()
        fig.savefig(paths["contagio"])
        plt.close(fig)
    return paths


# ------------------------------------------------------------------ PDF

def gerar(
    indicadores: str | Path | dict[str, Any],
    marca: str = "[Sua marca]",
    out_dir: str | Path = ".",
    baixado_em: str | None = None,
    bloco: dict[str, Any] | None = None,
    avisos: list[str] | None = None,
) -> Path:
    """Gera boletim-fidc-AAAA-MM.pdf (e os gráficos g_*.png) em ``out_dir`` e devolve o caminho do PDF."""
    R: dict[str, Any] = (
        indicadores if isinstance(indicadores, dict) else json.loads(Path(indicadores).read_text(encoding="utf-8"))
    )
    comp_cur, comp_prev, comp_yoy = list(R)[:3]
    cur, prev, yoy = R[comp_cur], R[comp_prev], R[comp_yoy]
    baixado_em = baixado_em or dt.date.today().strftime("%d/%m/%Y")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    with plt.rc_context({"font.family": "DejaVu Sans", "figure.facecolor": SURF}):
        g = _graficos(cur, out, bloco)

    H1 = ParagraphStyle("h1", fontName="Helvetica-Bold", fontSize=17, leading=21, textColor=INK)
    SUB = ParagraphStyle("sub", fontName="Helvetica", fontSize=9, leading=12, textColor=INK2)
    H2 = ParagraphStyle("h2", fontName="Helvetica-Bold", fontSize=11, leading=14, textColor=INK, spaceBefore=8, spaceAfter=3)
    P = ParagraphStyle("p", fontName="Helvetica", fontSize=8.6, leading=11.6, textColor=INK)
    SM = ParagraphStyle("sm", fontName="Helvetica", fontSize=7, leading=9, textColor=INK2)

    ops = cur["ops_bi"]
    net = ops.get("Captações no Mês", 0) - ops.get("Resgates no Mês", 0) - ops.get("Amortizações", 0)
    rows = [
        ["Indicador", mes_curto(comp_cur), mes_curto(comp_prev), mes_curto(comp_yoy), "No mês", "Em 12 meses"],
        ["Patrimônio líquido (R$ bi)", br(cur["pl_bi"]), br(prev["pl_bi"]), br(yoy["pl_bi"]), pct(cur["pl_bi"], prev["pl_bi"]), pct(cur["pl_bi"], yoy["pl_bi"])],
        ["Classes com informe", br(cur["n_classes"], 0), br(prev["n_classes"], 0), br(yoy["n_classes"], 0), pct(cur["n_classes"], prev["n_classes"]), pct(cur["n_classes"], yoy["n_classes"])],
        ["Carteira de direitos creditórios (R$ bi)", br(cur["carteira_dc_bi"]), br(prev["carteira_dc_bi"]), br(yoy["carteira_dc_bi"]), pct(cur["carteira_dc_bi"], prev["carteira_dc_bi"]), pct(cur["carteira_dc_bi"], yoy["carteira_dc_bi"])],
        ["Vencido > 30 dias (% carteira)", br(cur["inad30_pct"], 2), br(prev["inad30_pct"], 2), br(yoy["inad30_pct"], 2), pp(cur["inad30_pct"], prev["inad30_pct"]), pp(cur["inad30_pct"], yoy["inad30_pct"])],
        ["Vencido > 90 dias (% carteira)", br(cur["inad90_pct"], 2), br(prev["inad90_pct"], 2), br(yoy["inad90_pct"], 2), pp(cur["inad90_pct"], prev["inad90_pct"]), pp(cur["inad90_pct"], yoy["inad90_pct"])],
        ["Vencido > 180 dias (% carteira)", br(cur["inad180_pct"], 2), br(prev["inad180_pct"], 2), br(yoy["inad180_pct"], 2), pp(cur["inad180_pct"], prev["inad180_pct"]), pp(cur["inad180_pct"], yoy["inad180_pct"])],
        ["Provisão (PDD) / carteira (%)", br(cur["pdd_pct"], 2), br(prev["pdd_pct"], 2), br(yoy["pdd_pct"], 2), pp(cur["pdd_pct"], prev["pdd_pct"]), pp(cur["pdd_pct"], yoy["pdd_pct"])],
        ["Classes com > 10% vencido há 90+ dias", f"{cur['classes_estresse']} ({br(cur['classes_estresse_pct'], 1)}%)", f"{prev['classes_estresse']} ({br(prev['classes_estresse_pct'], 1)}%)", f"{yoy['classes_estresse']} ({br(yoy['classes_estresse_pct'], 1)}%)", pp(cur["classes_estresse_pct"], prev["classes_estresse_pct"]), pp(cur["classes_estresse_pct"], yoy["classes_estresse_pct"])],
        ["Recompras pelo cedente (R$ mi)", br(cur["recompra_mi"], 0), br(prev["recompra_mi"], 0), br(yoy["recompra_mi"], 0), pct(cur["recompra_mi"], prev["recompra_mi"]), pct(cur["recompra_mi"], yoy["recompra_mi"])],
    ]
    tbl = Table(rows, colWidths=[64 * mm, 20 * mm, 20 * mm, 20 * mm, 22 * mm, 24 * mm])
    tbl.setStyle(TableStyle([
        ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 7.8), ("FONT", (0, 1), (-1, -1), "Helvetica", 7.8),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor(INK2)), ("TEXTCOLOR", (0, 1), (-1, -1), colors.HexColor(INK)),
        ("TEXTCOLOR", (1, 8), (3, 8), colors.HexColor(VERMELHO)),  # classes em estresse
        ("ALIGN", (1, 0), (-1, -1), "RIGHT"), ("LINEBELOW", (0, 0), (-1, 0), 0.6, colors.HexColor(INK2)),
        ("LINEBELOW", (0, 1), (-1, -1), 0.3, colors.HexColor(GRID)), ("BOTTOMPADDING", (0, 0), (-1, -1), 3), ("TOPPADDING", (0, 0), (-1, -1), 3),
    ]))

    mes = MESES[int(comp_cur[5:7]) - 1]
    story: list[Any] = [
        Paragraph(f"{marca} · Boletim de crédito em FIDCs", SUB),
        Paragraph(manchete(R), H1),
        Spacer(1, 3),
        Paragraph("Indústria inteira, a partir dos informes mensais entregues à CVM. Nenhum fundo é identificado.", SUB),
        Spacer(1, 6),
        Paragraph("Destaques", H2),
        Paragraph(destaques(R), P),
        Paragraph("Painel do mês", H2),
        tbl,
        Paragraph("Vencido há mais de 90 dias, por segmento dominante da carteira", H2),
        Image(str(g["segmentos"]), width=170 * mm, height=170 * mm * 2.9 / 6.6),
        Paragraph("Cada classe foi atribuída ao segmento com maior valor na sua carteira. Rótulo: % vencido 90+ dias · tamanho da carteira. "
                  "Segmentos com menos de R$ 3 bi omitidos. Linha cinza: média da indústria.", SM),
        PageBreak(),
        Paragraph("Concentração do risco", H2),
        Paragraph(
            f"A média esconde a cauda: a maioria das classes não tem atraso relevante, mas <b>{cur['classes_estresse']} classes "
            f"({br(cur['classes_estresse_pct'], 1)}%)</b> têm mais de 10% da carteira vencida há mais de 90 dias. "
            "É nesse grupo que um comitê precisa olhar primeiro.", P),
        Spacer(1, 4),
        Table([[Image(str(g["distribuicao"]), width=84 * mm, height=63 * mm), Image(str(g["fluxos"]), width=84 * mm, height=63 * mm)],
               [Paragraph("Número de classes por faixa de vencido 90+ dias (% da carteira). Em vermelho, as faixas acima de 10% (estresse).", SM),
                Paragraph(f"Fluxos de {mes}. Captação líquida de resgates e amortizações: R$ {br(net, 1)} bi.", SM)]],
              colWidths=[88 * mm, 88 * mm]),
        Paragraph("Contágio por cedente", H2),
        Paragraph(
            f"Os informes listam os maiores cedentes de cada classe. Em {mes}, <b>{br(cur['cedentes_pj'], 0)} empresas cedentes</b> apareceram nessas listas; "
            f"<b>{cur['cedentes_compartilhados']}</b> cedem para duas ou mais classes ao mesmo tempo. Dessas, <b>{cur['cedentes_ligados_estresse']}</b> "
            "estão presentes em pelo menos uma classe sob estresse (mais de 10% vencido 90+ dias). "
            f"<b>{cur['classes_expostas']} classes ainda sem estresse</b>, somando <b>R$ {br(cur['pl_exposto_bi'], 1)} bi</b> de patrimônio, "
            "dividem ao menos um desses cedentes. Se o problema estiver no cedente, e não na carteira de um fundo só, ele pode aparecer nos outros: "
            "esse é o canal de transmissão que a análise fundo a fundo não enxerga.", P),
    ]
    if not bloco:
        story += _tabela_segmentos_12m(R, H2, SM)
    if bloco:
        story += [
            Paragraph(f"Estudo: {bloco['titulo']}", H2),
            Table(
                [[Paragraph(bloco["texto"], P), Image(str(g["contagio"]), width=58 * mm, height=58 * mm * 1.9 / 2.6)]],
                colWidths=[118 * mm, 60 * mm],
                style=TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (0, 0), 0)]),
            ),
        ]
    story += [
        Paragraph(f"O que acompanhar em {mes_seguinte(comp_cur)}", H2),
        Paragraph(acompanhar(R), P),
        Paragraph("Metodologia e limites", H2),
        Paragraph(
            f"Fonte: Informe Mensal de FIDC (CVM, dados abertos), competências {mm_aaaa(comp_cur)}, {mm_aaaa(comp_prev)} e {mm_aaaa(comp_yoy)}, baixadas em {baixado_em}. "
            "Vencido por faixa: soma das tabelas V e VI (com e sem aquisição substancial de riscos) sobre a carteira a vencer + vencida. "
            "Provisão: redução ao valor recuperável (tabela I). Recompras: tabela VII. Cedentes: até 9 maiores cedentes por classe (tabela I), "
            "só pessoas jurídicas. O patrimônio inclui fundos que investem em cotas de outros FIDCs e por isso tem dupla contagem; "
            "a carteira de direitos creditórios não. Informes podem ser reenviados e corrigidos pela CVM após a data de extração. "
            "O percentual de subordinação foi omitido: a série mudou de padrão após a RCVM 175 e não é comparável no tempo.", SM),
        *[Paragraph(f"<b>Atenção.</b> {a}", SM) for a in (avisos or [])],
        Spacer(1, 6),
        Paragraph(
            "<b>Aviso.</b> Material informativo com estatísticas agregadas de dados públicos. Não é recomendação de investimento, "
            "não analisa nem cita fundos ou emissores específicos e não substitui a análise de cada investidor.", SM),
    ]

    rodape = f"{marca} · Boletim FIDC · {mes_longo(comp_cur)}"

    def footer(c: Any, doc: Any) -> None:
        c.setFont("Helvetica", 7)
        c.setFillColor(colors.HexColor(INK2))
        c.drawString(18 * mm, 10 * mm, rodape)
        c.drawRightString(A4[0] - 18 * mm, 10 * mm, f"{doc.page}/2")

    pdf_path = out / f"boletim-fidc-{comp_cur}.pdf"
    doc = SimpleDocTemplate(str(pdf_path), pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=14 * mm, bottomMargin=16 * mm,
                            title=f"Boletim de crédito em FIDCs · {mes_longo(comp_cur)}")
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return pdf_path


def _tabela_segmentos_12m(R: dict[str, Any], H2: ParagraphStyle, SM: ParagraphStyle) -> list[Any]:
    """Vencido 90+ por segmento, há 12 meses e agora (ocupa a página 2 quando não há bloco de estudo)."""
    comp_cur, _, comp_yoy = list(R)[:3]
    antes = {s["seg"]: s for s in R[comp_yoy]["segmentos"]}
    segs = [s for s in R[comp_cur]["segmentos"] if s["carteira_bi"] >= SEGMENTO_MIN_BI] or R[comp_cur]["segmentos"][:8]
    linhas = [["Segmento", "Carteira (R$ bi)", f"Vencido 90+ {mes_curto(comp_yoy)}", f"Vencido 90+ {mes_curto(comp_cur)}", "Em 12 meses"]]
    for s in segs[:9]:
        a = antes.get(s["seg"])
        linhas.append([s["seg"], br(s["carteira_bi"], 0), "—" if a is None else f"{br(a['inad90_pct'], 1)}%",
                       f"{br(s['inad90_pct'], 1)}%", "—" if a is None else pp(s["inad90_pct"], a["inad90_pct"])])
    t = Table(linhas, colWidths=[52 * mm, 30 * mm, 32 * mm, 32 * mm, 28 * mm])
    t.setStyle(TableStyle([
        ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 7.6), ("FONT", (0, 1), (-1, -1), "Helvetica", 7.6),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor(INK2)), ("TEXTCOLOR", (0, 1), (-1, -1), colors.HexColor(INK)),
        ("ALIGN", (1, 0), (-1, -1), "RIGHT"), ("LINEBELOW", (0, 0), (-1, 0), 0.6, colors.HexColor(INK2)),
        ("LINEBELOW", (0, 1), (-1, -1), 0.3, colors.HexColor(GRID)),
        ("TOPPADDING", (0, 0), (-1, -1), 2.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
    ]))
    return [
        Paragraph("Vencido 90+ por segmento em 12 meses", H2),
        t,
        Paragraph("Segmentos com pelo menos R$ 3 bi de carteira; vencido 90+ em % da carteira de cada segmento.", SM),
    ]


def carregar_bloco(path: str | Path | None) -> dict[str, Any] | None:
    if path is None or not Path(path).exists():
        return None
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _cli(argv: list[str] | None = None) -> Path:
    ap = argparse.ArgumentParser(description="Gera o boletim agregado em PDF a partir de indicadores.json.")
    ap.add_argument("marca", nargs="?", default="[Sua marca]")
    ap.add_argument("--indicadores", default="indicadores.json")
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--baixado-em", default=None, help="data de extração dos dados da CVM (DD/MM/AAAA)")
    ap.add_argument("--bloco", default=None, help="JSON do bloco do estudo de contágio (opcional)")
    a = ap.parse_args(argv)
    p = gerar(a.indicadores, marca=a.marca, out_dir=a.out_dir, baixado_em=a.baixado_em, bloco=carregar_bloco(a.bloco))
    print(f"ok: {p}")
    return p


if __name__ == "__main__":
    _cli()

"""Página e PDF de Metodologia: "por que eu confiaria nisso?" para um analista de crédito.

Tudo o que é número vem de reports/metrics/methodology_summary.json (monitoring.methodology); os
limiares da definição de estresse vêm das constantes da camada Gold. Nada identifica fundos.
"""

from __future__ import annotations

import html
import json
import math
from pathlib import Path
from typing import Any

from fidc_risk.config import FIGURES_DIR, METRICS_DIR, WEB_DIR

MESES_ABREV = ["jan", "fev", "mar", "abr", "mai", "jun", "jul", "ago", "set", "out", "nov", "dez"]

# Paleta do observatório (web/styles.css)
INK, INK2, INK3, RULE, SHEET, NAVY, GRID = "#1b2433", "#4a5160", "#7a7f88", "#d9d8d2", "#fbfaf7", "#1f3a5f", "#e4e2db"


# ------------------------------------------------------------------ formatação

def _pct(x: float | None, d: int = 1) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    return f"{x * 100:.{d}f}%".replace(".", ",")


def _num(x: float | int | None, d: int = 0) -> str:
    if x is None:
        return "—"
    return f"{x:,.{d}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def _vezes(x: float | None) -> str:
    return "—" if x is None else f"{x:.1f}×".replace(".", ",")


def _mes_ano(ym: str | None) -> str:
    if not ym:
        return "—"
    return f"{MESES_ABREV[int(ym[5:7]) - 1]}/{ym[:4]}"


def _e(s: Any) -> str:
    return html.escape(str(s), quote=True)


# ------------------------------------------------------------------ leituras em texto

def _leitura_calibracao(safras: list[dict[str, Any]]) -> str:
    """Frase com a direção do erro de nível em cada safra (subestimou, superestimou, acertou)."""
    partes = []
    for s in safras:
        diff = s["taxa_eventos"] - s["score_medio"]
        if diff > 0.01:
            verbo = "subestimou"
        elif diff < -0.01:
            verbo = "superestimou"
        else:
            verbo = "acertou o nível de"
        partes.append(
            f"em {s['rotulo']} {verbo} o risco (score médio {_pct(s['score_medio'])}, aconteceu em {_pct(s['taxa_eventos'])})"
        )
    texto = "; ".join(partes)
    return texto[0].upper() + texto[1:] + "."


def _resumo_bullets(s: dict[str, Any]) -> list[str]:
    av, saf, err = s["avaliacao"], s["por_safra"], s["erros"]["total"]
    ordem = [x["acerto_ordenacao"] for x in saf if x["acerto_ordenacao"] is not None]
    conc = [x["concentracao_top5"] for x in saf if x["concentracao_top5"] is not None]
    pior = max(saf, key=lambda x: abs(x["taxa_eventos"] - x["score_medio"])) if saf else None
    bullets = [
        f"<b>Testado fora da amostra.</b> {_num(av['n_obs'])} previsões (fundo × mês) de {_mes_ano(av['inicio'])} a "
        f"{_mes_ano(av['fim'])}; cada ano foi previsto por um modelo que nunca viu aquele ano. "
        f"{_num(av['eventos'])} delas terminaram em estresse.",
    ]
    if ordem:
        bullets.append(
            f"<b>Ordena bem.</b> Em cada ano, entre {_pct(min(ordem), 0)} e {_pct(max(ordem), 0)} das vezes o fundo que "
            "entrou em estresse tinha score maior que um fundo que não entrou (50% seria sorteio)."
        )
    if conc:
        bullets.append(
            f"<b>Concentra o risco no topo.</b> Os 5% de fundos com maior score tiveram de {_vezes(min(conc))} a "
            f"{_vezes(max(conc))} a frequência média de estresse do ano."
        )
    if pior:
        bullets.append(
            f"<b>O nível é aproximado.</b> A maior distância entre o score médio e o que aconteceu foi em {pior['rotulo']}: "
            f"score médio de {_pct(pior['score_medio'])} contra {_pct(pior['taxa_eventos'])} de eventos."
        )
    bullets.append(
        f"<b>Não vê tudo.</b> {_pct(1 - err['eventos_no_top20'], 0)} dos eventos aconteceram em fundos fora dos 20% de maior "
        "score do ano, sobretudo quedas da cota sênior sem atraso prévio nos informes."
    )
    nv = s.get("nivel_modelo_servido")
    if nv and nv.get("score_medio") is not None and nv["score_medio"] < nv["taxa_eventos"] - 0.01:
        bullets.append(
            f"<b>O score do observatório está abaixo do risco real.</b> Nos meses de {_mes_ano(nv['meses'][0])} a "
            f"{_mes_ano(nv['meses'][1])}, que o modelo em produção não viu e que já têm 6 meses de resultado, o score médio foi "
            f"{_pct(nv['score_medio'])} e a frequência de estresse, {_pct(nv['taxa_eventos'])}. A ordem entre os fundos não "
            "muda; use o número para comparar fundos, não como probabilidade exata."
        )
    return bullets


def _definicao(s: dict[str, Any]) -> list[str]:
    d = s["definicoes"]
    h = int(d["horizonte_meses"])
    return [
        f"O score é a <b>probabilidade estimada de o fundo entrar em estresse nos próximos {h} meses</b>.",
        f"Estresse, aqui, é qualquer um destes dois acontecimentos dentro dos {h} meses seguintes ao informe:",
        f"(a) a cota sênior perde {_num(abs(d['queda_senior_acumulada_pct']), 1)}% ou mais no acumulado, ou cai "
        f"{_num(abs(d['queda_senior_mensal_pct']), 1)}% ou mais em um único mês;",
        f"(b) o estoque ruim (o maior entre vencido há mais de 90 dias e provisão, em % da carteira) sobe pelo menos "
        f"{_num(d['salto_estoque_ruim_pp'], 0)} p.p., chega a {_num(d['estoque_ruim_minimo_pct'], 0)}% ou mais e deixa a "
        f"subordinação com folga de {_num(d['folga_subordinacao_pp'], 0)} p.p. ou menos.",
        f"Recebem score os fundos com patrimônio de pelo menos R$ {_num(d['elegivel_pl_minimo_mi'], 0)} mi e carteira de "
        f"pelo menos R$ {_num(d['elegivel_carteira_minima_mi'], 0)} mi. O teste abaixo só usou fundos que ainda não estavam "
        f"em estresse na data da previsão (estoque ruim abaixo de {_num(d['elegivel_estoque_ruim_maximo_pct'], 0)}% e sem "
        f"queda de mais de {_num(abs(d['estresse_agudo_queda_senior_mes_pct']), 0)}% da cota sênior no mês). Para quem já "
        "está em estresse, o número aparece no observatório mas não foi testado.",
        "<b>Faixa de incerteza.</b> Cada score vem com um mínimo e um máximo. Faixa larga quer dizer poucos fundos "
        "parecidos no histórico: leia o número com mais cautela.",
        "<b>Não confunda com \"classe em estresse\" do boletim mensal</b> (vencido 90+ acima de 10% da carteira): aquilo é "
        "uma fotografia do mês; o score é a chance de um evento nos próximos meses.",
    ]


def _limitacoes(s: dict[str, Any]) -> list[str]:
    fora = s["fora_por_janela_incompleta"]
    saf = s["por_safra"]
    pior = max((abs(x["taxa_eventos"] - x["score_medio"]) for x in saf), default=0.0)
    nv = s.get("nivel_modelo_servido")
    extra = []
    if nv and nv.get("score_medio") is not None and nv["score_medio"] < nv["taxa_eventos"] - 0.01:
        extra.append(
            f"<b>Nível do score em produção.</b> A versão do modelo que gera o score do observatório ajustou a "
            f"probabilidade com os meses de {_mes_ano(nv['calibracao'][0])} a {_mes_ano(nv['calibracao'][1])}, quando a "
            "maior parte dos eventos ainda não tinha tido tempo de acontecer. Resultado: nos meses que ela não viu, o "
            f"score médio ({_pct(nv['score_medio'])}) ficou abaixo da frequência real ({_pct(nv['taxa_eventos'])}). "
            "Refazer esse ajuste só com meses de resultado completo corrige o nível; está pendente."
        )
    return extra + [
        "<b>Dados autodeclarados.</b> Os informes são preenchidos pelos administradores, podem ser reenviados e corrigidos "
        "depois da extração e às vezes têm erros de preenchimento. Regras de consistência filtram os casos grosseiros, não todos.",
        "<b>Cedentes parciais.</b> O informe lista só os maiores cedentes de cada classe (até 9 por grupo). Exposição a "
        "cedentes menores não aparece.",
        f"<b>Meses recentes ainda sem resultado.</b> As {_num(fora['n_obs'])} previsões de {_mes_ano(fora['inicio'])} a "
        f"{_mes_ano(fora['fim'])} ficaram fora do teste porque ainda não se passaram {int(s['definicoes']['horizonte_meses'])} "
        "meses: parte dos eventos ainda pode acontecer.",
        "<b>Mudança de regra.</b> A Resolução CVM 175 passou a exigir informes por classe, e não por fundo, a partir de 2023–2024; "
        "algumas séries, como a subordinação, não são comparáveis antes e depois.",
        f"<b>Probabilidade não é certeza.</b> O nível do score errou em até {_num(pior * 100, 1)} p.p. num ano. Use o score "
        "para ordenar e priorizar a análise, não como número exato.",
        "<b>Segmentos pequenos.</b> Com poucos eventos, os números por segmento oscilam muito de um ano para outro.",
        "<b>Só o que está nos informes.</b> Notícias, rating, contratos, garantias e diligência não entram no modelo. "
        "Perdas súbitas sem atraso prévio (marcação, liquidez, fraude) tendem a escapar.",
        "<b>Não é recomendação de investimento</b> e não substitui a análise de crédito de cada investidor.",
    ]


# ------------------------------------------------------------------ gráfico de calibração (SVG)

def _axis_max(safras: list[dict[str, Any]]) -> float:
    vals = [max(b["previsto"], b["observado"]) for s in safras for b in s["calibracao"]] or [0.1]
    return min(1.0, max(0.1, math.ceil(max(vals) * 10) / 10))


def calibration_svg(safras: list[dict[str, Any]]) -> str:
    """Small multiples (um SVG por safra, em grade que quebra linha): score previsto × frequência observada."""
    if not safras:
        return ""
    vmax = _axis_max(safras)
    pw, ph, ml, mr, mb, mt = 150, 150, 34, 8, 22, 22
    width, height = ml + pw + mr, mt + ph + mb
    ticks = [t / 100 for t in range(0, int(vmax * 100) + 1, 20 if vmax > 0.6 else 10)]

    def sx(v: float) -> float:
        return ml + min(v, vmax) / vmax * pw

    def sy(v: float) -> float:
        return mt + ph - min(v, vmax) / vmax * ph

    paineis = []
    for s in safras:
        parts = [
            f'<svg class="metod-calib" viewBox="0 0 {width} {height}" role="img" '
            f'aria-label="{_e(s["rotulo"])}: score previsto contra frequência observada de estresse">',
            f'<text class="calib-title" x="{ml}" y="{mt - 8}">{_e(s["rotulo"])}</text>',
        ]
        for t in ticks:
            parts.append(f'<line class="calib-grid" x1="{ml}" x2="{ml + pw}" y1="{sy(t):.1f}" y2="{sy(t):.1f}"/>')
            parts.append(f'<text class="calib-tick" x="{ml - 5}" y="{sy(t) + 3:.1f}" text-anchor="end">{int(t * 100)}%</text>')
            parts.append(f'<text class="calib-tick" x="{sx(t):.1f}" y="{mt + ph + 13}" text-anchor="middle">{int(t * 100)}%</text>')
        parts.append(f'<line class="calib-diag" x1="{sx(0):.1f}" y1="{sy(0):.1f}" x2="{sx(vmax):.1f}" y2="{sy(vmax):.1f}"/>')
        pts = [(sx(b["previsto"]), sy(b["observado"]), b) for b in s["calibracao"]]
        if len(pts) > 1:
            d = " ".join(f"{'M' if j == 0 else 'L'}{x:.1f},{y:.1f}" for j, (x, y, _) in enumerate(pts))
            parts.append(f'<path class="calib-line" d="{d}"/>')
        for x, y, b in pts:
            tip = (
                f"{s['rotulo']}: score médio {_pct(b['previsto'])}, aconteceu em {_pct(b['observado'])} "
                f"({_num(b['n'])} previsões)"
            )
            parts.append(f'<circle class="calib-dot" cx="{x:.1f}" cy="{y:.1f}" r="4"><title>{_e(tip)}</title></circle>')
        parts.append("</svg>")
        paineis.append("".join(parts))
    legenda = '<p class="metod-calib-axis">Eixo horizontal: score previsto. Eixo vertical: frequência observada de estresse.</p>'
    return f'<div class="metod-calib-grid">{"".join(paineis)}</div>{legenda}'


# ------------------------------------------------------------------ HTML

def _tabela_safras(saf: list[dict[str, Any]]) -> str:
    linhas = "".join(
        f"<tr><td>{_e(s['rotulo'])}</td><td class=\"num\">{_num(s['n_obs'])}</td><td class=\"num\">{_num(s['eventos'])} "
        f"({_pct(s['taxa_eventos'])})</td><td class=\"num\">{_pct(s['score_medio'])}</td>"
        f"<td class=\"num\">{_pct(s['acerto_ordenacao'], 0)}</td><td class=\"num\">{_vezes(s['concentracao_top5'])}</td>"
        f"<td class=\"num\">{_pct(s['eventos_no_top5'], 0)}</td></tr>"
        for s in saf
    )
    return (
        '<div class="table-scroll metod-table"><table class="data-table"><thead><tr>'
        "<th>Ano da previsão</th><th class=\"num\">Previsões</th><th class=\"num\">Entraram em estresse</th>"
        "<th class=\"num\">Score médio</th><th class=\"num\">Acerto na ordenação</th>"
        "<th class=\"num\">Topo 5%: vezes a média</th><th class=\"num\">Eventos pegos no topo 5%</th>"
        f"</tr></thead><tbody>{linhas}</tbody></table></div>"
    )


def _tabela_segmentos(segs: list[dict[str, Any]]) -> str:
    linhas = "".join(
        f"<tr><td>{_e(g['segmento'])}</td><td class=\"num\">{_num(g['n_obs'])}</td><td class=\"num\">{_num(g['eventos'])}</td>"
        f"<td class=\"num\">{_pct(g['taxa_eventos'])}</td><td class=\"num\">{_pct(g['eventos_no_top20'], 0)}</td>"
        f"<td class=\"num\">{_pct(g['eventos_abaixo_da_mediana'], 0)}</td></tr>"
        for g in segs
    )
    return (
        '<div class="table-scroll metod-table"><table class="data-table"><thead><tr>'
        "<th>Segmento dominante</th><th class=\"num\">Previsões</th><th class=\"num\">Eventos</th>"
        "<th class=\"num\">Taxa de eventos</th><th class=\"num\">Eventos no topo 20%</th>"
        "<th class=\"num\">Eventos abaixo da mediana</th>"
        f"</tr></thead><tbody>{linhas}</tbody></table></div>"
    )


def _perfil_erros(s: dict[str, Any]) -> str:
    p = s.get("perfil_erros")
    if not p:
        return ""
    v, n = p["vistos"], p["nao_vistos"]
    return (
        "<p>O que separa os eventos vistos dos não vistos: entre os eventos <b>fora</b> dos 20% de maior score, "
        f"{_pct(n['so_queda_cota_senior'], 0)} foram só queda da cota sênior, contra {_pct(v['so_queda_cota_senior'], 0)} "
        f"entre os eventos vistos. No mês da previsão, o fundo típico não visto tinha {_pct(n['mediana_vencido90_em_t'])} de "
        f"vencido há mais de 90 dias e {_pct(n['mediana_atraso_1_60_em_t'])} de atraso de 1 a 60 dias (os vistos: "
        f"{_pct(v['mediana_vencido90_em_t'])} e {_pct(v['mediana_atraso_1_60_em_t'])}). Sem sinal de crédito nos informes, "
        "o modelo não tem o que ler.</p>"
    )


def render_methodology_html(s: dict[str, Any]) -> str:
    """Página estática web/metodologia.html (mesma identidade visual do observatório)."""
    av, fora, err = s["avaliacao"], s["fora_por_janela_incompleta"], s["erros"]["total"]
    saf = s["por_safra"]
    resumo = "".join(f"<li>{b}</li>" for b in _resumo_bullets(s))
    defin = _definicao(s)
    defin_html = (
        f"<p>{defin[0]}</p><p>{defin[1]}</p><ul class=\"metod-list\"><li>{defin[2]}</li><li>{defin[3]}</li></ul>"
        + "".join(f"<p>{x}</p>" for x in defin[4:])
    )
    limites = "".join(f"<li>{x}</li>" for x in _limitacoes(s))
    baixado = f", dados baixados da CVM em {_e(s['baixado_em'])}" if s.get("baixado_em") else ""
    return f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Metodologia · Observatório de crédito em FIDCs</title>
  <meta name="description" content="Por que confiar no score de estresse de FIDCs: dados, definição, validação fora da amostra e limitações.">
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@400;500;600&family=Source+Serif+4:opsz,wght@8..60,600&display=swap" rel="stylesheet">
  <link rel="stylesheet" href="styles.css?v=8">
</head>
<body class="metod-page">
  <main class="metod">
    <nav class="metod-nav"><a href="index.html">← Observatório</a><a href="metodologia.pdf">Versão em PDF</a></nav>
    <header class="metod-header">
      <span class="eyebrow">Metodologia · score de estresse em 6 meses</span>
      <h1>Por que confiar no score, e quando não confiar</h1>
      <p class="metod-lede">Informes até {_mes_ano(s['ultima_competencia'])}{baixado}. Previsões testadas de
      {_mes_ano(av['inicio'])} a {_mes_ano(av['fim'])}, contra o que de fato aconteceu com cada fundo.</p>
    </header>

    <section class="metod-card"><h2>Em resumo</h2><ul class="metod-list">{resumo}</ul></section>

    <section class="metod-card" id="dados">
      <h2>De onde vêm os dados</h2>
      <p>Do Informe Mensal de FIDC que cada fundo entrega à CVM (dados abertos): carteira por faixa de atraso
      (tabelas V e VI), provisão e maiores cedentes (tabela I), segmentos (tabela II), patrimônio (tabela IV),
      recompras e substituições (tabela VII) e cotas, rentabilidade e fluxos (tabela X). Completam a base o cadastro de
      fundos da CVM e séries do Banco Central (CDI, Selic, inadimplência do sistema financeiro).</p>
      <p><b>Defasagem.</b> O informe de um mês pode ser enviado até 15 dias depois do fim do mês, e a CVM atualiza o
      arquivo aberto à medida que os fundos enviam ou corrigem. O score de um mês usa só informação disponível até
      aquele mês. Nesta versão, a competência mais recente é {_mes_ano(s['ultima_competencia'])}.</p>
      <p>Fundos liquidados continuam na base: o histórico não é só de sobreviventes. Não há dado sintético.</p>
    </section>

    <section class="metod-card" id="score">
      <h2>O que o score significa</h2>
      {defin_html}
    </section>

    <section class="metod-card" id="teste">
      <h2>Como testamos</h2>
      <p>Para cada ano, o modelo foi treinado só com anos anteriores, deixando {int(s['definicoes']['horizonte_meses'])}
      meses de intervalo entre o fim do treino e o começo do teste, para que nenhum resultado já conhecido entrasse no
      treino. Depois, o que ele previu foi comparado com o que aconteceu. É a mesma receita do modelo que gera o score do
      observatório; a versão em produção foi ajustada depois, com meses ainda sem resultado completo (veja Limitações).</p>
      <p>Ficaram fora as {_num(fora['n_obs'])} previsões de {_mes_ano(fora['inicio'])} a {_mes_ano(fora['fim'])}: ainda
      não se passaram {int(s['definicoes']['horizonte_meses'])} meses. Até agora, {_pct(fora['taxa_eventos_parcial'])}
      delas tiveram evento; contar esses meses como resolvidos faria o modelo parecer melhor do que é.</p>
      {_tabela_safras(saf)}
      <p class="metod-note"><b>Acerto na ordenação:</b> sorteando um fundo que entrou em estresse e um que não entrou, a
      fração das vezes em que o primeiro tinha score maior (50% = sorteio, 100% = perfeito).
      <b>Topo 5%: vezes a média:</b> entre os 5% de maior score do ano, quantas vezes a frequência média de estresse
      do ano. <b>Eventos pegos no topo 5%:</b> fração de todos os eventos do ano que estavam nesse grupo.</p>
      <h3>Quando o score diz X%, acontece em X%?</h3>
      <div class="metod-chart">{calibration_svg(saf)}</div>
      <p class="metod-note">Cada ponto junta um décimo das previsões do ano, das de score mais baixo às de score mais
      alto. Sobre a diagonal, o score acertou o nível; acima dela, aconteceu mais do que o previsto; abaixo, menos.</p>
      <p>{_leitura_calibracao(saf)}</p>
    </section>

    <section class="metod-card" id="erros">
      <h2>Onde o modelo erra</h2>
      <ul class="metod-list">
        <li><b>Eventos fora do radar:</b> {_pct(1 - err['eventos_no_top20'], 0)} dos {_num(err['eventos'])} eventos
        aconteceram em fundos fora dos 20% de maior score do ano, e {_pct(err['eventos_abaixo_da_mediana'], 0)} em fundos
        com score abaixo da mediana.</li>
        <li><b>Alarmes falsos:</b> entre os 5% de maior score, {_pct(err['alarmes_falsos_top5'], 0)} não tiveram evento
        em {int(s['definicoes']['horizonte_meses'])} meses. Parte é o preço de olhar cedo; parte é erro.</li>
      </ul>
      {_perfil_erros(s)}
      {_tabela_segmentos(s['erros']['por_segmento'])}
      <p class="metod-note">Segmentos com menos de 30 eventos estão somados em "Outros". Números agregados: nenhum fundo
      é identificado.</p>
    </section>

    <section class="metod-card" id="limites"><h2>Limitações conhecidas</h2><ul class="metod-list">{limites}</ul></section>

    <section id="nomes-tecnicos" class="metod-glossary">
      <h2>Nomes técnicos, para quem quiser conferir</h2>
      <p>Acerto na ordenação = ROC-AUC. Topo 5%: vezes a média = lift no top 5%. Eventos pegos no topo 5% = recall no
      top 5%. Distância entre score e frequência = ECE em 10 faixas ({'; '.join(f"{x['rotulo']}: {_num(x['erro_calibracao'] * 100, 1)} p.p." for x in saf)}).
      Faixa de incerteza = intervalo de Venn-Abers indutivo. Modelo = LightGBM com restrições monotônicas, calibrado por
      Venn-Abers (coluna {_e(s['coluna_score'])}). Teste = validação walk-forward anual com embargo de
      {int(s['definicoes']['horizonte_meses'])} meses.</p>
    </section>

    <footer class="metod-footer">Material informativo com estatísticas agregadas de dados públicos. Não é recomendação
    de investimento. Números gerados de reports/metrics/methodology_summary.json.</footer>
  </main>
</body>
</html>
"""


# ------------------------------------------------------------------ figura e PDF

def calibration_figure(safras: list[dict[str, Any]], path: Path) -> Path:
    """PNG com os mesmos small multiples da página (para o PDF e o README)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    vmax = _axis_max(safras)
    n = max(len(safras), 1)
    with plt.rc_context({"font.family": "DejaVu Sans", "figure.facecolor": SHEET}):
        fig, axes = plt.subplots(1, n, figsize=(1.45 * n + 0.4, 1.9), dpi=220, sharey=True)
        axes = [axes] if n == 1 else list(axes)
        for ax, s in zip(axes, safras, strict=False):
            ax.set_facecolor(SHEET)
            ax.plot([0, vmax], [0, vmax], color=INK3, lw=0.6, zorder=1)
            xs = [min(b["previsto"], vmax) for b in s["calibracao"]]
            ys = [min(b["observado"], vmax) for b in s["calibracao"]]
            ax.plot(xs, ys, color=NAVY, lw=1.4, zorder=2)
            ax.scatter(xs, ys, s=14, color=NAVY, edgecolors=SHEET, linewidths=0.8, zorder=3)
            ax.set_xlim(0, vmax)
            ax.set_ylim(0, vmax)
            ax.set_title(s["rotulo"], fontsize=7.5, color=INK, loc="left")
            ax.tick_params(labelsize=6, colors=INK2, length=0)
            ax.xaxis.set_major_formatter(lambda v, _: f"{v * 100:.0f}%")
            ax.yaxis.set_major_formatter(lambda v, _: f"{v * 100:.0f}%")
            ax.grid(color=GRID, lw=0.5)
            for sp in ax.spines.values():
                sp.set_visible(False)
        fig.text(0.5, 0.01, "score previsto", ha="center", fontsize=6.5, color=INK2)
        fig.text(0.0, 0.5, "aconteceu", va="center", rotation=90, fontsize=6.5, color=INK2)
        fig.tight_layout(rect=(0.02, 0.04, 1, 1))
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path)
        plt.close(fig)
    return path


def render_methodology_pdf(s: dict[str, Any], path: str | Path, figure_path: str | Path | None = None) -> Path:
    """Versão de 1–2 páginas A4 da Metodologia, para enviar a um analista."""
    import re

    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig = Path(figure_path) if figure_path else out.with_name("calibracao_por_safra.png")
    calibration_figure(s["por_safra"], fig)

    H1 = ParagraphStyle("h1", fontName="Times-Bold", fontSize=15, leading=18, textColor=INK)
    SUB = ParagraphStyle("sub", fontName="Helvetica", fontSize=8, leading=10.5, textColor=INK2)
    H2 = ParagraphStyle("h2", fontName="Times-Bold", fontSize=10.5, leading=13, textColor=INK, spaceBefore=6, spaceAfter=2)
    P = ParagraphStyle("p", fontName="Helvetica", fontSize=7.6, leading=10, textColor=INK, spaceAfter=2)
    SM = ParagraphStyle("sm", fontName="Helvetica", fontSize=6.6, leading=8.4, textColor=INK2)

    def para(txt: str, st: ParagraphStyle = P) -> Paragraph:
        # reportlab aceita <b>; o resto do HTML da página não entra aqui.
        return Paragraph(re.sub(r"</?(?:ul|li|p)>", "", txt), st)

    av, fora, err = s["avaliacao"], s["fora_por_janela_incompleta"], s["erros"]["total"]
    saf = s["por_safra"]
    defin = _definicao(s)

    tab = [["Ano", "Previsões", "Em estresse", "Score médio", "Ordenação", "Topo 5% × média", "Pegos no topo 5%"]] + [
        [x["rotulo"], _num(x["n_obs"]), f"{_num(x['eventos'])} ({_pct(x['taxa_eventos'])})", _pct(x["score_medio"]),
         _pct(x["acerto_ordenacao"], 0), _vezes(x["concentracao_top5"]), _pct(x["eventos_no_top5"], 0)]
        for x in saf
    ]
    t = Table(tab, colWidths=[24 * mm, 20 * mm, 27 * mm, 20 * mm, 20 * mm, 27 * mm, 26 * mm])
    t.setStyle(TableStyle([
        ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 6.8), ("FONT", (0, 1), (-1, -1), "Helvetica", 6.8),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor(INK2)), ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
        ("LINEBELOW", (0, 0), (-1, 0), 0.5, colors.HexColor(INK2)), ("LINEBELOW", (0, 1), (-1, -1), 0.25, colors.HexColor(RULE)),
        ("TOPPADDING", (0, 0), (-1, -1), 1.6), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.6),
    ]))
    segs = s["erros"]["por_segmento"][:8]
    tseg = Table(
        [["Segmento", "Eventos", "Taxa", "No topo 20%", "Abaixo da mediana"]]
        + [[g["segmento"], _num(g["eventos"]), _pct(g["taxa_eventos"]), _pct(g["eventos_no_top20"], 0),
            _pct(g["eventos_abaixo_da_mediana"], 0)] for g in segs],
        colWidths=[40 * mm, 20 * mm, 20 * mm, 24 * mm, 28 * mm],
    )
    tseg.setStyle(TableStyle([
        ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 6.6), ("FONT", (0, 1), (-1, -1), "Helvetica", 6.6),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor(INK2)), ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
        ("LINEBELOW", (0, 0), (-1, 0), 0.5, colors.HexColor(INK2)), ("LINEBELOW", (0, 1), (-1, -1), 0.25, colors.HexColor(RULE)),
        ("TOPPADDING", (0, 0), (-1, -1), 1.3), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.3),
    ]))
    baixado = f", dados baixados em {s['baixado_em']}" if s.get("baixado_em") else ""
    story: list[Any] = [
        Paragraph("Metodologia · score de estresse em 6 meses · Observatório de crédito em FIDCs", SUB),
        Paragraph("Por que confiar no score, e quando não confiar", H1),
        Paragraph(f"Informes até {_mes_ano(s['ultima_competencia'])}{baixado}. Previsões testadas de "
                  f"{_mes_ano(av['inicio'])} a {_mes_ano(av['fim'])}.", SUB),
        Spacer(1, 4),
        Paragraph("Em resumo", H2),
        *[para("• " + b) for b in _resumo_bullets(s)],
        Paragraph("De onde vêm os dados", H2),
        para("Informe Mensal de FIDC da CVM (dados abertos): faixas de atraso (tabelas V e VI), provisão e maiores "
             "cedentes (I), segmentos (II), patrimônio (IV), recompras (VII), cotas e rentabilidade (X); cadastro da CVM e "
             "séries do Banco Central. O informe pode ser enviado até 15 dias após o fim do mês e pode ser corrigido "
             "depois; o score de um mês usa só informação disponível até aquele mês. Fundos liquidados continuam na base."),
        Paragraph("O que o score significa", H2),
        para(defin[0] + " " + defin[1]),
        para("• " + defin[2]),
        para("• " + defin[3]),
        *[para(x) for x in defin[4:]],
        Paragraph("Como testamos", H2),
        para(f"Cada ano foi previsto por um modelo treinado só com anos anteriores, com {int(s['definicoes']['horizonte_meses'])} "
             "meses de intervalo entre treino e teste; é a mesma receita do modelo do observatório (a versão em produção foi "
             "ajustada depois, ver Limitações). Ficaram fora as "
             f"{_num(fora['n_obs'])} previsões de {_mes_ano(fora['inicio'])} a {_mes_ano(fora['fim'])}, ainda sem "
             f"{int(s['definicoes']['horizonte_meses'])} meses de resultado."),
        t,
        Spacer(1, 3),
        Image(str(fig), width=175 * mm, height=175 * mm * 1.9 / (1.45 * max(len(saf), 1) + 0.4)),
        Paragraph("Cada ponto junta um décimo das previsões do ano. Sobre a diagonal, o score acertou o nível; acima, "
                  "aconteceu mais do que o previsto; abaixo, menos. " + _leitura_calibracao(saf), SM),
        Paragraph("Onde o modelo erra", H2),
        para(f"{_pct(1 - err['eventos_no_top20'], 0)} dos {_num(err['eventos'])} eventos aconteceram fora dos 20% de maior "
             f"score do ano; {_pct(err['eventos_abaixo_da_mediana'], 0)}, abaixo da mediana. Entre os 5% de maior score, "
             f"{_pct(err['alarmes_falsos_top5'], 0)} não tiveram evento em {int(s['definicoes']['horizonte_meses'])} meses."),
    ]
    perfil = _perfil_erros(s)
    if perfil:
        story.append(para(perfil))
    story += [tseg, Paragraph("Limitações conhecidas", H2), *[para("• " + x, SM) for x in _limitacoes(s)]]

    def footer(c: Any, doc: Any) -> None:
        c.setFont("Helvetica", 6.5)
        c.setFillColor(colors.HexColor(INK3))
        c.drawString(16 * mm, 9 * mm, "Material informativo com estatísticas agregadas de dados públicos. Não é recomendação de investimento.")
        c.drawRightString(A4[0] - 16 * mm, 9 * mm, f"{doc.page}")

    doc = SimpleDocTemplate(str(out), pagesize=A4, leftMargin=16 * mm, rightMargin=16 * mm, topMargin=12 * mm,
                            bottomMargin=14 * mm, title="Metodologia · score de estresse em FIDCs")
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return out


def export_methodology() -> dict[str, Any]:
    """Recalcula o resumo e regrava a página, o PDF e a figura versionados."""
    from fidc_risk.monitoring.methodology import compute_methodology_summary

    summary = compute_methodology_summary()
    (WEB_DIR / "metodologia.html").write_text(render_methodology_html(summary), encoding="utf-8")
    fig = FIGURES_DIR / "fig9_calibracao_por_safra.png"
    render_methodology_pdf(summary, WEB_DIR / "metodologia.pdf", figure_path=fig)
    print(f"[Metodologia] Página, PDF e figura gerados a partir de {METRICS_DIR / 'methodology_summary.json'}.")
    return summary


def load_summary() -> dict[str, Any]:
    return json.loads((METRICS_DIR / "methodology_summary.json").read_text(encoding="utf-8"))

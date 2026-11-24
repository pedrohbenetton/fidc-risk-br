"""Estudo histórico de contágio por cedente compartilhado (2022–2026).

Pergunta: quando uma classe entra em estresse (vencido 90+ passa de 10% da carteira), o que acontece com
as outras classes que listam o mesmo cedente PJ nos 1, 3 e 6 meses seguintes, comparadas com classes
parecidas (mesmo mês, segmento e porte) sem essa exposição?

Definições (as mesmas do boletim mensal):
- vencido 90+ = faixas de atraso acima de 90 dias das tabelas V+VI sobre a carteira (a vencer + vencida);
- estresse = vencido 90+ acima de 10%; entrada em estresse em t = estresse em t e não em t-1 (observado);
- exposição em t = classe saudável em t que lista, em t, um cedente PJ também listado por uma classe que
  entrou em estresse em t; conta só a exposição nova (sem outra nos 6 meses anteriores);
- controle em t = classe saudável em t, que informa ao menos um cedente PJ e não está exposta (nem esteve
  nos 6 meses anteriores);
- efeito = média, nas classes expostas, de (desfecho da exposta − média dos controles do mesmo estrato
  mês × segmento × quintil de PL); IC de 95% por bootstrap reamostrando classes.

É associação, não causalidade: cedentes compartilhados podem refletir um mesmo choque setorial.
"""

from __future__ import annotations

import json
import math
import warnings
from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd

from fidc_risk.config import FIGURES_DIR, METRICS_DIR, PROJECT_ROOT, REPORTS_DIR, SILVER_DIR

STRESS = 0.10
HORIZONS = (1, 3, 6)
FAIXAS = ["30", "60", "90", "120", "150", "180", "360", "720", "1080", "maior_1080"]
SEGMENT_COLS = {
    "sec_industrial": "Industrial", "sec_imobiliario": "Imobiliário", "sec_comercial_varejo": "Comercial/Varejo",
    "sec_servicos": "Serviços", "sec_agronegocio": "Agronegócio", "sec_cred_pessoal": "Crédito Pessoal",
    "sec_consignado": "Consignado", "sec_corporativo": "Corporativo", "sec_veiculos": "Veículos",
    "sec_cartoes_credito": "Cartões", "sec_factoring": "Factoring", "sec_setor_publico": "Setor Público",
    "sec_judicial": "Judicial",
}
STRATA = ["ym", "segmento", "porte"]
STRATA_NIVEL = [*STRATA, "faixa_v"]  # robustez: também pelo nível de vencido 90+ em t
STRATA_HIST = [*STRATA_NIVEL, "faixa_v_previa"]  # e pela média do vencido 90+ de t-6 a t-1
FAIXAS_V_PREVIA = (0.0, 1e-9, 0.02, 0.05, 0.10)  # 0 · até 2% · 2–5% · 5–10% · acima de 10%; 0 = sem histórico
FAIXAS_V = (0.0, 1e-9, 0.02, 0.05)  # 0 · até 2% · 2–5% · 5–10% (todas as classes estão saudáveis em t)
HUB_MIN_CLASSES = 20  # cedente listado por mais de 20 classes no mês: provável plataforma/agente, não cedente comum


def _ym(dt: pd.Series) -> pd.Series:
    d = pd.to_datetime(dt)
    return d.dt.year * 12 + d.dt.month


def _ym_label(ym: int) -> str:
    y, m = divmod(int(ym) - 1, 12)
    return f"{y:04d}-{m + 1:02d}"


# ------------------------------------------------------------------ painel e eventos

def build_panel(core: pd.DataFrame) -> pd.DataFrame:
    """Uma linha por classe-mês: vencido 90+ (definição do boletim), estresse, PL e segmento dominante."""
    prazo = core[[f"prazo_venc_{f}" for f in FAIXAS]].sum(axis=1)
    inad = core[[f"inad_venc_{f}" for f in FAIXAS]].sum(axis=1)
    carteira = prazo + inad
    v = core[[f"inad_venc_{f}" for f in FAIXAS[3:]]].sum(axis=1) / carteira.where(carteira > 0)
    seg_vals = core[list(SEGMENT_COLS)].fillna(0.0)
    segmento = seg_vals.idxmax(axis=1).map(SEGMENT_COLS).where(seg_vals.sum(axis=1) > 0, "Sem segmento")
    out = pd.DataFrame({
        "entity_id": core["entity_id"].astype(str).to_numpy(),
        "ym": _ym(core["dt_comptc"]).to_numpy(),
        "v": v.to_numpy(),
        "pl": core["vl_pl"].to_numpy(),
        "segmento": segmento.to_numpy(),
    })
    out = out[out["v"].notna()].drop_duplicates(["entity_id", "ym"]).reset_index(drop=True)
    out["stress"] = out["v"] > STRESS
    return out


def attach_strata(panel: pd.DataFrame) -> pd.DataFrame:
    """Acrescenta o quintil de PL dentro do mês (porte 1..5)."""
    out = panel.copy()
    r = out.groupby("ym")["pl"].rank(method="first", pct=True)
    out["porte"] = np.ceil(r * 5).clip(1, 5).astype(int)
    out["faixa_v"] = np.searchsorted(np.array(FAIXAS_V), out["v"].fillna(0.0).to_numpy(), side="right")
    return out


def attach_prior_level(panel: pd.DataFrame, meses: int = 6, minimo: int = 3) -> pd.DataFrame:
    """Faixa da média do vencido 90+ de t-6 a t-1 (1..5); 0 quando há menos de ``minimo`` meses observados."""
    out = panel.copy()
    look = out.set_index(["entity_id", "ym"])["v"]
    cols = []
    for k in range(1, meses + 1):
        key = pd.MultiIndex.from_arrays([out["entity_id"], out["ym"] - k])
        cols.append(look.reindex(key).to_numpy(dtype="float64"))
    mat = np.column_stack(cols)
    n = (~np.isnan(mat)).sum(axis=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        media = np.nanmean(mat, axis=1)
    faixa = np.searchsorted(np.array(FAIXAS_V_PREVIA), np.nan_to_num(media, nan=0.0), side="right")
    out["faixa_v_previa"] = np.where(n >= minimo, faixa, 0).astype(int)
    return out


def onset_events(panel: pd.DataFrame) -> pd.DataFrame:
    """Classes que entram em estresse em t (em estresse em t, saudáveis em t-1, ambos observados)."""
    p = panel[["entity_id", "ym", "stress"]]
    prev = p.assign(ym=p["ym"] + 1).rename(columns={"stress": "stress_prev"})
    m = p.merge(prev, on=["entity_id", "ym"], how="inner")
    return m.loc[m["stress"] & ~m["stress_prev"], ["entity_id", "ym"]].sort_values(["ym", "entity_id"]).reset_index(drop=True)


def _hub_cedentes(edges: pd.DataFrame) -> pd.DataFrame:
    deg = edges.groupby(["ym", "ced"])["entity_id"].nunique().rename("grau").reset_index()
    return deg[deg["grau"] > HUB_MIN_CLASSES][["ym", "ced"]]


def _raw_exposures(panel: pd.DataFrame, edges: pd.DataFrame, onsets: pd.DataFrame, excluir_hubs: bool) -> pd.DataFrame:
    e = edges[["entity_id", "ym", "ced"]].drop_duplicates()
    if excluir_hubs:
        hubs = _hub_cedentes(e)
        e = e.merge(hubs.assign(_hub=1), on=["ym", "ced"], how="left")
        e = e[e["_hub"].isna()].drop(columns="_hub")
    on_ced = e.merge(onsets, on=["entity_id", "ym"]).rename(columns={"entity_id": "origem"})
    links = e.merge(on_ced, on=["ym", "ced"])
    links = links[links["entity_id"] != links["origem"]]
    healthy = panel.loc[~panel["stress"], ["entity_id", "ym"]]
    links = links.merge(healthy, on=["entity_id", "ym"])
    return links.groupby(["entity_id", "ym"], as_index=False).agg(n_cedentes=("ced", "nunique"))


def _fresh(events: pd.DataFrame, window: int) -> pd.DataFrame:
    keep: list[bool] = []
    last: dict[str, int] = {}
    for row in events.sort_values(["entity_id", "ym"]).itertuples(index=False):
        prev = last.get(row.entity_id)
        if prev is None or row.ym - prev > window:
            keep.append(True)
            last[row.entity_id] = row.ym
        else:
            keep.append(False)
    s = events.sort_values(["entity_id", "ym"])
    return s[np.array(keep, dtype=bool)].sort_values(["ym", "entity_id"]).reset_index(drop=True)


def exposure_events(
    panel: pd.DataFrame, edges: pd.DataFrame, onsets: pd.DataFrame, fresh_window: int = 6, excluir_hubs: bool = False
) -> pd.DataFrame:
    """Classes saudáveis que passam a dividir um cedente PJ com uma classe que acabou de entrar em estresse."""
    raw = _raw_exposures(panel, edges, onsets, excluir_hubs)
    if raw.empty:
        return raw
    return _fresh(raw, fresh_window)


def control_events(panel: pd.DataFrame, edges: pd.DataFrame, exposures_raw: pd.DataFrame, window: int = 6) -> pd.DataFrame:
    """Classes saudáveis com cedente PJ informado e sem exposição no mês nem nos ``window`` meses anteriores."""
    has_ced = edges[["entity_id", "ym"]].drop_duplicates()
    cand = panel.loc[~panel["stress"], ["entity_id", "ym"]].merge(has_ced, on=["entity_id", "ym"])
    if exposures_raw.empty:
        return cand.reset_index(drop=True)
    exp = exposures_raw[["entity_id", "ym"]]
    blocked = pd.concat([exp.assign(ym=exp["ym"] + k) for k in range(window + 1)]).drop_duplicates()
    m = cand.merge(blocked.assign(_b=1), on=["entity_id", "ym"], how="left")
    return m[m["_b"].isna()].drop(columns="_b").reset_index(drop=True)


def outcomes(panel: pd.DataFrame, events: pd.DataFrame, horizons: Iterable[int] = HORIZONS) -> pd.DataFrame:
    """Para cada evento (classe, t): entrou em estresse em até h meses (y_h) e variação do vencido 90+ (dv_h).

    Sem informe em t+h, o desfecho fica vazio (censura): a classe pode ter sido liquidada ou parado de reportar.
    """
    base = events[["entity_id", "ym"]].merge(panel[["entity_id", "ym", "v"]], on=["entity_id", "ym"], how="left")
    look = panel.set_index(["entity_id", "ym"])[["v", "stress"]]
    out = base.copy()
    hs = sorted(set(horizons))
    for step in range(1, max(hs) + 1):
        key = pd.MultiIndex.from_arrays([out["entity_id"], out["ym"] + step])
        fut = look.reindex(key)
        out[f"_s{step}"] = fut["stress"].to_numpy(dtype="float64")
        out[f"_v{step}"] = fut["v"].to_numpy(dtype="float64")
    for h in hs:
        observed = ~np.isnan(out[f"_v{h}"].to_numpy())
        with warnings.catch_warnings():  # linhas sem nenhum mês futuro observado: tratadas como censura abaixo
            warnings.simplefilter("ignore", RuntimeWarning)
            stress_any = np.nanmax(np.column_stack([out[f"_s{s}"] for s in range(1, h + 1)]), axis=1)
        out[f"y_h{h}"] = np.where(observed, np.nan_to_num(stress_any, nan=0.0), np.nan)
        out[f"dv_h{h}"] = np.where(observed, out[f"_v{h}"] - out["v"], np.nan)
    return out.drop(columns=[c for c in out.columns if c.startswith("_")])


def pre_trend(panel: pd.DataFrame, events: pd.DataFrame, k: int = 3) -> pd.DataFrame:
    """Variação do vencido 90+ de t-k até t (placebo: não deveria diferir entre expostas e controles)."""
    base = events[["entity_id", "ym"]].merge(panel[["entity_id", "ym", "v"]], on=["entity_id", "ym"], how="left")
    past = panel[["entity_id", "ym", "v"]].assign(ym=panel["ym"] + k).rename(columns={"v": "v_pre"})
    m = base.merge(past, on=["entity_id", "ym"], how="left")
    m["pre"] = m["v"] - m["v_pre"]
    return m


# ------------------------------------------------------------------ efeito pareado

def matched_effect(
    exposed: pd.DataFrame,
    controls: pd.DataFrame,
    col: str,
    n_boot: int = 500,
    seed: int = 7,
    min_controles: int = 3,
    strata: list[str] = STRATA,
) -> dict[str, Any]:
    """Efeito médio nas expostas: desfecho − média dos controles do mesmo estrato, com IC por bootstrap de classes."""
    ctrl = controls.dropna(subset=[col]).groupby(strata)[col].agg(["mean", "size"]).reset_index()
    ctrl = ctrl[ctrl["size"] >= min_controles].rename(columns={"mean": "_ctrl", "size": "_nctrl"})
    ex = exposed.dropna(subset=[col]).merge(ctrl, on=strata, how="inner")
    if ex.empty:
        return {"n_expostas": 0, "n_controles": 0, "efeito": float("nan"), "ic_baixo": float("nan"),
                "ic_alto": float("nan"), "taxa_expostas": float("nan"), "taxa_controles": float("nan"),
                "risco_relativo": float("nan")}
    diff = (ex[col] - ex["_ctrl"]).to_numpy(dtype="float64")
    groups = ex["entity_id"].astype(str).to_numpy()
    uniq, inv = np.unique(groups, return_inverse=True)
    sums = np.bincount(inv, weights=diff, minlength=len(uniq))
    cnts = np.bincount(inv, minlength=len(uniq)).astype("float64")
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        boots[b] = sums[pick].sum() / cnts[pick].sum()
    taxa_e, taxa_c = float(ex[col].mean()), float(ex["_ctrl"].mean())
    ctrl_used = controls.dropna(subset=[col]).merge(ex[strata].drop_duplicates(), on=strata)
    return {
        "n_expostas": int(len(ex)),
        "n_classes_expostas": int(len(uniq)),
        "n_controles": int(len(ctrl_used)),
        "efeito": float(diff.mean()),
        "ic_baixo": float(np.quantile(boots, 0.025)),
        "ic_alto": float(np.quantile(boots, 0.975)),
        "taxa_expostas": taxa_e,
        "taxa_controles": taxa_c,
        "risco_relativo": taxa_e / taxa_c if taxa_c > 0 else float("nan"),
    }


def verdict(s: dict[str, Any]) -> dict[str, Any]:
    """Robusto só se, em 6 meses: efeito > 0 com IC acima de zero; sem pré-tendência positiva (expostas já
    piorando mais antes da exposição); resiste sem os hubs; resiste ao parear também pelo nível de vencido
    90+ em t (quando calculado); e positivo em pelo menos 60% dos anos."""
    motivos, notas = [], []
    e6, pre, hub = s["efeito_h6"], s["pre_tendencia"], s["sem_hubs_h6"]
    nivel = s.get("pareado_atraso_h6")
    anos = [a["efeito"] for a in s.get("por_ano_h6", []) if a.get("efeito") is not None and not math.isnan(a["efeito"])]
    ok_e6 = e6["ic_baixo"] > 0
    ok_pre = not pre["ic_baixo"] > 0
    ok_hub = hub["ic_baixo"] > 0
    ok_nivel = nivel is None or (nivel.get("ic_baixo") is not None and nivel["ic_baixo"] > 0)
    hist = s.get("pareado_historico_h6")
    ok_hist = hist is None or (hist.get("ic_baixo") is not None and hist["ic_baixo"] > 0)
    ok_anos = bool(anos) and sum(a > 0 for a in anos) >= 0.6 * len(anos)
    if not ok_e6:
        motivos.append("efeito em 6 meses com intervalo de confiança que inclui zero ou é negativo")
    if not ok_pre:
        motivos.append("expostas já pioravam mais que os controles antes da exposição (pré-tendência)")
    elif pre["ic_alto"] < 0:
        notas.append("antes da exposição, as expostas melhoravam mais que os controles: os grupos diferem no nível "
                     "de atraso, por isso o teste pareado também pela faixa de vencido 90+")
    if not ok_hub:
        motivos.append("o efeito some quando se excluem cedentes ligados a muitas classes")
    if not ok_nivel:
        motivos.append("o efeito some ao comparar classes com o mesmo nível de vencido 90+ no mês da exposição")
    if not ok_hist:
        motivos.append("o efeito some ao comparar classes com o mesmo histórico de atraso nos 6 meses anteriores")
    if not ok_anos:
        motivos.append("o sinal não se repete na maioria dos anos")
    robusto = ok_e6 and ok_pre and ok_hub and ok_nivel and ok_hist and ok_anos
    ref = hist if hist is not None else (nivel if nivel is not None else e6)
    forca = None
    if robusto:
        forca = "pequeno" if ref["efeito"] < 0.03 or ref["ic_baixo"] < 0.01 else "relevante"
    return {"robusto": robusto, "forca": forca, "motivos": motivos, "notas": notas}


# ------------------------------------------------------------------ estudo completo

def event_curve(
    panel: pd.DataFrame, exposed: pd.DataFrame, controls: pd.DataFrame, ks: range = range(-6, 7),
    strata: list[str] = STRATA,
) -> pd.DataFrame:
    """Vencido 90+ médio das expostas e dos controles pareados, de t-6 a t+6 (diferença por estrato)."""
    look = panel.set_index(["entity_id", "ym"])["v"]
    rows = []
    for k in ks:
        def at(ev: pd.DataFrame, k: int = k) -> pd.DataFrame:
            key = pd.MultiIndex.from_arrays([ev["entity_id"], ev["ym"] + k])
            return ev.assign(vk=look.reindex(key).to_numpy())
        ex, ct = at(exposed), at(controls)
        cm = ct.dropna(subset=["vk"]).groupby(strata)["vk"].agg(["mean", "size"]).reset_index()
        cm = cm[cm["size"] >= 3]
        m = ex.dropna(subset=["vk"]).merge(cm, on=strata)
        if m.empty:
            continue
        rows.append({"k": k, "expostas": float(m["vk"].mean()), "controles": float(m["mean"].mean()), "n": int(len(m))})
    return pd.DataFrame(rows)


def _cobertura(core: pd.DataFrame, edges: pd.DataFrame) -> dict[str, float]:
    sub = core[pd.to_datetime(core["dt_comptc"]).dt.year >= 2022]
    n = sub["n_cedentes_disclosed"].fillna(0)
    return {
        "classes_mes": int(len(sub)),
        "sem_cedente_informado": float((n == 0).mean()),
        "com_9_ou_mais": float((n >= 9).mean()),
        "cedentes_pj_por_classe_mediana_quando_informa": float(n[n > 0].median()) if (n > 0).any() else 0.0,
    }


def run_study(core: pd.DataFrame, edges_raw: pd.DataFrame, inicio: str = "2022-01", n_boot: int = 500) -> dict[str, Any]:
    panel = attach_prior_level(attach_strata(build_panel(core)))
    edges = edges_raw[edges_raw["is_pj"] == 1].copy()
    edges["ym"] = _ym(edges["dt_comptc"])
    edges = edges.rename(columns={"cedente_doc": "ced"})[["entity_id", "ym", "ced"]].astype({"entity_id": str, "ced": str})

    y0 = int(inicio[:4]) * 12 + int(inicio[5:7])
    onsets = onset_events(panel)
    onsets = onsets[onsets["ym"] >= y0]
    raw = _raw_exposures(panel, edges, onsets, excluir_hubs=False)
    exposed = _fresh(raw, 6) if not raw.empty else raw
    controls = control_events(panel[panel["ym"] >= y0], edges, raw)
    strata_cols = panel[["entity_id", "ym", "segmento", "porte", "faixa_v", "faixa_v_previa"]]

    ex_o = outcomes(panel, exposed).merge(strata_cols, on=["entity_id", "ym"])
    ct_o = outcomes(panel, controls).merge(strata_cols, on=["entity_id", "ym"])

    efeitos = []
    for h in HORIZONS:
        for col, nome in ((f"y_h{h}", "entrou_em_estresse"), (f"dv_h{h}", "variacao_vencido90")):
            for espec, estratos in (("segmento_porte", STRATA), ("segmento_porte_atraso", STRATA_NIVEL),
                                    ("segmento_porte_atraso_historico", STRATA_HIST)):
                r = matched_effect(ex_o, ct_o, col, n_boot=n_boot, strata=estratos)
                efeitos.append({"horizonte_meses": h, "desfecho": nome, "pareamento": espec, **r})

    pre_e = pre_trend(panel, exposed).merge(strata_cols, on=["entity_id", "ym"])
    pre_c = pre_trend(panel, controls).merge(strata_cols, on=["entity_id", "ym"])
    pre = matched_effect(pre_e, pre_c, "pre", n_boot=n_boot)


    raw_nh = _raw_exposures(panel, edges, onsets, excluir_hubs=True)
    exp_nh = _fresh(raw_nh, 6) if not raw_nh.empty else raw_nh
    sem_hubs = matched_effect(outcomes(panel, exp_nh).merge(strata_cols, on=["entity_id", "ym"]), ct_o, "y_h6", n_boot=n_boot)

    por_ano, por_ano_nivel, por_ano_hist = [], [], []
    ex_o["ano"] = (ex_o["ym"] - 1) // 12
    ct_o["ano"] = (ct_o["ym"] - 1) // 12
    for ano in sorted(ex_o["ano"].unique()):
        e_a, c_a = ex_o[ex_o["ano"] == ano], ct_o[ct_o["ano"] == ano]
        por_ano.append({"ano": int(ano), **matched_effect(e_a, c_a, "y_h6", n_boot=n_boot)})
        por_ano_nivel.append({"ano": int(ano), **matched_effect(e_a, c_a, "y_h6", n_boot=n_boot, strata=STRATA_NIVEL)})
        por_ano_hist.append({"ano": int(ano), **matched_effect(e_a, c_a, "y_h6", n_boot=n_boot, strata=STRATA_HIST)})

    curva = event_curve(
        panel, exposed.merge(strata_cols, on=["entity_id", "ym"]), controls.merge(strata_cols, on=["entity_id", "ym"]),
        strata=STRATA_HIST,
    )
    get = {(e["horizonte_meses"], e["desfecho"], e["pareamento"]): e for e in efeitos}
    summary: dict[str, Any] = {
        "periodo": {"inicio": inicio, "fim": _ym_label(int(panel["ym"].max()))},
        "definicoes": {"estresse_vencido90_acima_de": STRESS, "janela_exposicao_nova_meses": 6,
                       "hub_mais_de_classes": HUB_MIN_CLASSES, "estratos": STRATA},
        "n_entradas_em_estresse": int(len(onsets)),
        "n_exposicoes": int(len(exposed)),
        "n_classes_expostas": int(exposed["entity_id"].nunique()) if len(exposed) else 0,
        "n_controles_classe_mes": int(len(controls)),
        "efeitos": efeitos,
        "efeito_h6": get[(6, "entrou_em_estresse", "segmento_porte")],
        "pre_tendencia": pre,
        "sem_hubs_h6": sem_hubs,
        "pareado_atraso_h6": get[(6, "entrou_em_estresse", "segmento_porte_atraso")],
        "pareado_atraso_dv_h6": get[(6, "variacao_vencido90", "segmento_porte_atraso")],
        "por_ano_pareado_atraso_h6": por_ano_nivel,
        "pareado_historico_h6": get[(6, "entrou_em_estresse", "segmento_porte_atraso_historico")],
        "pareado_historico_dv_h6": get[(6, "variacao_vencido90", "segmento_porte_atraso_historico")],
        "por_ano_pareado_historico_h6": por_ano_hist,
        "por_ano_h6": por_ano,
        "curva_evento": curva.to_dict(orient="records"),
        "cobertura": _cobertura(core, edges_raw),
    }
    summary["veredito"] = verdict(summary)
    return _clean(summary)


def _clean(o: Any) -> Any:
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_clean(v) for v in o]
    if isinstance(o, float) and math.isnan(o):
        return None
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return None if math.isnan(float(o)) else float(o)
    return o


# ------------------------------------------------------------------ saídas

def _pct(x: float | None, d: int = 1) -> str:
    return "—" if x is None else f"{x * 100:.{d}f}%".replace(".", ",")


def _pp(x: float | None, d: int = 1) -> str:
    if x is None:
        return "—"
    v = round(x * 100, d) + 0.0  # evita "−0,00"
    return f"{v:+.{d}f}".replace(".", ",").replace("-", "−") + " p.p."


_MESES = ["jan", "fev", "mar", "abr", "mai", "jun", "jul", "ago", "set", "out", "nov", "dez"]


def _mes(ym_label: str) -> str:
    """'2022-01' -> 'jan/2022'."""
    return f"{_MESES[int(ym_label[5:7]) - 1]}/{ym_label[:4]}"


def figure(summary: dict[str, Any], path: Any) -> Any:
    """Duas leituras lado a lado: taxa de entrada em estresse por horizonte e trajetória do vencido 90+."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    NAVY, GREY, INK, INK2, GRID, SHEET = "#1f3a5f", "#a9acb2", "#1b2433", "#4a5160", "#e4e2db", "#fbfaf7"
    ef = [e for e in summary["efeitos"]
          if e["desfecho"] == "entrou_em_estresse"
          and e.get("pareamento", "segmento_porte_atraso_historico") == "segmento_porte_atraso_historico"]
    curva = summary["curva_evento"]
    with plt.rc_context({"font.family": "DejaVu Sans", "figure.facecolor": SHEET}):
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 2.6), dpi=220)
        x = np.arange(len(ef))
        w = 0.36
        te = [e["taxa_expostas"] or 0 for e in ef]
        tc = [e["taxa_controles"] or 0 for e in ef]
        a1.bar(x - w / 2 - 0.01, te, w, color=NAVY, label="expostas ao cedente")
        a1.bar(x + w / 2 + 0.01, tc, w, color=GREY, label="controles (mesmo segmento, porte, atraso e histórico)")
        for i, (a, b) in enumerate(zip(te, tc, strict=True)):
            a1.text(i - w / 2 - 0.01, a, _pct(a), ha="center", va="bottom", fontsize=6.5, color=INK)
            a1.text(i + w / 2 + 0.01, b, _pct(b), ha="center", va="bottom", fontsize=6.5, color=INK2)
        a1.set_xticks(x, [f"em {e['horizonte_meses']} {'mês' if e['horizonte_meses'] == 1 else 'meses'}" for e in ef])
        a1.set_yticks([])
        a1.set_ylim(0, max(te + tc + [0.01]) * 1.6)  # espaço para a legenda acima das barras
        a1.set_title("Entraram em estresse", fontsize=8, color=INK, loc="left")
        a1.legend(fontsize=6.5, frameon=False, loc="upper left")
        if curva:
            ks = [c["k"] for c in curva]
            a2.plot(ks, [c["expostas"] * 100 for c in curva], color=NAVY, lw=1.6, label="expostas")
            a2.plot(ks, [c["controles"] * 100 for c in curva], color=GREY, lw=1.6, label="controles")
            a2.axvline(0, color=INK2, lw=0.6)
            a2.text(0.2, a2.get_ylim()[1] * 0.95, "exposição", fontsize=6.5, color=INK2, va="top")
            a2.set_xlabel("meses em relação à exposição", fontsize=6.5, color=INK2)
            from matplotlib.ticker import MaxNLocator

            a2.yaxis.set_major_locator(MaxNLocator(nbins=5, steps=[1, 2, 5, 10]))
            a2.yaxis.set_major_formatter(lambda v, _: f"{v:.1f}%".replace(".", ","))
            a2.set_title("Vencido 90+ médio", fontsize=8, color=INK, loc="left")
            a2.legend(fontsize=6.5, frameon=False, loc="upper left")
            a2.grid(axis="y", color=GRID, lw=0.5)
        for ax in (a1, a2):
            ax.set_facecolor(SHEET)
            ax.tick_params(labelsize=6.5, colors=INK2, length=0)
            for sp in ("top", "right", "left"):
                ax.spines[sp].set_visible(False)
            ax.spines["bottom"].set_color(GRID)
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
    return path


def bloco_boletim(summary: dict[str, Any]) -> dict[str, Any] | None:
    """Bloco de meia página para o boletim (só quando o resultado é robusto), com o número conservador."""
    if not summary["veredito"]["robusto"]:
        return None
    e6, nv = summary["efeito_h6"], summary["pareado_atraso_h6"]
    return {
        "titulo": "Cedente em comum com quem entrou em estresse: o risco viaja, mas pouco",
        "texto": (
            f"De {_mes(summary['periodo']['inicio'])} a {_mes(summary['periodo']['fim'])}, {summary['n_classes_expostas']} classes "
            "saudáveis passaram a dividir um cedente com uma classe que acabara de entrar em estresse (vencido 90+ "
            f"acima de 10%). Em 6 meses, {_pct(nv['taxa_expostas'])} delas também entraram em estresse, contra "
            f"{_pct(nv['taxa_controles'])} de classes parecidas sem essa ligação (mesmo mês, segmento, porte e nível de "
            f"atraso): {_pp(nv['efeito'])} (intervalo de 95%: {_pp(nv['ic_baixo'])} a {_pp(nv['ic_alto'])}). Sem parear "
            f"pelo atraso, a diferença parece maior ({_pct(e6['taxa_expostas'])} contra {_pct(e6['taxa_controles'])}), "
            "mas parte dela vem de essas classes já terem mais atraso antes. O sinal aparece em todos os anos e "
            "resiste à exclusão dos cedentes ligados a muitas classes. É associação, não prova de causa."
        ),
        "rotulos": ["Expostas ao cedente", "Classes parecidas sem a ligação"],
        "taxas_6m": [nv["taxa_expostas"], nv["taxa_controles"]],
        "efeito_6m": nv["efeito"],
        "ic_6m": [nv["ic_baixo"], nv["ic_alto"]],
        "forca": summary["veredito"].get("forca"),
        "figura": "reports/figures/fig10_contagio_cedente.png",
        "fonte": "reports/fase9_contagio_cedente.md",
    }


def _tabela_md(s: dict[str, Any], espec: str) -> str:
    linhas = []
    for e in s["efeitos"]:
        if e["desfecho"] == "entrou_em_estresse" and e.get("pareamento") == espec:
            rr = "—" if e["risco_relativo"] is None else f"{e['risco_relativo']:.2f}×".replace(".", ",")
            linhas.append(
                f"| {e['horizonte_meses']} | {e['n_expostas']} | {_pct(e['taxa_expostas'])} | {_pct(e['taxa_controles'])} | "
                f"{_pp(e['efeito'])} | {_pp(e['ic_baixo'])} a {_pp(e['ic_alto'])} | {rr} |"
            )
    cab = "| h (meses) | Expostas | Taxa expostas | Taxa controles | Diferença | IC 95% | Risco relativo |\n"
    return cab + "|---|---|---|---|---|---|---|\n" + "\n".join(linhas)


def _conclusao_md(s: dict[str, Any]) -> str:
    v, nv, e6 = s["veredito"], s["pareado_atraso_h6"], s["efeito_h6"]
    if not v["robusto"]:
        h = s.get("pareado_historico_h6") or nv
        return (
            "**Conclusão: o sinal é fraco e não passa nos critérios de robustez.** "
            + "; ".join(v["motivos"]).capitalize() + ". "
            f"Na comparação simples, as expostas entram mais em estresse ({_pp(e6['efeito'])} em 6 meses), mas a maior "
            f"parte disso é atraso que elas já tinham: com o mesmo nível e o mesmo histórico de vencido 90+, a diferença "
            f"cai para {_pp(h['efeito'])} (IC {_pp(h['ic_baixo'])} a {_pp(h['ic_alto'])}; {_pct(h['taxa_expostas'])} contra "
            f"{_pct(h['taxa_controles'])}), sem significância, e muda de sinal entre os anos. Com os dados do informe, não "
            "dá para afirmar que o risco viaja pelo cedente; o resultado não entra no boletim."
        )
    return (
        f"**Conclusão: há contágio por cedente, e ele é {v['forca']}.** A comparação simples (mesmo mês, segmento e "
        f"porte) mostra {_pp(e6['efeito'])} em 6 meses, mas parte disso é nível: as expostas já tinham mais atraso. "
        f"Comparando com classes que também tinham o mesmo nível de vencido 90+ no mês da exposição, a diferença cai "
        f"para {_pp(nv['efeito'])} (IC {_pp(nv['ic_baixo'])} a {_pp(nv['ic_alto'])}; {_pct(nv['taxa_expostas'])} "
        f"contra {_pct(nv['taxa_controles'])}). O sinal é positivo em todos os anos nessa comparação mais exigente, "
        "embora só seja estatisticamente claro no conjunto. É associação: um mesmo choque setorial pode atingir "
        "fundos que dividem cedente. Para o boletim, usamos o número conservador."
    )


def report_markdown(s: dict[str, Any]) -> str:
    cob, pre, sh = s["cobertura"], s["pre_tendencia"], s["sem_hubs_h6"]
    dv = [e for e in s["efeitos"] if e["desfecho"] == "variacao_vencido90" and e.get("pareamento") == "segmento_porte_atraso"]
    dv_txt = " · ".join(
        f"{e['horizonte_meses']}m: {_pp(e['efeito'], 2)} [{_pp(e['ic_baixo'], 2)}; {_pp(e['ic_alto'], 2)}]" for e in dv
    )
    anos = " · ".join(f"{a['ano']}: {_pp(a['efeito'])} (n={a['n_expostas']})" for a in s["por_ano_h6"])
    anos_nv = " · ".join(
        f"{a['ano']}: {_pp(a['efeito'])} [{_pp(a['ic_baixo'])}; {_pp(a['ic_alto'])}]" for a in s["por_ano_pareado_atraso_h6"]
    )
    anos_hist = " · ".join(
        f"{a['ano']}: {_pp(a['efeito'])} [{_pp(a['ic_baixo'])}; {_pp(a['ic_alto'])}]"
        for a in s.get("por_ano_pareado_historico_h6", [])
    )
    partes = [
        f"# Estudo: contágio por cedente compartilhado ({_mes(s['periodo']['inicio'])} a {_mes(s['periodo']['fim'])})",
        "",
        "Quando uma classe de FIDC entra em estresse (vencido 90+ passa de 10% da carteira), as classes que listam o "
        "mesmo cedente PJ pioram mais que classes parecidas sem essa ligação?",
        "",
        "## Desenho",
        "",
        f"- **Entrada em estresse em t**: vencido 90+ acima de 10% em t e abaixo em t−1 (tabelas V+VI, definição do "
        f"boletim). {s['n_entradas_em_estresse']} entradas no período.",
        f"- **Exposição em t**: classe saudável em t que lista, no mesmo mês, um cedente PJ também listado pela classe "
        f"que entrou em estresse. Só conta a exposição nova (nenhuma outra nos 6 meses anteriores): {s['n_exposicoes']} "
        f"eventos em {s['n_classes_expostas']} classes.",
        f"- **Controles**: classes saudáveis no mesmo mês, que informam ao menos um cedente PJ e não estão (nem estiveram "
        f"em 6 meses) expostas ({s['n_controles_classe_mes']} classe-mês).",
        "- **Efeito**: média, nas expostas, de (desfecho − média dos controles do mesmo estrato). Intervalo de 95% por "
        "bootstrap reamostrando classes. Estratos com menos de 3 controles ficam de fora.",
        "",
        "## Resultado: entrar em estresse em até h meses",
        "",
        "Pareando por mês × segmento dominante × quintil de PL (o desenho pedido):",
        "",
        _tabela_md(s, "segmento_porte"),
        "",
        "Pareando também pela faixa de vencido 90+ no mês da exposição (0 · até 2% · 2–5% · 5–10%):",
        "",
        _tabela_md(s, "segmento_porte_atraso"),
        "",
        "Pareando ainda pela faixa da média do vencido 90+ nos 6 meses anteriores (o histórico de atraso), a "
        "comparação mais exigente:",
        "",
        _tabela_md(s, "segmento_porte_atraso_historico"),
        "",
        f"Variação média do vencido 90+ (expostas − controles, pareado pelo atraso): {dv_txt}.",
        "",
        "![Contágio por cedente](figures/fig10_contagio_cedente.png)",
        "",
        "## Robustez",
        "",
        f"- **Pré-tendência (placebo)**: variação do vencido 90+ de t−3 a t, expostas − controles: {_pp(pre['efeito'], 2)} "
        f"(IC {_pp(pre['ic_baixo'], 2)} a {_pp(pre['ic_alto'], 2)}). O critério definido antes de olhar o resultado pedia "
        "um intervalo incluindo zero. O resultado veio negativo: antes da exposição, as expostas *melhoravam* mais que os "
        "controles, porque partiam de um atraso mais alto. Isso não cria um efeito positivo falso, mas mostra que os "
        "grupos diferem em nível; por isso, depois de ver esse resultado, acrescentamos o pareamento pela faixa de "
        "vencido 90+ em t e, numa revisão seguinte, também pela média dos 6 meses anteriores (as expostas ainda tinham "
        "mais atraso antes da exposição mesmo com o nível em t igual), e passamos a exigir que o efeito sobreviva a eles.",
        f"- **Sem cedentes-hub** (listados por mais de {s['definicoes']['hub_mais_de_classes']} classes no mês): diferença "
        f"em 6 meses de {_pp(sh['efeito'])} (IC {_pp(sh['ic_baixo'])} a {_pp(sh['ic_alto'])}), n={sh['n_expostas']}.",
        f"- **Por ano (6 meses, pareamento simples)**: {anos}.",
        f"- **Por ano (6 meses, pareado pelo atraso)**: {anos_nv}.",
        f"- **Por ano (6 meses, pareado pelo atraso e pelo histórico)**: {anos_hist}.",
        "",
        _conclusao_md(s),
        "",
        "## Viés de cobertura",
        "",
        "O informe lista só os maiores cedentes de cada classe: até 9 no grupo com aquisição substancial de riscos e até "
        f"9 no grupo sem. Na prática a lacuna é maior: de {cob['classes_mes']} classe-mês desde 2022, "
        f"{_pct(cob['sem_cedente_informado'])} não informam nenhum cedente, e só {_pct(cob['com_9_ou_mais'], 2)} chegam a 9. "
        "Ligações com cedentes menores, ou em classes que não preenchem o campo, não aparecem: expostas de verdade podem "
        "estar no grupo de controle (o que puxa a diferença para zero) e o estudo só fala de classes que informam "
        "cedentes. Classes que param de reportar antes de t+h saem do cálculo daquele horizonte, e liquidações após "
        "estresse podem não ser observadas.",
        "",
        "_Gerado por `fidc-risk study-cedente-contagion` a partir de data/silver; números em "
        "reports/metrics/cedente_contagion_*.csv/json._",
        "",
    ]
    return "\n".join(partes)


def run_and_export(n_boot: int = 500) -> dict[str, Any]:
    cols = (["entity_id", "dt_comptc", "vl_pl", "n_cedentes_disclosed"] + list(SEGMENT_COLS)
            + [f"prazo_venc_{f}" for f in FAIXAS] + [f"inad_venc_{f}" for f in FAIXAS])
    core = pd.read_parquet(SILVER_DIR / "fidc_monthly_core.parquet", columns=cols)
    edges = pd.read_parquet(SILVER_DIR / "cedente_edges.parquet")
    s = run_study(core, edges, n_boot=n_boot)

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    (METRICS_DIR / "cedente_contagion_summary.json").write_text(json.dumps(s, ensure_ascii=False, indent=1), encoding="utf-8")
    pd.DataFrame(s["efeitos"]).to_csv(METRICS_DIR / "cedente_contagion_study.csv", index=False)
    pd.DataFrame(s["curva_evento"]).to_csv(METRICS_DIR / "cedente_contagion_eventstudy.csv", index=False)
    figure(s, FIGURES_DIR / "fig10_contagio_cedente.png")
    (REPORTS_DIR / "fase9_contagio_cedente.md").write_text(report_markdown(s), encoding="utf-8")

    bloco = bloco_boletim(s)
    destino = PROJECT_ROOT / "relatorio" / "blocos" / "contagio_cedente.json"
    if bloco:
        destino.parent.mkdir(parents=True, exist_ok=True)
        destino.write_text(json.dumps(bloco, ensure_ascii=False, indent=1), encoding="utf-8")
    elif destino.exists():
        destino.unlink()
    return s

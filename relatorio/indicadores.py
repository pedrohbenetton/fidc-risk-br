"""Indicadores agregados da indústria de FIDC a partir do Informe Mensal da CVM.

Uso: python -m relatorio.indicadores <dir_mes_atual> <dir_mes_anterior> <dir_ano_anterior> <saida.json>
Cada diretório contém os CSVs inf_mensal_fidc_tab_*_AAAAMM.csv. Nenhum fundo é identificado na saída.
"""
from __future__ import annotations

import glob
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd

SEG = {
    "TAB_II_A_VL_INDUST": "Industrial", "TAB_II_B_VL_IMOBIL": "Imobiliário", "TAB_II_C_VL_COMERC": "Comercial/varejo",
    "TAB_II_D_VL_SERV": "Serviços", "TAB_II_E_VL_AGRONEG": "Agronegócio", "TAB_II_F_VL_FINANC": "Financeiro",
    "TAB_II_G_VL_CREDITO": "Cartão de crédito", "TAB_II_H_VL_FACTOR": "Factoring", "TAB_II_I_VL_SETOR_PUBLICO": "Setor público",
    "TAB_II_J_VL_JUDICIAL": "Judicial", "TAB_II_K_VL_MARCA": "Propriedade intelectual",
}
STRESS_INAD90 = 0.10  # classe "sob estresse": inadimplência >90d acima de 10% da carteira de DCs

_COMPETENCIA_RE = re.compile(r"inf_mensal_fidc_tab_I_(\d{4})(\d{2})\.csv$", re.IGNORECASE)


def _read(d: str, tab: str) -> pd.DataFrame:
    f = glob.glob(os.path.join(d, f"inf_mensal_fidc_tab_{tab}_*.csv"))
    df = pd.read_csv(f[0], sep=";", encoding="latin1", dtype=str)
    df["k"] = df["CNPJ_FUNDO_CLASSE"].fillna("").str.replace(r"\D", "", regex=True)
    return df


def _num(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    return df


FAIXAS = ["30", "60", "90", "120", "150", "180", "360", "720", "1080", "MAIOR_1080"]
PDD_COLS = ["TAB_I2A11_VL_REDUCAO_RECUP", "TAB_I2B11_VL_REDUCAO_RECUP"]


def _pl(d: str) -> pd.Series:
    return _num(_read(d, "IV"), ["TAB_IV_A_VL_PL"]).groupby("k")["TAB_IV_A_VL_PL"].sum()


def _aging(d: str) -> pd.DataFrame:
    """Carteira por classe nas tabelas V + VI: a vencer, vencido total e acumulados por faixa de atraso."""
    aging = {}
    for tab in ("V", "VI"):
        df = _read(d, tab)
        p = f"TAB_{tab}_"
        a = [f"{p}A{i}_VL_PRAZO_VENC_{s}" for i, s in enumerate(FAIXAS, 1)]
        b = [f"{p}B{i}_VL_INAD_{s}" for i, s in enumerate(FAIXAS, 1)]
        df = _num(df, a + b).groupby("k")[a + b].sum()
        aging[tab] = pd.DataFrame({
            "avencer": df[a].sum(axis=1), "inad": df[b].sum(axis=1),
            "inad30": df[b[1:]].sum(axis=1), "inad90": df[b[3:]].sum(axis=1), "inad180": df[b[6:]].sum(axis=1),
            "inad1_30": df[b[0]], "inad31_60": df[b[1]], "inad61_90": df[b[2]],
        })
    ag = aging["V"].add(aging["VI"], fill_value=0)
    ag["carteira"] = ag["avencer"] + ag["inad"]
    return ag


def _pdd(t1: pd.DataFrame) -> pd.Series:
    """Provisão por classe: |I2A11| + |I2B11| (redução ao valor recuperável)."""
    t1 = _num(t1, PDD_COLS)
    return t1.groupby("k")[PDD_COLS].sum().abs().sum(axis=1)


def _cedentes(t1: pd.DataFrame) -> pd.DataFrame:
    """Cedentes PJ listados na tabela I (até 9 por grupo A/B): k, ced (14 dígitos) e % da carteira."""
    parts = []
    for grp in ("A", "B"):
        for i in range(1, 10):
            c = f"TAB_I2{grp}12_CPF_CNPJ_CEDENTE_{i}"
            pr = f"TAB_I2{grp}12_PR_CEDENTE_{i}"
            if c in t1:
                s = t1[["k", c]].rename(columns={c: "ced"})
                s["pct"] = pd.to_numeric(t1[pr], errors="coerce") if pr in t1 else float("nan")
                parts.append(s)
    if not parts:
        return pd.DataFrame(columns=["k", "ced", "pct"])
    ced = pd.concat(parts)
    ced["ced"] = ced["ced"].fillna("").str.replace(r"\D", "", regex=True)
    return ced[ced["ced"].str.len() == 14]


def cedentes_pj(d: str) -> pd.DataFrame:
    """Ligações classe -> cedente PJ da competência (sem repetição; % é o maior informado)."""
    ced = _cedentes(_read(d, "I"))
    return ced.groupby(["k", "ced"], as_index=False)["pct"].max()


def por_classe(d: str) -> pd.DataFrame:
    """Indicadores por classe (índice: CNPJ só com dígitos), com as mesmas definições do boletim.

    vencido90_pct: faixas a partir de 91 dias nas tabelas V+VI sobre a carteira (a vencer + vencido);
    atraso_1_60_pct: faixas até 60 dias; pdd_pct: |I2A11| + |I2B11| sobre a mesma carteira;
    em_estresse: vencido 90+ acima de 10% da carteira.
    """
    t1 = _read(d, "I")
    nomes = t1.groupby("k")["DENOM_SOCIAL"].first()
    pl, ag, pdd = _pl(d), _aging(d), _pdd(t1)
    vii = _num(_read(d, "VII"), ["TAB_VII_D_2_VL_RECOMPRA"]).groupby("k")["TAB_VII_D_2_VL_RECOMPRA"].sum()
    seg = _num(_read(d, "II"), list(SEG)).groupby("k")[list(SEG)].sum()
    dom = seg.idxmax(axis=1).where(seg.sum(axis=1) > 0).map(SEG)

    idx = sorted(set(nomes.index) | set(pl.index) | set(ag.index))
    out = pd.DataFrame(index=pd.Index(idx, name="k"))
    out["nome"] = nomes.reindex(out.index)
    out["pl"] = pl.reindex(out.index).fillna(0.0)
    for c in ("carteira", "inad90", "inad1_30", "inad31_60", "inad61_90"):
        out[c] = ag[c].reindex(out.index).fillna(0.0)
    cart = out["carteira"].where(out["carteira"] > 0)
    out["vencido90_pct"] = out["inad90"] / cart
    out["atraso_1_60_pct"] = (out["inad1_30"] + out["inad31_60"]) / cart
    out["atraso_61_90_pct"] = out["inad61_90"] / cart
    out["pdd_pct"] = pdd.reindex(out.index).fillna(0.0) / cart
    out["recompra"] = vii.reindex(out.index).fillna(0.0)
    out["segmento"] = dom.reindex(out.index)
    out["em_estresse"] = (out["vencido90_pct"] > STRESS_INAD90).fillna(False).astype(bool)
    return out


def competencia_do_dir(d: str | Path) -> str:
    """Lê a competência (AAAA-MM) do nome do CSV da tabela I no diretório."""
    for p in sorted(Path(d).glob("inf_mensal_fidc_tab_I_*.csv")):
        m = _COMPETENCIA_RE.search(p.name)
        if m:
            return f"{m.group(1)}-{m.group(2)}"
    raise FileNotFoundError(f"nenhum inf_mensal_fidc_tab_I_AAAAMM.csv em {d}")


def month(d: str) -> dict[str, Any]:
    t1 = _read(d, "I")
    iv = _pl(d)
    ag = _aging(d)
    pdd = _pdd(t1)

    vii = _num(_read(d, "VII"), ["TAB_VII_D_2_VL_RECOMPRA", "TAB_VII_C_2_VL_SUBST"]).groupby("k")[["TAB_VII_D_2_VL_RECOMPRA", "TAB_VII_C_2_VL_SUBST"]].sum()

    seg = _num(_read(d, "II"), list(SEG)).groupby("k")[list(SEG)].sum()
    dom = seg.idxmax(axis=1).where(seg.sum(axis=1) > 0).map(SEG)

    x2 = _num(_read(d, "X_2"), ["TAB_X_QT_COTA", "TAB_X_VL_COTA"])
    x2["vl"] = x2["TAB_X_QT_COTA"] * x2["TAB_X_VL_COTA"]
    x2["sub"] = x2["TAB_X_CLASSE_SERIE"].fillna("").str.contains("Subordinada", case=False)
    # Por classe, descartando informes cujo valor das cotas destoa do PL (erros de preenchimento), ponderado pelo PL.
    pc = x2.groupby("k").apply(lambda g: pd.Series({"tot": g["vl"].sum(), "sub": g.loc[g["sub"], "vl"].sum()}), include_groups=False)
    pc["pl"] = iv.reindex(pc.index).fillna(0)
    ok = pc[(pc["pl"] > 0) & (pc["tot"] / pc["pl"]).between(0.5, 2)]
    sub_share = ((ok["sub"] / ok["tot"]) * ok["pl"]).sum() / ok["pl"].sum()

    x4 = _num(_read(d, "X_4"), ["TAB_X_VL_TOTAL"])
    ops = x4.groupby(x4["TAB_X_TP_OPER"].fillna("?"))["TAB_X_VL_TOTAL"].sum().to_dict()

    base = ag[ag["carteira"] > 0].copy()
    base["inad90_pct"] = base["inad90"] / base["carteira"]
    base["seg"] = dom.reindex(base.index)
    stressed = set(base.index[base["inad90_pct"] > STRESS_INAD90])

    seg_tab = (base.groupby("seg").agg(carteira=("carteira", "sum"), inad90=("inad90", "sum"), n=("carteira", "size"))
               .assign(inad90_pct=lambda s: s.inad90 / s.carteira).sort_values("carteira", ascending=False))

    # Contágio: cedentes PJ que aparecem em 2+ classes, e quantas classes sãs dividem cedente com uma classe sob estresse.
    ced = _cedentes(t1)[["k", "ced"]].drop_duplicates()
    per_ced = ced.groupby("ced")["k"].apply(set)
    shared = per_ced[per_ced.map(len) >= 2]
    hot_ced = shared[shared.map(lambda ks: bool(ks & stressed))]
    exposed = set().union(*hot_ced.tolist()) - stressed if len(hot_ced) else set()

    n_cls = int((iv > 0).sum())
    cart = base["carteira"].sum()
    return {
        "n_classes": n_cls,
        "pl_bi": iv.sum() / 1e9,
        "carteira_dc_bi": cart / 1e9,
        "inad30_pct": base["inad30"].sum() / cart * 100,
        "inad90_pct": base["inad90"].sum() / cart * 100,
        "inad180_pct": base["inad180"].sum() / cart * 100,
        "inad90_mediana_pct": float(base["inad90_pct"].median() * 100),
        "pdd_pct": pdd.reindex(base.index).fillna(0).sum() / cart * 100,
        "recompra_mi": vii["TAB_VII_D_2_VL_RECOMPRA"].sum() / 1e6,
        "recompra_pct": vii["TAB_VII_D_2_VL_RECOMPRA"].sum() / cart * 100,
        "subst_mi": vii["TAB_VII_C_2_VL_SUBST"].sum() / 1e6,
        "subord_pct": sub_share * 100,
        "classes_estresse": len(stressed),
        "classes_estresse_pct": len(stressed) / len(base) * 100,
        "ops_bi": {k: v / 1e9 for k, v in ops.items()},
        "cedentes_pj": int(len(per_ced)),
        "cedentes_compartilhados": int(len(shared)),
        "cedentes_ligados_estresse": int(len(hot_ced)),
        "classes_expostas": len(exposed),
        "pl_exposto_bi": iv.reindex(sorted(exposed)).fillna(0).sum() / 1e9,
        "segmentos": [
            {"seg": s, "carteira_bi": r.carteira / 1e9, "inad90_pct": r.inad90_pct * 100, "n": int(r.n)}
            for s, r in seg_tab.iterrows()
        ],
        "dist_inad90": base["inad90_pct"].clip(upper=1).mul(100).tolist(),
    }


def main(cur: str, prev: str, yoy: str, out: str) -> dict[str, dict[str, Any]]:
    """Calcula os indicadores do mês, do mês anterior e de 12 meses antes e grava o JSON."""
    res = {competencia_do_dir(d): month(d) for d in (cur, prev, yoy)}
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1)
    for lab, r in res.items():
        print(lab, {k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items() if k not in ("segmentos", "dist_inad90")})
    for s in next(iter(res.values()))["segmentos"]:
        print("  ", s)
    return res


if __name__ == "__main__":
    if len(sys.argv) != 5:
        sys.exit(__doc__)
    main(*sys.argv[1:5])

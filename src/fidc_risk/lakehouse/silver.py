"""Camada Silver: Harmonização multi-geração CVM (2018–2026), RCVM 175, BCB e DuckDB Lakehouse."""

from __future__ import annotations

import io
import json
import re
import time
import zipfile
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from fidc_risk.config import (
    BACEN_2682_PROVISION_WEIGHTS,
    BCB_SERIES,
    BRONZE_DIR,
    LAKEHOUSE_DB_PATH,
    REPORTS_DIR,
    SILVER_DIR,
    ensure_directories,
)
from fidc_risk.lakehouse.transforms import (
    classify_tranche_label,
    harmonize_percentage_share,
    normalize_doc_series,
    sanitize_monthly_return_pct,
    strip_accents_upper,
)


def _num(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(0.0, index=df.index, dtype="float64")
    return pd.to_numeric(df[col], errors="coerce").fillna(0.0).astype("float64")


def _sum_cols(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    acc = pd.Series(0.0, index=df.index, dtype="float64")
    for c in cols:
        if c in df.columns:
            acc = acc + pd.to_numeric(df[c], errors="coerce").fillna(0.0).astype("float64")
    return acc


def _extract_entity_id(df: pd.DataFrame) -> pd.Series:
    raw_col = "CNPJ_FUNDO_CLASSE" if "CNPJ_FUNDO_CLASSE" in df.columns else "CNPJ_FUNDO"
    primary = normalize_doc_series(df[raw_col])
    if "CNPJ_CLASSE" in df.columns:
        cls_doc = normalize_doc_series(df["CNPJ_CLASSE"])
        primary = np.where(cls_doc != "", cls_doc, primary)
        return pd.Series(primary, index=df.index, dtype="object")
    return primary


def _deduplicate_entity_table(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """
    Resolve duplicatas de transição RCVM 175 (quando 'Fundo' e 'Classe' aparecem juntos
    no mesmo CNPJ e competência), priorizando 'Classe' e maior ativo/carteira.
    """
    df = df.copy()
    df["entity_id"] = _extract_entity_id(df)
    df = df[df["entity_id"] != ""].copy()
    before = len(df)
    if before == 0:
        return df, 0

    priority = pd.Series(0, index=df.index)
    if "TP_FUNDO_CLASSE" in df.columns:
        tp = df["TP_FUNDO_CLASSE"].fillna("").astype(str).str.upper()
        priority = np.where(tp.str.contains("CLASSE"), 2, 1)
    df["_prio"] = priority

    sort_cols = ["entity_id", "DT_COMPTC", "_prio"]
    if "TAB_I_VL_ATIVO" in df.columns:
        df["_sort_val"] = _num(df, "TAB_I_VL_ATIVO")
        sort_cols.append("_sort_val")
    df = df.sort_values(sort_cols, ascending=[True, True, False] + ([False] if "_sort_val" in df.columns else []))
    df = df.drop_duplicates(subset=["entity_id", "DT_COMPTC"], keep="first")
    drop_aux = [c for c in ("_prio", "_sort_val") if c in df.columns]
    df = df.drop(columns=drop_aux)
    return df, before - len(df)


def build_macro_monthly() -> pd.DataFrame:
    """Consolida as 11 séries do Banco Central (SGS) em frequência mensal alinhada ao fim do mês."""
    frames: list[pd.DataFrame] = []
    for name, code in BCB_SERIES.items():
        fpath = BRONZE_DIR / "bcb_sgs" / f"{name}_{code}.json"
        payload = json.loads(fpath.read_text(encoding="utf-8"))
        sdf = pd.DataFrame(payload)
        sdf["dt"] = pd.to_datetime(sdf["data"], format="%d/%m/%Y", errors="coerce")
        sdf["val"] = pd.to_numeric(sdf["valor"], errors="coerce")
        sdf = sdf.dropna(subset=["dt", "val"]).sort_values("dt")
        sdf["ym"] = sdf["dt"].dt.to_period("M")
        # Para séries diárias (ex.: Selic Meta 432), pegamos o último valor vigente no fim do mês
        monthly = sdf.groupby("ym", as_index=False)["val"].last()
        monthly = monthly.rename(columns={"val": name})
        frames.append(monthly)

    macro = frames[0]
    for f in frames[1:]:
        macro = macro.merge(f, on="ym", how="outer")
    macro = macro.sort_values("ym").reset_index(drop=True)
    macro["dt_comptc"] = macro["ym"].dt.to_timestamp("M").dt.normalize()

    # Forward-fill limitado (séries trimestrais/defasadas do BCB como PNAD/Endividamento)
    num_cols = [c for c in BCB_SERIES if c in macro.columns]
    macro[num_cols] = macro[num_cols].ffill(limit=3)

    # Derivadas macroeconômicas causais (usando apenas janelas passadas <= t)
    cdi_fac = 1.0 + macro["cdi_mensal_pct"].fillna(0.0) / 100.0
    ipca_fac = 1.0 + macro["ipca_mensal_pct"].fillna(0.0) / 100.0
    macro["cdi_12m_pct"] = (cdi_fac.rolling(12, min_periods=6).apply(np.prod, raw=True) - 1.0) * 100.0
    macro["ipca_12m_pct"] = (ipca_fac.rolling(12, min_periods=6).apply(np.prod, raw=True) - 1.0) * 100.0
    macro["juro_real_expost_aa"] = (
        ((1.0 + macro["selic_meta_aa"] / 100.0) / (1.0 + macro["ipca_12m_pct"].fillna(4.0) / 100.0) - 1.0)
        * 100.0
    )
    macro["selic_delta_12m"] = macro["selic_meta_aa"] - macro["selic_meta_aa"].shift(12)
    macro["ibc_br_yoy_pct"] = macro["ibc_br"].pct_change(12) * 100.0
    macro["inad_sfn_pj_delta_6m"] = macro["inad_sfn_pj_pct"] - macro["inad_sfn_pj_pct"].shift(6)
    macro["inad_sfn_pf_delta_6m"] = macro["inad_sfn_pf_pct"] - macro["inad_sfn_pf_pct"].shift(6)

    macro = macro.drop(columns=["ym"])
    macro = macro[macro["dt_comptc"] >= pd.Timestamp("2017-01-01")].reset_index(drop=True)
    return macro


def build_fund_registry() -> pd.DataFrame:
    """
    Constrói cadastro unificado livre de viés de sobrevivência combinando:
    1) registro_fundo_classe.zip (RCVM 175: vínculo Fundo <-> Classe <-> Gestor/Custodiante)
    2) cad_fi.csv (histórico legado p/ fundos liquidados/cancelados antes da RCVM 175 + Taxa Adm)
    """
    rcvm_zip = BRONZE_DIR / "cvm_cad" / "registro_fundo_classe.zip"
    with zipfile.ZipFile(rcvm_zip) as zf:
        rf = pd.read_csv(zf.open("registro_fundo.csv"), sep=";", encoding="latin1", dtype=str)
        rc = pd.read_csv(zf.open("registro_classe.csv"), sep=";", encoding="latin1", dtype=str)

    rf_sub = rf[
        [
            "ID_Registro_Fundo",
            "CNPJ_Fundo",
            "Tipo_Fundo",
            "Data_Constituicao",
            "Data_Cancelamento",
            "Situacao",
            "CNPJ_Administrador",
            "Administrador",
            "CPF_CNPJ_Gestor",
            "Gestor",
        ]
    ].drop_duplicates(subset=["ID_Registro_Fundo"], keep="first")

    merged_175 = rc.merge(rf_sub, on="ID_Registro_Fundo", how="left", suffixes=("_classe", "_fundo"))
    merged_175["entity_id"] = normalize_doc_series(merged_175["CNPJ_Classe"])
    fallback_f = normalize_doc_series(merged_175["CNPJ_Fundo"])
    merged_175["entity_id"] = np.where(merged_175["entity_id"] != "", merged_175["entity_id"], fallback_f)
    merged_175 = merged_175[merged_175["entity_id"] != ""].copy()

    reg_175 = pd.DataFrame(
        {
            "entity_id": merged_175["entity_id"],
            "denom_social_reg": merged_175["Denominacao_Social"].fillna(""),
            "dt_constituicao": pd.to_datetime(
                merged_175["Data_Constituicao_classe"].fillna(merged_175["Data_Constituicao_fundo"]),
                errors="coerce",
            ),
            "dt_cancelamento": pd.to_datetime(merged_175["Data_Cancelamento"], errors="coerce"),
            "situacao_reg": merged_175["Situacao_classe"].fillna(merged_175["Situacao_fundo"]).fillna(""),
            "condominio_reg": merged_175["Forma_Condominio"].fillna(""),
            "exclusivo_reg": merged_175["Exclusivo"].fillna(""),
            "publico_alvo_reg": merged_175["Publico_Alvo"].fillna(""),
            "cnpj_gestor": normalize_doc_series(merged_175["CPF_CNPJ_Gestor"]),
            "nome_gestor": merged_175["Gestor"].fillna(""),
            "cnpj_admin_reg": normalize_doc_series(merged_175["CNPJ_Administrador"]),
            "nome_admin_reg": merged_175["Administrador"].fillna(""),
            "cnpj_custodiante": normalize_doc_series(merged_175["CNPJ_Custodiante"]),
            "nome_custodiante": merged_175["Custodiante"].fillna(""),
            "cnpj_auditor": normalize_doc_series(merged_175["CNPJ_Auditor"]),
            "taxa_adm_aa": np.nan,
            "reg_source": "RCVM175",
        }
    )

    # Também inclui registros de nível fundo puro da registro_fundo (caso não tenha CNPJ_Classe separado)
    rf_only = pd.DataFrame(
        {
            "entity_id": normalize_doc_series(rf["CNPJ_Fundo"]),
            "denom_social_reg": rf["Denominacao_Social"].fillna(""),
            "dt_constituicao": pd.to_datetime(rf["Data_Constituicao"], errors="coerce"),
            "dt_cancelamento": pd.to_datetime(rf["Data_Cancelamento"], errors="coerce"),
            "situacao_reg": rf["Situacao"].fillna(""),
            "condominio_reg": "",
            "exclusivo_reg": "",
            "publico_alvo_reg": "",
            "cnpj_gestor": normalize_doc_series(rf["CPF_CNPJ_Gestor"]),
            "nome_gestor": rf["Gestor"].fillna(""),
            "cnpj_admin_reg": normalize_doc_series(rf["CNPJ_Administrador"]),
            "nome_admin_reg": rf["Administrador"].fillna(""),
            "cnpj_custodiante": "",
            "nome_custodiante": "",
            "cnpj_auditor": "",
            "taxa_adm_aa": np.nan,
            "reg_source": "RCVM175_FUNDO",
        }
    )
    rf_only = rf_only[rf_only["entity_id"] != ""]

    cad = pd.read_csv(BRONZE_DIR / "cvm_cad" / "cad_fi.csv", sep=";", encoding="latin1", dtype=str)
    cad_df = pd.DataFrame(
        {
            "entity_id": normalize_doc_series(cad["CNPJ_FUNDO"]),
            "denom_social_reg": cad["DENOM_SOCIAL"].fillna(""),
            "dt_constituicao": pd.to_datetime(cad["DT_CONST"], errors="coerce"),
            "dt_cancelamento": pd.to_datetime(cad["DT_CANCEL"], errors="coerce"),
            "situacao_reg": cad["SIT"].fillna(""),
            "condominio_reg": cad["CONDOM"].fillna(""),
            "exclusivo_reg": cad["FUNDO_EXCLUSIVO"].fillna(""),
            "publico_alvo_reg": cad["PUBLICO_ALVO"].fillna(""),
            "cnpj_gestor": normalize_doc_series(cad["CPF_CNPJ_GESTOR"]),
            "nome_gestor": cad["GESTOR"].fillna(""),
            "cnpj_admin_reg": normalize_doc_series(cad["CNPJ_ADMIN"]),
            "nome_admin_reg": cad["ADMIN"].fillna(""),
            "cnpj_custodiante": normalize_doc_series(cad["CNPJ_CUSTODIANTE"]),
            "nome_custodiante": cad["CUSTODIANTE"].fillna(""),
            "cnpj_auditor": normalize_doc_series(cad["CNPJ_AUDITOR"]),
            "taxa_adm_aa": pd.to_numeric(cad["TAXA_ADM"], errors="coerce"),
            "reg_source": "CAD_LEGACY",
        }
    )
    cad_df = cad_df[cad_df["entity_id"] != ""].drop_duplicates(subset=["entity_id"], keep="first")

    combined = pd.concat([reg_175, rf_only, cad_df], ignore_index=True)
    combined = combined.drop_duplicates(subset=["entity_id"], keep="first").reset_index(drop=True)

    # Enriquecer taxa_adm_aa e custodiante/público-alvo faltantes a partir do legado quando disponível
    legacy_lookup = cad_df.set_index("entity_id")
    for col in ("taxa_adm_aa", "condominio_reg", "publico_alvo_reg", "cnpj_custodiante", "nome_custodiante"):
        missing = combined[col].isna() | (combined[col].astype(str) == "")
        mapped = combined.loc[missing, "entity_id"].map(legacy_lookup[col])
        combined.loc[missing, col] = mapped

    return combined


def _parse_single_month_tables(
    ym: str,
    tables: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    """
    Processa todas as tabelas de uma única competência YYYYMM em:
    - 1 DataFrame de estado mensal por entity_id
    - 1 DataFrame de arestas de cedentes (entity_id, dt_comptc, cedente_doc, share_pct, risk_mode, is_pj)
    """
    stats = {"rcvm175_dups_removed": 0}
    t1_raw = tables["tab_I"]
    t1, dups = _deduplicate_entity_table(t1_raw)
    stats["rcvm175_dups_removed"] += dups
    dt_val = pd.to_datetime(f"{ym[:4]}-{ym[4:6]}-01") + pd.offsets.MonthEnd(0)

    idx = t1.index

    # --- Extração de Arestas de Cedentes (Grafo Dinâmico FIDC <-> Cedente) ---
    edge_chunks: list[pd.DataFrame] = []
    # Geração 2020–2026: TAB_I2A12 (com risco) e TAB_I2B12 (sem risco)
    # Geração 2018–2019: TAB_I2B1 (cedentes gerais)
    prefixes = [
        ("TAB_I2A12", "COM_RISCO"),
        ("TAB_I2B12", "SEM_RISCO"),
        ("TAB_I2B1", "COM_RISCO"),
    ]
    for pref, mode in prefixes:
        for k in range(1, 10):
            doc_col = f"{pref}_CPF_CNPJ_CEDENTE_{k}"
            pr_col = f"{pref}_PR_CEDENTE_{k}"
            if doc_col in t1.columns and pr_col in t1.columns:
                docs = normalize_doc_series(t1[doc_col])
                shares = harmonize_percentage_share(t1[pr_col], max_cap=100.0)
                valid = (docs != "") & (shares > 0.0)
                if valid.any():
                    edge_chunks.append(
                        pd.DataFrame(
                            {
                                "dt_comptc": dt_val,
                                "entity_id": t1.loc[valid, "entity_id"].values,
                                "cedente_doc": docs.loc[valid].values,
                                "share_pct": shares.loc[valid].values,
                                "risk_mode": mode,
                                "is_pj": (docs.loc[valid].str.len() == 14).astype("int8").values,
                            }
                        )
                    )

    if edge_chunks:
        edges_df = pd.concat(edge_chunks, ignore_index=True)
        # Se o mesmo cedente aparece em COM_RISCO e SEM_RISCO no mesmo fundo/mês, soma participação limitada a 100%
        edges_df = (
            edges_df.groupby(["dt_comptc", "entity_id", "cedente_doc", "is_pj"], as_index=False)["share_pct"]
            .sum()
        )
        edges_df["share_pct"] = edges_df["share_pct"].clip(upper=100.0)
    else:
        edges_df = pd.DataFrame(columns=["dt_comptc", "entity_id", "cedente_doc", "is_pj", "share_pct"])

    # Métricas resumidas de concentração de cedentes por fundo no mês
    if len(edges_df) > 0:
        edges_sorted = edges_df.sort_values(["entity_id", "share_pct"], ascending=[True, False])
        ced_agg = edges_sorted.groupby("entity_id").agg(
            n_cedentes_disclosed=("cedente_doc", "count"),
            n_cedentes_pj=("is_pj", "sum"),
            top1_cedente_pct=("share_pct", "first"),
            top3_cedente_pct=("share_pct", lambda s: float(s.iloc[:3].sum())),
            top9_cedente_pct=("share_pct", "sum"),
            cedente_hhi=("share_pct", lambda s: float(np.sum((s / 100.0) ** 2))),
        )
    else:
        ced_agg = pd.DataFrame(
            columns=[
                "n_cedentes_disclosed",
                "n_cedentes_pj",
                "top1_cedente_pct",
                "top3_cedente_pct",
                "top9_cedente_pct",
                "cedente_hhi",
            ]
        )

    # Prazos de conversão e pagamento de resgate em dias corridos aproximados
    prazo_conv = _num(t1, "PRAZO_CONVERSAO_COTA").clip(lower=0.0, upper=1800.0)
    prazo_pag = _num(t1, "PRAZO_PAGTO_RESGATE").clip(lower=0.0, upper=1800.0)
    if "TP_PRAZO_CONVERSAO_COTA" in t1.columns:
        is_du = t1["TP_PRAZO_CONVERSAO_COTA"].fillna("").astype(str).str.upper().str.contains("UTE|ÚTE|DU")
        prazo_conv = np.where(is_du, prazo_conv * (30.0 / 21.0), prazo_conv)
    if "TP_PRAZO_PAGTO_RESGATE" in t1.columns:
        is_du_p = t1["TP_PRAZO_PAGTO_RESGATE"].fillna("").astype(str).str.upper().str.contains("UTE|ÚTE|DU")
        prazo_pag = np.where(is_du_p, prazo_pag * (30.0 / 21.0), prazo_pag)

    condom = (
        t1["CONDOM"].fillna("").astype(str).map(strip_accents_upper)
        if "CONDOM" in t1.columns
        else pd.Series("", index=idx)
    )
    exclusivo = (
        t1["FUNDO_EXCLUSIVO"].fillna("N").astype(str).str.upper().str.startswith("S").astype("int8")
        if "FUNDO_EXCLUSIVO" in t1.columns
        else pd.Series(0, index=idx, dtype="int8")
    )
    cotst_interesse = (
        t1["COTST_INTERESSE"].fillna("N").astype(str).str.upper().str.startswith("S").astype("int8")
        if "COTST_INTERESSE" in t1.columns
        else pd.Series(0, index=idx, dtype="int8")
    )
    classe_unica_flag = (
        t1["CLASSE_UNICA"].fillna("N").astype(str).str.upper().str.startswith("S").astype("int8")
        if "CLASSE_UNICA" in t1.columns
        else pd.Series(0, index=idx, dtype="int8")
    )

    core = pd.DataFrame(
        {
            "entity_id": t1["entity_id"].values,
            "cnpj_raw": (
                t1["CNPJ_FUNDO_CLASSE"].fillna("")
                if "CNPJ_FUNDO_CLASSE" in t1.columns
                else t1["CNPJ_FUNDO"].fillna("")
            ).values,
            "denom_social": t1["DENOM_SOCIAL"].fillna("").astype(str).values,
            "dt_comptc": dt_val,
            "cnpj_admin": normalize_doc_series(t1["CNPJ_ADMIN"]).values if "CNPJ_ADMIN" in t1.columns else "",
            "nome_admin": t1["ADMIN"].fillna("").astype(str).values if "ADMIN" in t1.columns else "",
            "condominio": condom.values,
            "is_aberto": condom.str.contains("ABERT").astype("int8").values,
            "is_exclusivo": exclusivo.values,
            "has_cotista_interesse": cotst_interesse.values,
            "is_classe_unica_flag": classe_unica_flag.values,
            "prazo_conversao_dias": np.asarray(prazo_conv, dtype="float64"),
            "prazo_pagto_resgate_dias": np.asarray(prazo_pag, dtype="float64"),
            "vl_ativo": _num(t1, "TAB_I_VL_ATIVO").values,
            "vl_disp": _num(t1, "TAB_I1_VL_DISP").values,
            "vl_carteira": _num(t1, "TAB_I2_VL_CARTEIRA").values,
            "vl_dircred_risco": _num(t1, "TAB_I2A_VL_DIRCRED_RISCO").values,
            "vl_dircred_sem_risco": _num(t1, "TAB_I2B_VL_DIRCRED_SEM_RISCO").values,
            "vl_cred_venc_ad": _sum_cols(t1, ["TAB_I2A1_VL_CRED_VENC_AD", "TAB_I2B1_VL_CRED_VENC_AD"]).values,
            "vl_cred_venc_inad": _sum_cols(t1, ["TAB_I2A2_VL_CRED_VENC_INAD", "TAB_I2B2_VL_CRED_VENC_INAD"]).values,
            "vl_cred_inad": _sum_cols(t1, ["TAB_I2A3_VL_CRED_INAD", "TAB_I2B3_VL_CRED_INAD"]).values,
            "vl_cred_perform": _sum_cols(t1, ["TAB_I2A4_VL_CRED_DIRCRED_PERFM", "TAB_I2B4_VL_CRED_DIRCRED_PERFM"]).values,
            "vl_cred_emp_recup": _sum_cols(t1, ["TAB_I2A6_VL_CRED_EMP_RECUP", "TAB_I2B6_VL_CRED_EMP_RECUP"]).values,
            "vl_cred_acao_judic": _sum_cols(t1, ["TAB_I2A8_VL_CRED_ACAO_JUDIC", "TAB_I2B8_VL_CRED_ACAO_JUDIC"]).values,
            "vl_pdd_total": _sum_cols(t1, ["TAB_I2A11_VL_REDUCAO_RECUP", "TAB_I2B11_VL_REDUCAO_RECUP"]).values,
            "vl_titpub_liq": _sum_cols(
                t1, ["TAB_I2D_VL_TITPUB_FED", "TAB_I2E_VL_CDB", "TAB_I2F_VL_OPER_COMPROM"]
            ).values,
            "vl_cota_fidc": _sum_cols(t1, ["TAB_I2H_VL_COTA_FIDC", "TAB_I2_COTA_FIDC"]).values,
            "vl_cota_fidc_np": _num(t1, "TAB_I2I_VL_COTA_FIDC_NP").values,
        }
    )

    core = core.merge(ced_agg, on="entity_id", how="left")
    for c in ("n_cedentes_disclosed", "n_cedentes_pj", "top1_cedente_pct", "top3_cedente_pct", "top9_cedente_pct", "cedente_hhi"):
        core[c] = core[c].fillna(0.0)

    # --- Tabela II: Segmentos Setoriais ---
    if "tab_II" in tables:
        t2, _ = _deduplicate_entity_table(tables["tab_II"])
        t2_df = pd.DataFrame(
            {
                "entity_id": t2["entity_id"].values,
                "vl_carteira_tab2": _num(t2, "TAB_II_VL_CARTEIRA").values,
                "sec_industrial": _num(t2, "TAB_II_A_VL_INDUST").values,
                "sec_imobiliario": _sum_cols(
                    t2, ["TAB_II_B_VL_IMOBIL", "TAB_II_F6_VL_IMOBIL_EMPRESA", "TAB_II_F7_VL_IMOBIL_RESID"]
                ).values,
                "sec_comercial_varejo": _num(t2, "TAB_II_C_VL_COMERC").values,
                "sec_servicos": _num(t2, "TAB_II_D_VL_SERV").values,
                "sec_agronegocio": _num(t2, "TAB_II_E_VL_AGRONEG").values,
                "sec_cred_pessoal": _num(t2, "TAB_II_F1_VL_CRED_PESSOA").values,
                "sec_consignado": _num(t2, "TAB_II_F2_VL_CRED_PESSOA_CONSIG").values,
                "sec_corporativo": _sum_cols(t2, ["TAB_II_F3_VL_CRED_CORP", "TAB_II_F4_VL_MIDMARKET"]).values,
                "sec_veiculos": _num(t2, "TAB_II_F5_VL_VEICULO").values,
                "sec_cartoes_credito": _num(t2, "TAB_II_G_VL_CREDITO").values,
                "sec_factoring": _num(t2, "TAB_II_H_VL_FACTOR").values,
                "sec_setor_publico": _num(t2, "TAB_II_I_VL_SETOR_PUBLICO").values,
                "sec_judicial": _num(t2, "TAB_II_J_VL_JUDICIAL").values,
            }
        )
        core = core.merge(t2_df, on="entity_id", how="left")

    # --- Tabelas III e IV: Passivo, Derivativos e Patrimônio Líquido ---
    if "tab_III" in tables:
        t3, _ = _deduplicate_entity_table(tables["tab_III"])
        t3_df = pd.DataFrame(
            {
                "entity_id": t3["entity_id"].values,
                "vl_passivo": _num(t3, "TAB_III_VL_PASSIVO").values,
                "vl_derivativos_passivo": _num(t3, "TAB_III_B_VL_POSICAO_DERIV").values,
            }
        )
        core = core.merge(t3_df, on="entity_id", how="left")

    if "tab_IV" in tables:
        t4, _ = _deduplicate_entity_table(tables["tab_IV"])
        t4_df = pd.DataFrame(
            {
                "entity_id": t4["entity_id"].values,
                "vl_pl": _num(t4, "TAB_IV_A_VL_PL").values,
                "vl_pl_medio": _num(t4, "TAB_IV_B_VL_PL_MEDIO").values,
            }
        )
        core = core.merge(t4_df, on="entity_id", how="left")

    # --- Tabelas V e VI: Vencimentos e 10 Faixas de Inadimplência (Com + Sem Risco) ---
    aging_buckets = [
        ("30", "1"),
        ("60", "2"),
        ("90", "3"),
        ("120", "4"),
        ("150", "5"),
        ("180", "6"),
        ("360", "7"),
        ("720", "8"),
        ("1080", "9"),
        ("MAIOR_1080", "10"),
    ]
    t5 = _deduplicate_entity_table(tables["tab_V"])[0].set_index("entity_id") if "tab_V" in tables else pd.DataFrame()
    t6 = _deduplicate_entity_table(tables["tab_VI"])[0].set_index("entity_id") if "tab_VI" in tables else pd.DataFrame()

    core_idx = core["entity_id"]
    tot_prazo = pd.Series(0.0, index=core.index)
    tot_inad = pd.Series(0.0, index=core.index)
    if len(t5) > 0:
        tot_prazo = tot_prazo + core_idx.map(_num(t5, "TAB_V_A_VL_DIRCRED_PRAZO")).fillna(0.0)
        tot_inad = tot_inad + core_idx.map(_num(t5, "TAB_V_B_VL_DIRCRED_INAD")).fillna(0.0)
    if len(t6) > 0:
        tot_prazo = tot_prazo + core_idx.map(_num(t6, "TAB_VI_A_VL_DIRCRED_PRAZO")).fillna(0.0)
        tot_inad = tot_inad + core_idx.map(_num(t6, "TAB_VI_B_VL_DIRCRED_INAD")).fillna(0.0)

    core["vl_dircred_prazo_total"] = tot_prazo
    core["vl_dircred_inad_total"] = tot_inad

    for suffix, idx_num in aging_buckets:
        col_p5 = f"TAB_V_A{idx_num}_VL_PRAZO_VENC_{suffix}"
        col_p6 = f"TAB_VI_A{idx_num}_VL_PRAZO_VENC_{suffix}"
        col_i5 = f"TAB_V_B{idx_num}_VL_INAD_{suffix}"
        col_i6 = f"TAB_VI_B{idx_num}_VL_INAD_{suffix}"

        vp = pd.Series(0.0, index=core.index)
        vi = pd.Series(0.0, index=core.index)
        if len(t5) > 0:
            vp = vp + core_idx.map(_num(t5, col_p5)).fillna(0.0)
            vi = vi + core_idx.map(_num(t5, col_i5)).fillna(0.0)
        if len(t6) > 0:
            vp = vp + core_idx.map(_num(t6, col_p6)).fillna(0.0)
            vi = vi + core_idx.map(_num(t6, col_i6)).fillna(0.0)

        clean_suf = suffix.lower()
        core[f"prazo_venc_{clean_suf}"] = vp.clip(lower=0.0)
        core[f"inad_venc_{clean_suf}"] = vi.clip(lower=0.0)

    # --- Tabela VII: Aquisições, Substituições e Recompras (Sinais Forenses!) ---
    if "tab_VII" in tables:
        t7, _ = _deduplicate_entity_table(tables["tab_VII"])
        t7_df = pd.DataFrame(
            {
                "entity_id": t7["entity_id"].values,
                "vl_aquisicoes_mes": _sum_cols(
                    t7, ["TAB_VII_A1_2_VL_DIRCRED_RISCO", "TAB_VII_A2_2_VL_DIRCRED_SEM_RISCO"]
                ).values,
                "vl_alienacoes_cedente": _num(t7, "TAB_VII_B1_2_VL_CEDENTE").values,
                "vl_alienacoes_terceiros": _num(t7, "TAB_VII_B3_2_VL_TERCEIRO").values,
                "qt_substituicoes_mes": _num(t7, "TAB_VII_C_1_QT_SUBST").values,
                "vl_substituicoes_mes": _num(t7, "TAB_VII_C_2_VL_SUBST").values,
                "qt_recompras_mes": _num(t7, "TAB_VII_D_1_QT_RECOMPRA").values,
                "vl_recompras_mes": _num(t7, "TAB_VII_D_2_VL_RECOMPRA").values,
            }
        )
        core = core.merge(t7_df, on="entity_id", how="left")

    # --- Tabela IX: Taxas Médias de Desconto na Compra de Recebíveis ---
    if "tab_IX" in tables:
        t9, _ = _deduplicate_entity_table(tables["tab_IX"])
        media_cols = [c for c in t9.columns if c.endswith("_COMPRA_MEDIA")]
        if media_cols:
            vals = t9[media_cols].apply(pd.to_numeric, errors="coerce")
            vals = vals.where((vals > 0.0) & (vals < 300.0), np.nan)
            taxa_med = vals.mean(axis=1)
        else:
            taxa_med = pd.Series(np.nan, index=t9.index)
        t9_df = pd.DataFrame({"entity_id": t9["entity_id"].values, "taxa_desconto_compra_media": taxa_med.values})
        core = core.merge(t9_df, on="entity_id", how="left")
    else:
        core["taxa_desconto_compra_media"] = np.nan

    # --- Tabela X (SCR Risco AA..H do Banco Central — disponível a partir de out/2023) ---
    if "tab_X" in tables:
        tx, _ = _deduplicate_entity_table(tables["tab_X"])
        scr_tot = pd.Series(0.0, index=tx.index)
        scr_prov_min = pd.Series(0.0, index=tx.index)
        scr_efgh = pd.Series(0.0, index=tx.index)
        for rating, weight in BACEN_2682_PROVISION_WEIGHTS.items():
            # Usa risco da operação se preenchido; caso contrário usa risco do devedor
            op_val = _num(tx, f"TAB_X_SCR_RISCO_OPER_{rating}").clip(lower=0.0)
            dev_val = _num(tx, f"TAB_X_SCR_RISCO_DEVEDOR_{rating}").clip(lower=0.0)
            bucket_val = np.where(op_val > 0, op_val, dev_val)
            scr_tot = scr_tot + bucket_val
            scr_prov_min = scr_prov_min + bucket_val * weight
            if rating in ("E", "F", "G", "H"):
                scr_efgh = scr_efgh + bucket_val

        tx_df = pd.DataFrame(
            {
                "entity_id": tx["entity_id"].values,
                "has_scr_report": (scr_tot > 0.0).astype("int8").values,
                "vl_scr_total": scr_tot.values,
                "vl_scr_efgh": scr_efgh.values,
                "vl_scr_pdd_min_bacen2682": scr_prov_min.values,
            }
        )
        core = core.merge(tx_df, on="entity_id", how="left")
    else:
        core["has_scr_report"] = np.int8(0)
        core["vl_scr_total"] = 0.0
        core["vl_scr_efgh"] = 0.0
        core["vl_scr_pdd_min_bacen2682"] = 0.0

    # --- Tabelas X_1, X_2, X_3, X_4, X_6: Waterfall de Tranches (Sênior, Mezanino, Subordinada) ---
    tranche_pl_lookup: dict[tuple[str, str], float] = {}
    if "tab_X_2" in tables:
        tx2 = tables["tab_X_2"].copy()
        tx2["entity_id"] = _extract_entity_id(tx2)
        tx2 = tx2[tx2["entity_id"] != ""]
        tx2["tranche"] = tx2["TAB_X_CLASSE_SERIE"].map(classify_tranche_label)
        tx2["serie_key"] = tx2["TAB_X_CLASSE_SERIE"].fillna("").astype(str).str.strip()
        tx2["pl_serie"] = (_num(tx2, "TAB_X_QT_COTA") * _num(tx2, "TAB_X_VL_COTA")).clip(lower=0.0)

        for row in tx2[["entity_id", "serie_key", "pl_serie"]].itertuples(index=False):
            tranche_pl_lookup[(row.entity_id, row.serie_key)] = (
                tranche_pl_lookup.get((row.entity_id, row.serie_key), 0.0) + float(row.pl_serie)
            )

        piv_pl = tx2.pivot_table(index="entity_id", columns="tranche", values="pl_serie", aggfunc="sum", fill_value=0.0)
        for tr_col, out_col in (
            ("SENIOR", "pl_senior"),
            ("MEZANINO", "pl_mezanino"),
            ("SUBORDINADA", "pl_subordinada"),
            ("UNICA", "pl_unica"),
        ):
            core[out_col] = core["entity_id"].map(piv_pl[tr_col]).fillna(0.0) if tr_col in piv_pl.columns else 0.0
        sr_cnt = tx2[(tx2["tranche"] == "SENIOR") & (tx2["pl_serie"] > 0)].groupby("entity_id")["serie_key"].nunique()
        core["n_series_senior"] = core["entity_id"].map(sr_cnt).fillna(0).astype("int16")
    else:
        for out_col in ("pl_senior", "pl_mezanino", "pl_subordinada", "pl_unica"):
            core[out_col] = 0.0
        core["n_series_senior"] = np.int16(0)

    if "tab_X_3" in tables:
        tx3 = tables["tab_X_3"].copy()
        tx3["entity_id"] = _extract_entity_id(tx3)
        tx3 = tx3[tx3["entity_id"] != ""]
        tx3["tranche"] = tx3["TAB_X_CLASSE_SERIE"].map(classify_tranche_label)
        tx3["serie_key"] = tx3["TAB_X_CLASSE_SERIE"].fillna("").astype(str).str.strip()
        tx3["rentab"] = sanitize_monthly_return_pct(tx3["TAB_X_VL_RENTAB_MES"])
        tx3 = tx3.dropna(subset=["rentab"])
        if len(tx3) > 0:
            tx3["w"] = [
                max(tranche_pl_lookup.get((eid, sk), 0.0), 1.0)
                for eid, sk in zip(tx3["entity_id"], tx3["serie_key"], strict=False)
            ]
            tx3["wr"] = tx3["w"] * tx3["rentab"]
            grp = tx3.groupby(["entity_id", "tranche"], as_index=False).agg(w_sum=("w", "sum"), wr_sum=("wr", "sum"))
            grp["w_rentab"] = grp["wr_sum"] / grp["w_sum"]
            piv_r = grp.pivot(index="entity_id", columns="tranche", values="w_rentab")
            for tr_col, out_col in (
                ("SENIOR", "rentab_senior_mes"),
                ("MEZANINO", "rentab_mezanino_mes"),
                ("SUBORDINADA", "rentab_subordinada_mes"),
                ("UNICA", "rentab_unica_mes"),
            ):
                core[out_col] = core["entity_id"].map(piv_r[tr_col]) if tr_col in piv_r.columns else np.nan
        else:
            for out_col in ("rentab_senior_mes", "rentab_mezanino_mes", "rentab_subordinada_mes", "rentab_unica_mes"):
                core[out_col] = np.nan

    if "tab_X_4" in tables:
        tx4 = tables["tab_X_4"].copy()
        tx4["entity_id"] = _extract_entity_id(tx4)
        tx4 = tx4[tx4["entity_id"] != ""]
        tx4["tranche"] = tx4["TAB_X_CLASSE_SERIE"].map(classify_tranche_label)
        tx4["op"] = tx4["TAB_X_TP_OPER"].fillna("").astype(str).map(strip_accents_upper)
        tx4["vl"] = _num(tx4, "TAB_X_VL_TOTAL").clip(lower=0.0)

        def _map_op(mask: pd.Series) -> pd.Series:
            if not mask.any():
                return pd.Series(0.0, index=core.index)
            s = tx4.loc[mask].groupby("entity_id")["vl"].sum()
            return core["entity_id"].map(s).fillna(0.0)

        is_capt = tx4["op"].str.contains("CAPTAC")
        is_amort = tx4["op"].str.contains("AMORTIZ")
        is_resg_solic = tx4["op"].str.contains("SOLICIT")
        is_resg_pago = tx4["op"].str.contains("RESGAT") & (~is_resg_solic)
        is_sr = tx4["tranche"] == "SENIOR"

        core["captacoes_total"] = _map_op(is_capt)
        core["amortizacoes_total"] = _map_op(is_amort)
        core["resgates_pagos_total"] = _map_op(is_resg_pago)
        core["resgates_solicitados_total"] = _map_op(is_resg_solic)
        core["captacoes_senior"] = _map_op(is_capt & is_sr)
        core["resgates_pagos_senior"] = _map_op(is_resg_pago & is_sr)
        core["resgates_solicitados_senior"] = _map_op(is_resg_solic & is_sr)

    if "tab_X_1" in tables:
        tx1 = tables["tab_X_1"].copy()
        tx1["entity_id"] = _extract_entity_id(tx1)
        tx1 = tx1[tx1["entity_id"] != ""]
        tx1["tranche"] = tx1["TAB_X_CLASSE_SERIE"].map(classify_tranche_label)
        tx1["nr"] = _num(tx1, "TAB_X_NR_COTST").clip(lower=0.0)
        piv_c = tx1.pivot_table(index="entity_id", columns="tranche", values="nr", aggfunc="sum", fill_value=0.0)
        core["nr_cotistas_senior"] = (
            core["entity_id"].map(piv_c["SENIOR"]).fillna(0.0) if "SENIOR" in piv_c.columns else 0.0
        )
        core["nr_cotistas_subord"] = (
            core["entity_id"].map(piv_c["SUBORDINADA"]).fillna(0.0) if "SUBORDINADA" in piv_c.columns else 0.0
        )
        core["nr_cotistas_total"] = core["entity_id"].map(tx1.groupby("entity_id")["nr"].sum()).fillna(0.0)

    if "tab_X_5" in tables:
        tx5, _ = _deduplicate_entity_table(tables["tab_X_5"])
        tx5_df = pd.DataFrame(
            {
                "entity_id": tx5["entity_id"].values,
                "liq_imediata_0": _num(tx5, "TAB_X_VL_LIQUIDEZ_0").values,
                "liq_30d": _num(tx5, "TAB_X_VL_LIQUIDEZ_30").values,
                "liq_90d": _sum_cols(tx5, ["TAB_X_VL_LIQUIDEZ_60", "TAB_X_VL_LIQUIDEZ_90"]).values,
                "liq_180_360d": _sum_cols(tx5, ["TAB_X_VL_LIQUIDEZ_180", "TAB_X_VL_LIQUIDEZ_360"]).values,
                "liq_maior_360d": _num(tx5, "TAB_X_VL_LIQUIDEZ_MAIOR_360").values,
            }
        )
        core = core.merge(tx5_df, on="entity_id", how="left")

    if "tab_X_6" in tables:
        tx6 = tables["tab_X_6"].copy()
        tx6["entity_id"] = _extract_entity_id(tx6)
        tx6 = tx6[tx6["entity_id"] != ""]
        tx6["tranche"] = tx6["TAB_X_CLASSE_SERIE"].map(classify_tranche_label)
        tx6_sr = tx6[tx6["tranche"] == "SENIOR"].copy()
        if len(tx6_sr) > 0:
            tx6_sr["esp"] = sanitize_monthly_return_pct(tx6_sr["TAB_X_PR_DESEMP_ESPERADO"])
            tx6_sr["real"] = sanitize_monthly_return_pct(tx6_sr["TAB_X_PR_DESEMP_REAL"])
            g6 = tx6_sr.groupby("entity_id")[["esp", "real"]].mean()
            core["desemp_esperado_senior"] = core["entity_id"].map(g6["esp"])
            core["desemp_real_senior"] = core["entity_id"].map(g6["real"])
        else:
            core["desemp_esperado_senior"] = np.nan
            core["desemp_real_senior"] = np.nan

    if "tab_X_7" in tables:
        tx7, _ = _deduplicate_entity_table(tables["tab_X_7"])
        tx7_df = pd.DataFrame(
            {
                "entity_id": tx7["entity_id"].values,
                "vl_garantia_dircred": _num(tx7, "TAB_X_VL_GARANTIA_DIRCRED").values,
                "pr_garantia_dircred": harmonize_percentage_share(tx7["TAB_X_PR_GARANTIA_DIRCRED"], max_cap=200.0).values,
            }
        )
        core = core.merge(tx7_df, on="entity_id", how="left")
    else:
        core["vl_garantia_dircred"] = 0.0
        core["pr_garantia_dircred"] = 0.0

    # Preencher zeros p/ colunas de valor monetário/fluxo (mantendo retornos mensais como NaN quando ausentes)
    nullable_cols = {
        "rentab_senior_mes",
        "rentab_mezanino_mes",
        "rentab_subordinada_mes",
        "rentab_unica_mes",
        "desemp_esperado_senior",
        "desemp_real_senior",
        "taxa_desconto_compra_media",
    }
    for col in core.columns:
        if col not in nullable_cols and pd.api.types.is_numeric_dtype(core[col]):
            core[col] = core[col].fillna(0.0)

    return core, edges_df, stats


def _iter_cvm_zip_months() -> list[tuple[str, Path]]:
    """Lista ordenada de (YYYYMM, zip_path) cobrindo 201801 até a competência mais recente."""
    month_map: dict[str, Path] = {}
    for zpath in sorted((BRONZE_DIR / "cvm_fidc_hist").glob("inf_mensal_fidc_*.zip")):
        with zipfile.ZipFile(zpath) as zf:
            yms = sorted({m.group(1) for name in zf.namelist() if (m := re.search(r"_(\d{6})\.csv$", name))})
            for ym in yms:
                month_map[ym] = zpath
    for zpath in sorted((BRONZE_DIR / "cvm_fidc_monthly").glob("inf_mensal_fidc_*.zip")):
        m = re.search(r"_(\d{6})\.zip$", zpath.name)
        if m:
            month_map[m.group(1)] = zpath
    return sorted(month_map.items(), key=lambda x: x[0])


def build_silver_layer() -> dict[str, object]:
    """Executa a transformação completa Bronze -> Silver e materializa tabelas no DuckDB + Parquet."""
    ensure_directories()
    t0 = time.perf_counter()

    print("[Silver] 1/4 Consolidando séries macroeconômicas e de crédito do Banco Central...")
    macro_df = build_macro_monthly()
    macro_path = SILVER_DIR / "macro_monthly.parquet"
    macro_df.to_parquet(macro_path, index=False)

    print("[Silver] 2/4 Reconciliando cadastro estrutural CVM (RCVM 175 + Legado sem viés de sobrevivência)...")
    registry_df = build_fund_registry()
    registry_path = SILVER_DIR / "fund_registry.parquet"
    registry_df.to_parquet(registry_path, index=False)

    month_sources = _iter_cvm_zip_months()
    print(f"[Silver] 3/4 Processando {len(month_sources)} competências mensais ({month_sources[0][0]} .. {month_sources[-1][0]})...")

    core_chunks: list[pd.DataFrame] = []
    edge_chunks: list[pd.DataFrame] = []
    total_dups_removed = 0

    # Agrupar leitura por arquivo ZIP para abrir cada ZIP apenas uma vez
    by_zip: dict[Path, list[str]] = {}
    for ym, zpath in month_sources:
        by_zip.setdefault(zpath, []).append(ym)

    table_regex = re.compile(r"^inf_mensal_fidc_(tab_[A-Za-z0-9_]+)_(\d{6})\.csv$")
    processed_months = 0

    for zpath, yms in by_zip.items():
        with zipfile.ZipFile(zpath) as zf:
            members_by_ym: dict[str, dict[str, str]] = {ym: {} for ym in yms}
            for member in zf.namelist():
                m = table_regex.match(Path(member).name)
                if m and m.group(2) in members_by_ym:
                    members_by_ym[m.group(2)][m.group(1)] = member

            for ym in sorted(yms):
                tmap = members_by_ym[ym]
                if "tab_I" not in tmap:
                    continue
                loaded_tables: dict[str, pd.DataFrame] = {}
                for tname, member_name in tmap.items():
                    if tname == "tab_VIII":
                        continue  # Tabela VIII é resumo contábil redundante às tabelas I–VII
                    raw_bytes = zf.read(member_name)
                    loaded_tables[tname] = pd.read_csv(
                        io.BytesIO(raw_bytes),
                        sep=";",
                        encoding="latin1",
                        dtype=str,
                    )
                month_core, month_edges, st = _parse_single_month_tables(ym, loaded_tables)
                core_chunks.append(month_core)
                if len(month_edges) > 0:
                    edge_chunks.append(month_edges)
                total_dups_removed += st["rcvm175_dups_removed"]
                processed_months += 1
                if processed_months % 18 == 0 or processed_months == len(month_sources):
                    print(f"  -> {processed_months}/{len(month_sources)} competências harmonizadas (última: {ym}, fundos={len(month_core)})")

    fidc_core = pd.concat(core_chunks, ignore_index=True)
    fidc_core = fidc_core.sort_values(["entity_id", "dt_comptc"]).reset_index(drop=True)
    cedente_edges = pd.concat(edge_chunks, ignore_index=True).sort_values(["dt_comptc", "entity_id"]).reset_index(drop=True)

    # Enriquecer com metadados cadastrais (Gestor, Custodiante, Público-Alvo, Data Constituição)
    reg_cols = [
        "entity_id",
        "dt_constituicao",
        "dt_cancelamento",
        "situacao_reg",
        "publico_alvo_reg",
        "cnpj_gestor",
        "nome_gestor",
        "cnpj_custodiante",
        "nome_custodiante",
        "taxa_adm_aa",
    ]
    fidc_core = fidc_core.merge(registry_df[reg_cols], on="entity_id", how="left")
    fidc_core["cnpj_gestor"] = fidc_core["cnpj_gestor"].fillna("")
    fidc_core["nome_gestor"] = fidc_core["nome_gestor"].fillna("")
    fidc_core["publico_alvo_reg"] = fidc_core["publico_alvo_reg"].fillna("")

    # --- Auditoria Automática de Invariantes Contábeis ---
    inad_buckets_sum = fidc_core[
        [
            "inad_venc_30",
            "inad_venc_60",
            "inad_venc_90",
            "inad_venc_120",
            "inad_venc_150",
            "inad_venc_180",
            "inad_venc_360",
            "inad_venc_720",
            "inad_venc_1080",
            "inad_venc_maior_1080",
        ]
    ].sum(axis=1)
    aging_diff = (inad_buckets_sum - fidc_core["vl_dircred_inad_total"]).abs()
    aging_invariant_ok_rate = float((aging_diff <= np.maximum(1000.0, 0.02 * fidc_core["vl_dircred_inad_total"])).mean())

    tranche_pl_sum = fidc_core[["pl_senior", "pl_mezanino", "pl_subordinada", "pl_unica"]].sum(axis=1)
    has_tranche_pl = tranche_pl_sum > 0
    pl_diff_rel = (tranche_pl_sum[has_tranche_pl] - fidc_core.loc[has_tranche_pl, "vl_pl"]).abs() / np.maximum(
        fidc_core.loc[has_tranche_pl, "vl_pl"].abs(), 1.0
    )
    tranche_pl_reconciled_rate = float((pl_diff_rel <= 0.05).mean()) if has_tranche_pl.any() else 0.0

    core_path = SILVER_DIR / "fidc_monthly_core.parquet"
    edges_path = SILVER_DIR / "cedente_edges.parquet"
    fidc_core.to_parquet(core_path, index=False)
    cedente_edges.to_parquet(edges_path, index=False)

    print("[Silver] 4/4 Materializando views e tabelas analíticas no DuckDB Lakehouse...")
    if LAKEHOUSE_DB_PATH.exists():
        LAKEHOUSE_DB_PATH.unlink()
    con = duckdb.connect(str(LAKEHOUSE_DB_PATH))
    try:
        con.execute("CREATE SCHEMA IF NOT EXISTS silver;")
        con.execute(f"CREATE OR REPLACE VIEW silver.macro_monthly AS SELECT * FROM read_parquet('{macro_path.as_posix()}');")
        con.execute(f"CREATE OR REPLACE VIEW silver.fund_registry AS SELECT * FROM read_parquet('{registry_path.as_posix()}');")
        con.execute(f"CREATE OR REPLACE VIEW silver.fidc_monthly_core AS SELECT * FROM read_parquet('{core_path.as_posix()}');")
        con.execute(f"CREATE OR REPLACE VIEW silver.cedente_edges AS SELECT * FROM read_parquet('{edges_path.as_posix()}');")
    finally:
        con.close()

    audit_summary = {
        "elapsed_seconds": round(time.perf_counter() - t0, 2),
        "months_processed": processed_months,
        "min_competence": str(fidc_core["dt_comptc"].min().date()),
        "max_competence": str(fidc_core["dt_comptc"].max().date()),
        "fund_month_rows": int(len(fidc_core)),
        "unique_entities": int(fidc_core["entity_id"].nunique()),
        "unique_gestores": int(fidc_core.loc[fidc_core["cnpj_gestor"] != "", "cnpj_gestor"].nunique()),
        "cedente_edge_rows": int(len(cedente_edges)),
        "unique_cedentes": int(cedente_edges["cedente_doc"].nunique()),
        "unique_pj_cedentes": int(cedente_edges.loc[cedente_edges["is_pj"] == 1, "cedente_doc"].nunique()),
        "rcvm175_transition_duplicates_resolved": int(total_dups_removed),
        "aging_bucket_sum_invariant_pass_rate": round(aging_invariant_ok_rate, 5),
        "tranche_pl_vs_tab4_reconciliation_rate": round(tranche_pl_reconciled_rate, 5),
        "gestor_coverage_rate": round(float((fidc_core["cnpj_gestor"] != "").mean()), 5),
    }
    audit_path = REPORTS_DIR / "metrics" / "silver_quality_audit.json"
    audit_path.write_text(json.dumps(audit_summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[Silver] Concluído em {audit_summary['elapsed_seconds']}s! Resumo de auditoria:")
    for k, v in audit_summary.items():
        print(f"  - {k}: {v}")
    return audit_summary

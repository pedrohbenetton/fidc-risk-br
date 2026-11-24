"""Testes matemáticos de Invariância Anti-Vazamento Temporal e Grafo Leave-One-Out (LOO)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from fidc_risk.features.gold import ALL_FEATURE_COLS, compute_gold_features


def _make_synthetic_silver_panel() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dates = pd.date_range("2023-01-31", periods=10, freq="ME")
    entities = ["11111111000191", "22222222000192", "33333333000193"]
    rows = []
    edges = []

    rng = np.random.default_rng(42)
    for dt in dates:
        for idx, eid in enumerate(entities):
            cart = 20_000_000.0 + idx * 5_000_000.0
            inad_30 = float(rng.uniform(100_000, 800_000))
            inad_120 = float(rng.uniform(100_000, 900_000))
            rows.append(
                {
                    "entity_id": eid,
                    "cnpj_raw": eid,
                    "denom_social": f"FUNDO TESTE {idx} FIDC",
                    "dt_comptc": dt,
                    "cnpj_admin": "90000000000101",
                    "nome_admin": "ADMIN TESTE",
                    "cnpj_gestor": "80000000000102",
                    "nome_gestor": "GESTOR COMPARTILHADO",
                    "cnpj_custodiante": "70000000000103",
                    "nome_custodiante": "CUSTODIANTE",
                    "dt_constituicao": pd.Timestamp("2021-01-15"),
                    "dt_cancelamento": pd.NaT,
                    "situacao_reg": "EM FUNCIONAMENTO NORMAL",
                    "publico_alvo_reg": "QUALIFICADO",
                    "taxa_adm_aa": 1.0,
                    "condominio": "ABERTO",
                    "is_aberto": 1,
                    "is_exclusivo": 0,
                    "has_cotista_interesse": 0,
                    "is_classe_unica_flag": 0,
                    "prazo_conversao_dias": 30.0,
                    "prazo_pagto_resgate_dias": 2.0,
                    "vl_ativo": cart * 1.15,
                    "vl_disp": cart * 0.10,
                    "vl_carteira": cart,
                    "vl_carteira_tab2": cart,
                    "vl_dircred_risco": cart,
                    "vl_dircred_sem_risco": 0.0,
                    "vl_cred_venc_ad": cart * 0.9,
                    "vl_cred_venc_inad": inad_30 + inad_120,
                    "vl_cred_inad": inad_120,
                    "vl_cred_perform": cart * 0.85,
                    "vl_cred_emp_recup": 0.0,
                    "vl_cred_acao_judic": 0.0,
                    "vl_pdd_total": inad_120 * 0.7,
                    "vl_titpub_liq": cart * 0.05,
                    "vl_cota_fidc": 0.0,
                    "vl_cota_fidc_np": 0.0,
                    "n_cedentes_disclosed": 2.0,
                    "n_cedentes_pj": 2.0,
                    "top1_cedente_pct": 30.0,
                    "top3_cedente_pct": 50.0,
                    "top9_cedente_pct": 50.0,
                    "cedente_hhi": 0.13,
                    "sec_industrial": cart * 0.6,
                    "sec_imobiliario": 0.0,
                    "sec_comercial_varejo": cart * 0.4,
                    "sec_servicos": 0.0,
                    "sec_agronegocio": 0.0,
                    "sec_cred_pessoal": 0.0,
                    "sec_consignado": 0.0,
                    "sec_corporativo": 0.0,
                    "sec_veiculos": 0.0,
                    "sec_cartoes_credito": 0.0,
                    "sec_factoring": 0.0,
                    "sec_setor_publico": 0.0,
                    "sec_judicial": 0.0,
                    "vl_passivo": 50_000.0,
                    "vl_derivativos_passivo": 0.0,
                    "vl_pl": cart * 1.10,
                    "vl_pl_medio": cart * 1.10,
                    "vl_dircred_prazo_total": cart - inad_30 - inad_120,
                    "vl_dircred_inad_total": inad_30 + inad_120,
                    "prazo_venc_30": cart * 0.3,
                    "prazo_venc_60": cart * 0.3,
                    "prazo_venc_90": cart * 0.2,
                    "prazo_venc_120": 0.0,
                    "prazo_venc_150": 0.0,
                    "prazo_venc_180": 0.0,
                    "prazo_venc_360": 0.0,
                    "prazo_venc_720": 0.0,
                    "prazo_venc_1080": 0.0,
                    "prazo_venc_maior_1080": 0.0,
                    "inad_venc_30": inad_30,
                    "inad_venc_60": 50_000.0,
                    "inad_venc_90": 20_000.0,
                    "inad_venc_120": inad_120,
                    "inad_venc_150": 0.0,
                    "inad_venc_180": 0.0,
                    "inad_venc_360": 0.0,
                    "inad_venc_720": 0.0,
                    "inad_venc_1080": 0.0,
                    "inad_venc_maior_1080": 0.0,
                    "vl_aquisicoes_mes": cart * 0.2,
                    "vl_alienacoes_cedente": 0.0,
                    "vl_alienacoes_terceiros": 0.0,
                    "qt_substituicoes_mes": 2.0,
                    "vl_substituicoes_mes": 40_000.0,
                    "qt_recompras_mes": 5.0,
                    "vl_recompras_mes": 90_000.0,
                    "taxa_desconto_compra_media": 18.5,
                    "has_scr_report": 1,
                    "vl_scr_total": cart,
                    "vl_scr_efgh": inad_120,
                    "vl_scr_pdd_min_bacen2682": inad_120 * 0.8,
                    "pl_senior": cart * 0.85,
                    "pl_mezanino": cart * 0.05,
                    "pl_subordinada": cart * 0.20,
                    "pl_unica": 0.0,
                    "n_series_senior": 1,
                    "rentab_senior_mes": 1.05,
                    "rentab_mezanino_mes": 1.20,
                    "rentab_subordinada_mes": 1.80,
                    "rentab_unica_mes": np.nan,
                    "captacoes_total": 100_000.0,
                    "amortizacoes_total": 0.0,
                    "resgates_pagos_total": 50_000.0,
                    "resgates_solicitados_total": 30_000.0,
                    "captacoes_senior": 100_000.0,
                    "resgates_pagos_senior": 50_000.0,
                    "resgates_solicitados_senior": 30_000.0,
                    "nr_cotistas_senior": 45.0,
                    "nr_cotistas_subord": 2.0,
                    "nr_cotistas_total": 47.0,
                    "liq_imediata_0": cart * 0.1,
                    "liq_30d": cart * 0.25,
                    "liq_90d": cart * 0.30,
                    "liq_180_360d": cart * 0.20,
                    "liq_maior_360d": 0.0,
                    "desemp_esperado_senior": 1.00,
                    "desemp_real_senior": 1.05,
                    "vl_garantia_dircred": 0.0,
                    "pr_garantia_dircred": 0.0,
                }
            )
            # Cedente corporativo compartilhado entre os fundos 0 e 1
            if idx in (0, 1):
                edges.append(
                    {
                        "dt_comptc": dt,
                        "entity_id": eid,
                        "cedente_doc": "55555555000188",
                        "share_pct": 30.0,
                        "is_pj": 1,
                    }
                )

    macro = pd.DataFrame(
        {
            "dt_comptc": dates,
            "cdi_mensal_pct": 1.0,
            "cdi_12m_pct": 12.7,
            "selic_meta_aa": 13.25,
            "selic_delta_12m": 0.5,
            "ipca_mensal_pct": 0.4,
            "ipca_12m_pct": 4.5,
            "juro_real_expost_aa": 8.37,
            "ibc_br_yoy_pct": 2.4,
            "inad_sfn_total_pct": 3.5,
            "inad_sfn_pj_pct": 2.6,
            "inad_sfn_pf_pct": 5.4,
            "inad_sfn_pj_delta_6m": 0.3,
            "spread_sfn_pp": 20.1,
            "endividamento_familias_pct": 48.0,
            "comprometimento_renda_pct": 27.0,
            "desocupacao_pnad_pct": 7.5,
        }
    )
    return pd.DataFrame(rows), pd.DataFrame(edges), macro


def test_future_perturbation_invariance_zero_temporal_leakage() -> None:
    """
    Prova formal por perturbação adversarial do futuro:
    Corrompendo drasticamente todos os dados em dt > cutoff, NENHUMA feature causal em dt <= cutoff
    pode mudar em sequer 1e-12.
    """
    core, edges, macro = _make_synthetic_silver_panel()
    cutoff = pd.Timestamp("2023-06-30")

    gold_clean = compute_gold_features(core, edges, macro, include_targets=True)
    past_clean = gold_clean[gold_clean["dt_comptc"] <= cutoff].sort_values(["entity_id", "dt_comptc"])

    # Perturbar severamente tudo após cutoff
    core_corrupt = core.copy()
    fut_mask = core_corrupt["dt_comptc"] > cutoff
    core_corrupt.loc[fut_mask, "inad_venc_120"] = 19_000_000.0
    core_corrupt.loc[fut_mask, "vl_pdd_total"] = 18_000_000.0
    core_corrupt.loc[fut_mask, "vl_recompras_mes"] = 15_000_000.0
    core_corrupt.loc[fut_mask, "rentab_senior_mes"] = -45.0
    core_corrupt.loc[fut_mask, "vl_pl"] = 1_000_000.0

    edges_corrupt = edges.copy()
    edges_corrupt.loc[edges_corrupt["dt_comptc"] > cutoff, "share_pct"] = 99.0

    macro_corrupt = macro.copy()
    macro_corrupt.loc[macro_corrupt["dt_comptc"] > cutoff, "selic_meta_aa"] = 45.0

    gold_corrupt = compute_gold_features(core_corrupt, edges_corrupt, macro_corrupt, include_targets=True)
    past_corrupt = gold_corrupt[gold_corrupt["dt_comptc"] <= cutoff].sort_values(["entity_id", "dt_comptc"])

    for col in ALL_FEATURE_COLS:
        assert np.allclose(
            past_clean[col].values,
            past_corrupt[col].values,
            atol=1e-12,
            equal_nan=True,
        ), f"Vazamento temporal detectado na feature: {col}"


def test_leave_one_out_contagion_excludes_self() -> None:
    """
    Se apenas o fundo 0 sofre choque de inadimplência no mês t, o contágio LOO do fundo 1 (que
    compartilha cedente com o fundo 0) deve subir, enquanto o contágio LOO do próprio fundo 0
    NÃO deve refletir seu próprio choque!
    """
    core, edges, macro = _make_synthetic_silver_panel()
    target_dt = pd.Timestamp("2023-05-31")

    base_gold = compute_gold_features(core, edges, macro, include_targets=False)

    shocked_core = core.copy()
    m0 = (shocked_core["entity_id"] == "11111111000191") & (shocked_core["dt_comptc"] == target_dt)
    shocked_core.loc[m0, "inad_venc_120"] = 15_000_000.0  # Choque exclusivo no fundo 0

    shocked_gold = compute_gold_features(shocked_core, edges, macro, include_targets=False)

    b0 = base_gold[(base_gold["entity_id"] == "11111111000191") & (base_gold["dt_comptc"] == target_dt)]
    s0 = shocked_gold[(shocked_gold["entity_id"] == "11111111000191") & (shocked_gold["dt_comptc"] == target_dt)]
    b1 = base_gold[(base_gold["entity_id"] == "22222222000192") & (base_gold["dt_comptc"] == target_dt)]
    s1 = shocked_gold[(shocked_gold["entity_id"] == "22222222000192") & (shocked_gold["dt_comptc"] == target_dt)]

    # Fundo 0 NÃO pode vazar seu próprio choque para suas features LOO
    assert np.isclose(b0["contagion_cedente_inad90p_wt"].iloc[0], s0["contagion_cedente_inad90p_wt"].iloc[0])
    assert np.isclose(b0["gestor_loo_inad90p"].iloc[0], s0["gestor_loo_inad90p"].iloc[0])

    # Fundo 1 (par do mesmo cedente e mesmo gestor) DEVE receber o spillover do choque do fundo 0
    assert s1["contagion_cedente_inad90p_wt"].iloc[0] > b1["contagion_cedente_inad90p_wt"].iloc[0]
    assert s1["gestor_loo_inad90p"].iloc[0] > b1["gestor_loo_inad90p"].iloc[0]

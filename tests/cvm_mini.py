"""Mini-informe sintético da CVM (tabelas I, II, IV, V, VI, VII, X_2 e X_4) para testar o boletim.

Quatro classes fictícias, valores pequenos e redondos para que os indicadores possam ser
conferidos à mão (ver EXPECTED_MONTH). Os CSVs saem em latin1 e separados por ';', como na CVM.

Classe  Carteira (V+VI)                      PL     PDD  Recompra  Segmento    Cedentes PJ
F1      800 a vencer + 100 (≤30d) + 50 (31–60d) + 50 (91–120d)
                                             1000   30   40        Industrial  X
F2      600 a vencer + 300 (121–150d) + 100 (361–720d)
                                             900    100  0         Financeiro  X, Y   ← estresse (40% vencido 90+)
F3      1900 a vencer + 100 (61–90d)         2100   0    60        Financeiro  Y, Z
F4      sem carteira nas tabelas V/VI        500    50   0         —           Z
"""

from __future__ import annotations

import csv
from pathlib import Path

FAIXAS = ["30", "60", "90", "120", "150", "180", "360", "720", "1080", "MAIOR_1080"]

F1, F2, F3, F4 = "11.111.111/0001-11", "22.222.222/0001-22", "33.333.333/0001-33", "44.444.444/0001-44"
CED_X, CED_Y, CED_Z = "99.999.999/0001-91", "88.888.888/0001-88", "77.777.777/0001-77"
CPF_PF = "123.456.789-09"

NOMES = {F1: "FIDC Ação Sintética", F2: "FIDC Crédito Fictício", F3: "FIDC Exemplo Três", F4: "FIDC Sem Carteira"}


_REMAP: dict[str, str] = {}


def _write(path: Path, header: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="latin1", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=header, delimiter=";", extrasaction="raise")
        w.writeheader()
        for r in rows:
            w.writerow({k: _REMAP.get(str(r.get(k, "")), r.get(k, "")) for k in header})


def _base(cnpj: str, aaaamm: str) -> dict[str, object]:
    return {
        "TP_FUNDO_CLASSE": "CLASSE",
        "CNPJ_FUNDO_CLASSE": cnpj,
        "DENOM_SOCIAL": NOMES[cnpj],
        "DT_COMPTC": f"{aaaamm[:4]}-{aaaamm[4:]}-28",
    }


BASE_COLS = ["TP_FUNDO_CLASSE", "CNPJ_FUNDO_CLASSE", "DENOM_SOCIAL", "DT_COMPTC"]


def aging_cols(tab: str) -> list[str]:
    a = [f"TAB_{tab}_A{i}_VL_PRAZO_VENC_{s}" for i, s in enumerate(FAIXAS, 1)]
    b = [f"TAB_{tab}_B{i}_VL_INAD_{s}" for i, s in enumerate(FAIXAS, 1)]
    return a + b


def write_mini_competencia(d: Path, aaaamm: str = "202610", remap: dict[str, str] | None = None) -> Path:
    """Escreve os CSVs de uma competência sintética em ``d`` e devolve o diretório.

    ``remap`` troca CNPJs na escrita (ex.: por CNPJs fictícios com dígitos verificadores válidos).
    """
    d.mkdir(parents=True, exist_ok=True)
    _REMAP.clear()
    _REMAP.update(remap or {})
    f = f"inf_mensal_fidc_tab_{{}}_{aaaamm}.csv"

    # Tabela I: PDD (I2A11/I2B11, informadas com sinal negativo) e cedentes (I2A12/I2B12).
    ced_cols = [f"TAB_I2{g}12_CPF_CNPJ_CEDENTE_{i}" for g in ("A", "B") for i in range(1, 10)]
    t1_cols = BASE_COLS + ["TAB_I2A11_VL_REDUCAO_RECUP", "TAB_I2B11_VL_REDUCAO_RECUP"] + ced_cols
    _write(d / f.format("I"), t1_cols, [
        {**_base(F1, aaaamm), "TAB_I2A11_VL_REDUCAO_RECUP": "-20", "TAB_I2B11_VL_REDUCAO_RECUP": "-10",
         "TAB_I2A12_CPF_CNPJ_CEDENTE_1": CED_X, "TAB_I2A12_CPF_CNPJ_CEDENTE_2": CPF_PF},
        {**_base(F2, aaaamm), "TAB_I2A11_VL_REDUCAO_RECUP": "-100",
         "TAB_I2A12_CPF_CNPJ_CEDENTE_1": CED_X, "TAB_I2B12_CPF_CNPJ_CEDENTE_1": CED_Y},
        {**_base(F3, aaaamm), "TAB_I2A11_VL_REDUCAO_RECUP": "0",
         "TAB_I2A12_CPF_CNPJ_CEDENTE_1": CED_Y, "TAB_I2A12_CPF_CNPJ_CEDENTE_2": CED_Z},
        {**_base(F4, aaaamm), "TAB_I2A11_VL_REDUCAO_RECUP": "-50", "TAB_I2A12_CPF_CNPJ_CEDENTE_1": CED_Z},
    ])

    seg_cols = ["TAB_II_A_VL_INDUST", "TAB_II_B_VL_IMOBIL", "TAB_II_C_VL_COMERC", "TAB_II_D_VL_SERV",
                "TAB_II_E_VL_AGRONEG", "TAB_II_F_VL_FINANC", "TAB_II_G_VL_CREDITO", "TAB_II_H_VL_FACTOR",
                "TAB_II_I_VL_SETOR_PUBLICO", "TAB_II_J_VL_JUDICIAL", "TAB_II_K_VL_MARCA"]
    _write(d / f.format("II"), BASE_COLS + seg_cols, [
        {**_base(F1, aaaamm), "TAB_II_A_VL_INDUST": "1000"},
        {**_base(F2, aaaamm), "TAB_II_F_VL_FINANC": "1000"},
        {**_base(F3, aaaamm), "TAB_II_F_VL_FINANC": "1500", "TAB_II_E_VL_AGRONEG": "500"},
        {**_base(F4, aaaamm)},
    ])

    _write(d / f.format("IV"), BASE_COLS + ["TAB_IV_A_VL_PL"], [
        {**_base(F1, aaaamm), "TAB_IV_A_VL_PL": "1000"},
        {**_base(F2, aaaamm), "TAB_IV_A_VL_PL": "900"},
        {**_base(F3, aaaamm), "TAB_IV_A_VL_PL": "2100"},
        {**_base(F4, aaaamm), "TAB_IV_A_VL_PL": "500"},
    ])

    _write(d / f.format("V"), BASE_COLS + aging_cols("V"), [
        {**_base(F1, aaaamm), "TAB_V_A1_VL_PRAZO_VENC_30": "800", "TAB_V_B1_VL_INAD_30": "100",
         "TAB_V_B2_VL_INAD_60": "50", "TAB_V_B4_VL_INAD_120": "50"},
        {**_base(F2, aaaamm), "TAB_V_A1_VL_PRAZO_VENC_30": "500", "TAB_V_B5_VL_INAD_150": "300"},
        {**_base(F3, aaaamm), "TAB_V_A1_VL_PRAZO_VENC_30": "1900", "TAB_V_B3_VL_INAD_90": "100"},
    ])
    _write(d / f.format("VI"), BASE_COLS + aging_cols("VI"), [
        {**_base(F2, aaaamm), "TAB_VI_A2_VL_PRAZO_VENC_60": "100", "TAB_VI_B8_VL_INAD_720": "100"},
    ])

    _write(d / f.format("VII"), BASE_COLS + ["TAB_VII_C_2_VL_SUBST", "TAB_VII_D_2_VL_RECOMPRA"], [
        {**_base(F1, aaaamm), "TAB_VII_D_2_VL_RECOMPRA": "40"},
        {**_base(F2, aaaamm), "TAB_VII_C_2_VL_SUBST": "20", "TAB_VII_D_2_VL_RECOMPRA": "0"},
        {**_base(F3, aaaamm), "TAB_VII_D_2_VL_RECOMPRA": "60"},
    ])

    # X_2: F3 tem cotas que somam 100 contra PL de 2100 (razão < 0,5) e é descartada da subordinação.
    _write(d / f.format("X_2"), BASE_COLS + ["TAB_X_CLASSE_SERIE", "TAB_X_QT_COTA", "TAB_X_VL_COTA"], [
        {**_base(F1, aaaamm), "TAB_X_CLASSE_SERIE": "Sênior", "TAB_X_QT_COTA": "8", "TAB_X_VL_COTA": "100"},
        {**_base(F1, aaaamm), "TAB_X_CLASSE_SERIE": "Subordinada", "TAB_X_QT_COTA": "2", "TAB_X_VL_COTA": "100"},
        {**_base(F2, aaaamm), "TAB_X_CLASSE_SERIE": "Sênior Série 1", "TAB_X_QT_COTA": "9", "TAB_X_VL_COTA": "100"},
        {**_base(F3, aaaamm), "TAB_X_CLASSE_SERIE": "Subordinada Júnior", "TAB_X_QT_COTA": "1", "TAB_X_VL_COTA": "100"},
        {**_base(F4, aaaamm), "TAB_X_CLASSE_SERIE": "Subordinada", "TAB_X_QT_COTA": "5", "TAB_X_VL_COTA": "100"},
    ])

    _write(d / f.format("X_4"), BASE_COLS + ["TAB_X_TP_OPER", "TAB_X_CLASSE_SERIE", "TAB_X_VL_TOTAL"], [
        {**_base(F1, aaaamm), "TAB_X_TP_OPER": "Captações no Mês", "TAB_X_VL_TOTAL": "300"},
        {**_base(F1, aaaamm), "TAB_X_TP_OPER": "Captações no Mês", "TAB_X_VL_TOTAL": "200"},
        {**_base(F2, aaaamm), "TAB_X_TP_OPER": "Resgates no Mês", "TAB_X_VL_TOTAL": "100"},
        {**_base(F3, aaaamm), "TAB_X_TP_OPER": "Amortizações", "TAB_X_VL_TOTAL": "50"},
    ])
    return d


# Valores esperados, conferidos à mão a partir da tabela do docstring.
# Carteira total (F1+F2+F3) = 1000 + 1000 + 2000 = 4000; F4 não tem carteira e fica fora da base.
EXPECTED_MONTH: dict[str, object] = {
    "n_classes": 4,                      # classes com PL > 0
    "pl_bi": 4500 / 1e9,                 # 1000 + 900 + 2100 + 500
    "carteira_dc_bi": 4000 / 1e9,
    "inad30_pct": 600 / 4000 * 100,      # F1: 50+50; F2: 300+100; F3: 100 (61–90d)
    "inad90_pct": 450 / 4000 * 100,      # F1: 50; F2: 400; F3: 0
    "inad180_pct": 100 / 4000 * 100,     # só F2 (361–720d)
    "inad90_mediana_pct": 5.0,           # mediana de {5%, 40%, 0%}
    "pdd_pct": 130 / 4000 * 100,         # F1: 20+10; F2: 100; F3: 0 (F4 fora da base)
    "recompra_mi": 100 / 1e6,
    "recompra_pct": 100 / 4000 * 100,
    "subst_mi": 20 / 1e6,
    "subord_pct": (0.2 * 1000 + 0.0 * 900 + 1.0 * 500) / 2400 * 100,  # F3 descartada (cotas/PL < 0,5)
    "classes_estresse": 1,               # F2, com 40% vencido 90+
    "classes_estresse_pct": 1 / 3 * 100,
    "ops_bi": {"Amortizações": 50 / 1e9, "Captações no Mês": 500 / 1e9, "Resgates no Mês": 100 / 1e9},
    "cedentes_pj": 3,                    # X, Y, Z (o CPF de F1 é ignorado)
    "cedentes_compartilhados": 3,        # X: F1,F2 · Y: F2,F3 · Z: F3,F4
    "cedentes_ligados_estresse": 2,      # X e Y aparecem em F2
    "classes_expostas": 2,               # F1 e F3
    "pl_exposto_bi": (1000 + 2100) / 1e9,
    "segmentos": [
        {"seg": "Financeiro", "carteira_bi": 3000 / 1e9, "inad90_pct": 400 / 3000 * 100, "n": 2},
        {"seg": "Industrial", "carteira_bi": 1000 / 1e9, "inad90_pct": 5.0, "n": 1},
    ],
    "dist_inad90": [5.0, 40.0, 0.0],     # ordem das classes na base (F1, F2, F3)
}

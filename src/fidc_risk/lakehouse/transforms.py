"""Funções puras e determinísticas de normalização contábil, documentos e tranches da CVM."""

from __future__ import annotations

import re
import unicodedata

import numpy as np
import pandas as pd

_NON_DIGITS_RE = re.compile(r"[^0-9]")


def normalize_doc(raw: object) -> str:
    """Remove pontuação de CPF/CNPJ e descarta placeholders inválidos (ex.: zeros)."""
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return ""
    s = re.sub(r"\.0+$", "", str(raw).strip())
    digits = _NON_DIGITS_RE.sub("", s)
    if not digits or len(digits) not in (11, 14):
        return ""
    if len(set(digits)) == 1:  # Ex.: 00000000000000, 11111111111111
        return ""
    return digits


def normalize_doc_series(series: pd.Series) -> pd.Series:
    """Versão vetorizada rápida para colunas inteiras de CPF/CNPJ."""
    cleaned = (
        series.fillna("")
        .astype(str)
        .str.strip()
        .str.replace(r"\.0+$", "", regex=True)
        .str.replace(r"[^0-9]", "", regex=True)
    )
    valid_len = cleaned.str.len().isin([11, 14])
    not_repeated = ~cleaned.str.match(r"^(\d)\1+$", na=True)
    return cleaned.where(valid_len & not_repeated, "")


def strip_accents_upper(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    ascii_text = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    return ascii_text.upper().strip()


def classify_tranche_label(raw_label: object) -> str:
    """
    Classifica descrições heterogêneas de TAB_X_CLASSE_SERIE na hierarquia de waterfall:
    - 'Subordinada Mezanino' deve ser classificada como MEZANINO (antes de SUBORDINADA!).
    - 'Subordinada' / 'Subordinada Júnior' -> SUBORDINADA.
    - 'Sênior' / 'Subclasse Senior' / 'Série 1' -> SENIOR.
    - 'Única' / vazio -> UNICA.
    """
    if raw_label is None or (isinstance(raw_label, float) and np.isnan(raw_label)):
        return "UNICA"
    txt = strip_accents_upper(str(raw_label))
    if not txt:
        return "UNICA"
    if "MEZ" in txt:
        return "MEZANINO"
    if "SUBORD" in txt or "JUNIOR" in txt or re.search(r"\bJR\b", txt):
        return "SUBORDINADA"
    if "SEN" in txt or "SERIE" in txt or re.search(r"\bSR\b", txt):
        return "SENIOR"
    if "UNIC" in txt:
        return "UNICA"
    return "UNICA"


def harmonize_percentage_share(values: pd.Series, max_cap: float = 100.0) -> pd.Series:
    """
    Corrige erros clássicos de preenchimento em informes da CVM para participações (%):
    - Valores negativos -> 0.0
    - Valores entre (100, 10000] onde o administrador digitou pontos-base (ex.: 2540 -> 25.40%) -> / 100
    - Acima do teto -> truncado em max_cap.
    """
    num = pd.to_numeric(values, errors="coerce").fillna(0.0).astype(float)
    num = num.clip(lower=0.0)
    bps_mask = (num > 100.0) & (num <= 10_000.0)
    num = np.where(bps_mask, num / 100.0, num)
    return pd.Series(np.clip(num, 0.0, max_cap), index=values.index, dtype="float64")


def sanitize_monthly_return_pct(values: pd.Series) -> pd.Series:
    """
    Higieniza TAB_X_VL_RENTAB_MES (% ao mês) preservando quedas reais de default (-99% a +40% a.m.)
    e marcando como NaN erros grosseiros de unidade da CVM (ex.: 105000% num único mês).
    """
    num = pd.to_numeric(values, errors="coerce").astype("float64")
    # Retornos mensais abaixo de -100% são contabilmente impossíveis p/ cota limitada,
    # e acima de +60% em 1 único mês em FIDC indicam erro de escala/reagrupamento de cota.
    invalid = (num < -100.0) | (num > 60.0)
    return num.mask(invalid, np.nan)

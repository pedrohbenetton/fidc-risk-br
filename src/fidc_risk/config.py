"""Configurações centrais, caminhos do Lakehouse, constantes regulatórias e janelas temporais."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
BRONZE_DIR = DATA_DIR / "bronze"
SILVER_DIR = DATA_DIR / "silver"
GOLD_DIR = DATA_DIR / "gold"
LAKEHOUSE_DB_PATH = DATA_DIR / "lakehouse.duckdb"

MODELS_DIR = PROJECT_ROOT / "models" / "checkpoints"
REPORTS_DIR = PROJECT_ROOT / "reports"
METRICS_DIR = REPORTS_DIR / "metrics"
FIGURES_DIR = REPORTS_DIR / "figures"
WEB_DIR = PROJECT_ROOT / "web"
WEB_DATA_DIR = WEB_DIR / "data"

# Artefatos com fundos identificados: ficam fora de web/ (nunca publicados) e fora do git.
PRIVATE_DATA_DIR = DATA_DIR / "private"
PRIVATE_BUNDLE_PATH = PRIVATE_DATA_DIR / "observatory_bundle.json"
DEMO_CODES_PATH = PRIVATE_DATA_DIR / "demo_codes.json"  # mapeamento fundo/gestora -> código estável
DEMO_DIST_DIR = PROJECT_ROOT / "dist" / "demo"  # build estático anonimizado (Vercel)

# Endpoints oficiais públicos (CVM Dados Abertos + Banco Central SGS)
CVM_FIDC_MONTHLY_BASE = "https://dados.cvm.gov.br/dados/FIDC/DOC/INF_MENSAL/DADOS/"
CVM_FIDC_HIST_BASE = "https://dados.cvm.gov.br/dados/FIDC/DOC/INF_MENSAL/DADOS/HIST/"
CVM_CAD_RCVM175_URL = "https://dados.cvm.gov.br/dados/FI/CAD/DADOS/registro_fundo_classe.zip"
CVM_CAD_LEGACY_URL = "https://dados.cvm.gov.br/dados/FI/CAD/DADOS/cad_fi.csv"
BCB_SGS_TEMPLATE = (
    "https://api.bcb.gov.br/dados/serie/bcdata.sgs.{code}/dados"
    "?formato=json&dataInicial={start_date}&dataFinal={end_date}"
)

HIST_YEARS: tuple[int, ...] = (2018, 2019, 2020, 2021, 2022, 2023, 2024)

# Séries oficiais do Sistema Gerenciador de Séries Temporais (SGS) do Banco Central
BCB_SERIES: dict[str, int] = {
    "cdi_mensal_pct": 4391,
    "selic_meta_aa": 432,
    "ipca_mensal_pct": 433,
    "ibc_br": 24363,
    "inad_sfn_total_pct": 21082,
    "inad_sfn_pj_pct": 21083,
    "inad_sfn_pf_pct": 21084,
    "spread_sfn_pp": 20783,
    "endividamento_familias_pct": 29037,
    "comprometimento_renda_pct": 29034,
    "desocupacao_pnad_pct": 24369,
}

# Percentuais mínimos regulatórios de provisão por nível de risco (Resolução CMN/BCB nº 2.682)
BACEN_2682_PROVISION_WEIGHTS: dict[str, float] = {
    "AA": 0.000,
    "A": 0.005,
    "B": 0.010,
    "C": 0.030,
    "D": 0.100,
    "E": 0.300,
    "F": 0.500,
    "G": 0.700,
    "H": 1.000,
}


@dataclass(frozen=True)
class WalkForwardFold:
    """Janela de validação temporal honesta (expanding window com embargo = horizon)."""

    name: str
    regime_label: str
    test_start: str  # YYYY-MM-DD (inclusive)
    test_end: str    # YYYY-MM-DD (inclusive)


WALK_FORWARD_FOLDS: tuple[WalkForwardFold, ...] = (
    WalkForwardFold(
        name="fold_2022",
        regime_label="2022: Ciclo de Alta da Selic (9,25% -> 13,75%)",
        test_start="2022-01-01",
        test_end="2022-12-31",
    ),
    WalkForwardFold(
        name="fold_2023",
        regime_label="2023: Crise Corporativa (Americanas/Light) e Juros Restritivos",
        test_start="2023-01-01",
        test_end="2023-12-31",
    ),
    WalkForwardFold(
        name="fold_2024",
        regime_label="2024: Expansão RCVM 175 e Reabertura do Crédito Privado",
        test_start="2024-01-01",
        test_end="2024-12-31",
    ),
    WalkForwardFold(
        name="fold_2025_2026",
        regime_label="2025–26: Novo Ciclo de Aperto Monetário e Estresse Setorial",
        test_start="2025-01-01",
        test_end="2026-10-31",
    ),
)


def ensure_directories() -> None:
    """Cria a hierarquia completa de diretórios do projeto caso não exista."""
    for path in (
        BRONZE_DIR / "cvm_fidc_hist",
        BRONZE_DIR / "cvm_fidc_monthly",
        BRONZE_DIR / "cvm_cad",
        BRONZE_DIR / "bcb_sgs",
        SILVER_DIR,
        GOLD_DIR,
        MODELS_DIR,
        METRICS_DIR,
        FIGURES_DIR,
        WEB_DATA_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)

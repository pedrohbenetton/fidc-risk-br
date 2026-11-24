""" Ponto de entrada da CLI do fidc-risk-br."""

from __future__ import annotations

import argparse
import sys


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fidc-risk",
        description="Pipeline end-to-end de engenharia de dados, modelagem de risco, GNN em GPU e alocação para FIDCs.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("ingest", help="Baixa informes mensais da CVM, cadastro de fundos e séries do BCB (Bronze).")
    sub.add_parser("silver", help="Normaliza tabelas, reconcilia RCVM 175 e valida invariantes contábeis (Silver).")
    sub.add_parser("gold", help="Gera painel point-in-time, sinais forenses e grafo de contágio (Gold).")
    sub.add_parser("train-tabular", help="Treina e avalia baselines, modelos lineares, LightGBM, CatBoost e calibração.")
    sub.add_parser("train-rollrate", help="Estima matrizes de roll-rate de Markov e regressão quantílica conformal (CQR).")
    sub.add_parser("train-gnn", help="Treina redes temporais e GNN heterogênea de contágio em GPU (PyTorch CUDA).")
    sub.add_parser("backtest-allocation", help="Executa estudo forense e backtest de alocação sob restrição de liquidez e CVaR.")
    sub.add_parser("export-web", help="Exporta artefatos estáticos verificados para o observatório web e figuras.")
    demo_p = sub.add_parser(
        "export-demo",
        help="Gera o site demo com fundos anonimizados (códigos estáveis, PL em faixas), pronto para a Vercel.",
    )
    demo_p.add_argument("--out", default=None, help="Diretório de saída (padrão: dist/demo).")
    demo_p.add_argument(
        "--rebuild-bundle",
        action="store_true",
        help="Remonta o pacote nominal a partir dos artefatos Gold/Silver em vez de reaproveitar data/private/.",
    )
    pages_p = sub.add_parser(
        "build-pages",
        help="Monta o site do GitHub Pages a partir de web/ (pacote anonimizado versionado); falha se houver vazamento.",
    )
    pages_p.add_argument("--out", default="_site", help="Diretório de saída (padrão: _site).")
    sub.add_parser(
        "export-methodology",
        help="Recalcula a validação por safra e gera a página e o PDF de Metodologia (web/metodologia.*).",
    )
    sub.add_parser(
        "study-cedente-contagion",
        help="Estudo 2022–2026: classes que dividem cedente com quem entrou em estresse pioram mais que controles?",
    )
    serve_p = sub.add_parser("serve", help="Inicia API analítica FastAPI sobre o Lakehouse DuckDB.")
    serve_p.add_argument("--host", default="127.0.0.1")
    serve_p.add_argument("--port", type=int, default=8000)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    cmd = args.command

    if cmd == "ingest":
        from fidc_risk.ingestion.run import run_ingestion
        run_ingestion()
    elif cmd == "silver":
        from fidc_risk.lakehouse.silver import build_silver_layer
        build_silver_layer()
    elif cmd == "gold":
        from fidc_risk.features.gold import build_gold_panel
        build_gold_panel()
    elif cmd == "train-tabular":
        from fidc_risk.models.tabular_suite import run_tabular_benchmark
        run_tabular_benchmark()
    elif cmd == "train-rollrate":
        from fidc_risk.models.rollrate_cqr import run_rollrate_and_cqr_benchmark
        run_rollrate_and_cqr_benchmark()
    elif cmd == "train-gnn":
        from fidc_risk.models.deep_contagion import run_deep_contagion_benchmark
        run_deep_contagion_benchmark()
    elif cmd == "backtest-allocation":
        from fidc_risk.allocation.portfolio import run_forensic_and_allocation_suite
        run_forensic_and_allocation_suite()
    elif cmd == "export-web":
        from fidc_risk.export.web_bundle import export_all
        export_all()
    elif cmd == "export-demo":
        from pathlib import Path

        from fidc_risk.config import DEMO_DIST_DIR
        from fidc_risk.export.web_bundle import export_demo
        export_demo(Path(args.out) if args.out else DEMO_DIST_DIR, rebuild_bundle=args.rebuild_bundle)
    elif cmd == "build-pages":
        from fidc_risk.export.demo import build_pages_site
        build_pages_site(args.out)
    elif cmd == "export-methodology":
        from fidc_risk.export.methodology_page import export_methodology
        export_methodology()
    elif cmd == "study-cedente-contagion":
        from fidc_risk.studies.cedente_contagion import run_and_export
        run_and_export()
    elif cmd == "serve":
        import uvicorn
        uvicorn.run("fidc_risk.api.app:app", host=args.host, port=args.port, reload=False)
    else:
        parser.print_help()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

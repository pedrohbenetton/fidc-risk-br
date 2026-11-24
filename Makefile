.PHONY: install ingest silver gold train backtest export serve test lint all

install:
	uv sync --extra dev

ingest:
	uv run fidc-risk ingest

silver:
	uv run fidc-risk silver

gold:
	uv run fidc-risk gold

train:
	uv run fidc-risk train-tabular
	uv run fidc-risk train-rollrate
	uv run fidc-risk train-gnn

backtest:
	uv run fidc-risk backtest-allocation

export:
	uv run fidc-risk export-web

serve:
	uv run fidc-risk serve

test:
	uv run pytest -v --cov=fidc_risk

lint:
	uv run ruff check src tests
	uv run mypy src

all: ingest silver gold train backtest export

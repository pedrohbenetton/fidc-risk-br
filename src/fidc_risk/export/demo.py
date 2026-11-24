"""Build "demo" do Observatório: site estático com fundos anonimizados, pronto para publicar (Vercel).

O build copia só as páginas, scripts, estilos e o pacote anonimizado; nunca o pacote nominal. Antes de
terminar, varre todos os arquivos gerados atrás de CNPJs e de nomes reais de fundos e gestores; se
achar algo, apaga o build e falha.
"""

from __future__ import annotations

import json
import re
import shutil
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fidc_risk.config import WEB_DIR
from fidc_risk.export.anonymize import CodeBook, anonymize_bundle

DEMO_BANNER_TEXT = "Demonstração: fundos anonimizados"
# Tudo o que sai de web/ para o público. O pacote nominal (data/private/) nunca entra.
PUBLISHED_WEB_FILES: tuple[str, ...] = (
    "index.html",
    "metodologia.html",
    "metodologia.pdf",
    "app.js",
    "styles.css",
    "data/observatory_bundle.json",
)
_BUILD_MARKER = ".fidc-demo-build"
_TEXT_SUFFIXES = {".html", ".htm", ".js", ".css", ".json", ".txt", ".md", ".svg", ".xml", ".csv"}

CNPJ_FMT = re.compile(r"\d{2}\.\d{3}\.\d{3}/\d{4}-\d{2}")
# 14 dígitos isolados; ignora casas decimais longas (precedidas de ponto) e números maiores.
CNPJ_RAW = re.compile(r"(?<![\d.])\d{14}(?![\d.])")
_MIN_NAME_LEN = 6  # nomes curtos demais geram falso positivo em texto comum

VERCEL_CONFIG: dict[str, Any] = {
    "headers": [
        {
            "source": "/(.*)",
            "headers": [
                {"key": "X-Robots-Tag", "value": "noindex, nofollow"},
                {"key": "Referrer-Policy", "value": "no-referrer"},
            ],
        }
    ]
}


class LeakError(RuntimeError):
    """O build demo contém CNPJ ou nome real de fundo/gestor."""


@dataclass(frozen=True)
class Finding:
    path: str
    kind: str  # "CNPJ" ou "nome"
    sample: str


def _norm(text: str) -> str:
    nfkd = unicodedata.normalize("NFKD", text)
    no_marks = "".join(ch for ch in nfkd if not unicodedata.combining(ch))
    return " ".join(no_marks.casefold().split())


def real_names(nominal_bundle: dict[str, Any]) -> set[str]:
    """Nomes reais de fundos e gestores presentes no pacote nominal (a lista do que não pode vazar)."""
    names: set[str] = set()
    for f in nominal_bundle.get("watchlist_live", []):
        names.update(str(f.get(k) or "") for k in ("name", "gestor"))
    for n in (nominal_bundle.get("contagion_graph") or {}).get("nodes", []):
        if n.get("kind") != "cedente":
            names.update(str(n.get(k) or "") for k in ("full_name", "label", "gestor"))
    return {s.strip() for s in names if len(_norm(s)) >= _MIN_NAME_LEN and _norm(s) not in {"nao informado"}}


def find_leaks(base: str | Path, names: Iterable[str]) -> list[Finding]:
    """Varre os arquivos de texto em ``base`` atrás de CNPJs e de nomes reais (sem acento/caixa)."""
    norm_names = sorted({_norm(n) for n in names if len(_norm(n)) >= _MIN_NAME_LEN})
    findings: list[Finding] = []
    for path in sorted(Path(base).rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() == ".pdf":
            text = _pdf_text(path)
        elif path.suffix.lower() in _TEXT_SUFFIXES:
            text = path.read_bytes().decode("utf-8", errors="ignore")
        else:
            continue
        for rx in (CNPJ_FMT, CNPJ_RAW):
            m = rx.search(text)
            if m:
                findings.append(Finding(str(path), "CNPJ", m.group(0)))
                break
        if norm_names:
            hay = _norm(text)
            hit = next((n for n in norm_names if n in hay), None)
            if hit:
                findings.append(Finding(str(path), "nome", hit))
    return findings


def _pdf_text(path: Path) -> str:
    import pymupdf

    with pymupdf.open(path) as doc:
        return "\n".join(page.get_text() for page in doc)


def _demo_index(html: str) -> str:
    """Torna o aviso de demonstração visível sem JavaScript e liga o modo demo no app."""
    out = html.replace('id="demo-banner" role="note" hidden', 'id="demo-banner" role="note"', 1)
    out = out.replace('<script src="app.js', '<script>window.FIDC_DEMO = true;</script>\n  <script src="app.js', 1)
    if DEMO_BANNER_TEXT not in out or "window.FIDC_DEMO = true" not in out:
        raise ValueError("web/index.html precisa do elemento #demo-banner e do <script src=\"app.js\">")
    return out


def _prepare_out_dir(out: Path) -> None:
    if out.exists():
        if any(out.iterdir()) and not (out / _BUILD_MARKER).exists():
            raise FileExistsError(f"{out} já existe e não é um build demo; escolha outro diretório")
        shutil.rmtree(out)
    out.mkdir(parents=True)
    (out / _BUILD_MARKER).write_text("gerado por fidc-risk export-demo\n", encoding="utf-8")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def build_demo_site(
    nominal_bundle: dict[str, Any],
    codebook: CodeBook,
    out_dir: str | Path,
    web_dir: str | Path = WEB_DIR,
) -> Path:
    """Gera o site demo em ``out_dir``; falha (e apaga o build) se encontrar CNPJ ou nome real."""
    out, web = Path(out_dir), Path(web_dir)
    _prepare_out_dir(out)

    demo_bundle = anonymize_bundle(nominal_bundle, codebook)
    for rel in PUBLISHED_WEB_FILES:
        src = web / rel
        if rel == "data/observatory_bundle.json" or not src.exists():
            continue
        dst = out / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if rel == "index.html":
            dst.write_text(_demo_index(src.read_text(encoding="utf-8")), encoding="utf-8")
        else:
            shutil.copyfile(src, dst)
    write_json(out / "data" / "observatory_bundle.json", demo_bundle)
    (out / "vercel.json").write_text(json.dumps(VERCEL_CONFIG, indent=2), encoding="utf-8")

    leaks = find_leaks(out, real_names(nominal_bundle))
    if leaks:
        shutil.rmtree(out, ignore_errors=True)
        detail = "; ".join(f"{f.kind} em {Path(f.path).name}: {f.sample}" for f in leaks[:10])
        raise LeakError(f"build demo cancelado, dados identificáveis encontrados: {detail}")
    return out


_CODE_NAME = re.compile(r"(Fundo [A-Z]\d{3,}|Gestora G\d{2,}|Cedente C\d{3,}|Não informado|)")


def _non_coded_names(bundle: dict[str, Any]) -> list[str]:
    """Nomes do pacote que não são códigos de demonstração (Fundo A001, Gestora G01, Cedente C001)."""
    values: list[str] = []
    for f in bundle.get("watchlist_live", []):
        values += [str(f.get(k) or "") for k in ("name", "gestor")]
    for n in (bundle.get("contagion_graph") or {}).get("nodes", []):
        values += [str(n.get(k) or "") for k in ("label", "full_name", "gestor")]
    return sorted({v for v in values if not _CODE_NAME.fullmatch(v.strip())})


def build_pages_site(out_dir: str | Path, web_dir: str | Path = WEB_DIR) -> Path:
    """Site do GitHub Pages a partir de ``web/``, cujo pacote versionado já é o anonimizado.

    Não precisa do pacote nominal: confere que o pacote está marcado como demo, que todo nome de fundo,
    gestora e cedente é um código e que não há CNPJ em nenhum arquivo publicado. Falha e apaga o build
    se algo não bater.
    """
    out, web = Path(out_dir), Path(web_dir)
    _prepare_out_dir(out)
    for rel in PUBLISHED_WEB_FILES:
        src = web / rel
        if not src.exists():
            continue
        dst = out / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if rel == "index.html":
            dst.write_text(_demo_index(src.read_text(encoding="utf-8")), encoding="utf-8")
        else:
            shutil.copyfile(src, dst)
    (out / ".nojekyll").write_text("", encoding="utf-8")

    bundle = json.loads((out / "data" / "observatory_bundle.json").read_text(encoding="utf-8"))
    problems = [f"nome {n!r}" for n in _non_coded_names(bundle)[:10]]
    if (bundle.get("meta") or {}).get("demo") is not True:
        problems.append("pacote sem meta.demo = true")
    problems += [f"{f.kind} em {Path(f.path).name}: {f.sample}" for f in find_leaks(out, [])[:10]]
    if problems:
        shutil.rmtree(out, ignore_errors=True)
        raise LeakError("site do Pages cancelado, o pacote em web/ não está anonimizado: " + "; ".join(problems))
    return out

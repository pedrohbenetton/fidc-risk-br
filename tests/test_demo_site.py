"""Build "demo" do observatório: site estático anonimizado e varredura de vazamentos.

Inclui o teste que falha se algum CNPJ ou nome real de fundo/gestor aparecer em qualquer arquivo
publicado do demo (pacote público versionado em web/ e, quando existir, o build local dist/demo).
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from fidc_risk.config import DEMO_DIST_DIR, PRIVATE_BUNDLE_PATH, WEB_DIR
from fidc_risk.export.anonymize import CodeBook
from fidc_risk.export.demo import (
    DEMO_BANNER_TEXT,
    PUBLISHED_WEB_FILES,
    LeakError,
    build_demo_site,
    find_leaks,
    real_names,
)
from tests.test_web_bundle_anonymize import NOMES_REAIS, _bundle


def test_build_gera_site_estatico_anonimizado(tmp_path: Path) -> None:
    out = build_demo_site(_bundle(), CodeBook(rng=random.Random(3)), tmp_path / "demo")

    index = (out / "index.html").read_text(encoding="utf-8")
    assert DEMO_BANNER_TEXT in index
    assert 'id="demo-banner" role="note" hidden' not in index  # visível mesmo sem JavaScript
    assert "window.FIDC_DEMO = true" in index
    for f in ("app.js", "styles.css", "data/observatory_bundle.json", "vercel.json"):
        assert (out / f).exists(), f
    assert not (out / "data" / "private").exists()

    bundle = json.loads((out / "data" / "observatory_bundle.json").read_text(encoding="utf-8"))
    assert bundle["meta"]["demo"] is True
    assert find_leaks(out, NOMES_REAIS) == []


def test_vercel_pede_para_nao_indexar(tmp_path: Path) -> None:
    out = build_demo_site(_bundle(), CodeBook(rng=random.Random(3)), tmp_path / "demo")
    cfg = json.loads((out / "vercel.json").read_text(encoding="utf-8"))
    headers = {h["key"]: h["value"] for rule in cfg["headers"] for h in rule["headers"]}
    assert "noindex" in headers["X-Robots-Tag"]


def test_build_recusa_publicar_quando_ha_vazamento(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import fidc_risk.export.demo as demo

    monkeypatch.setattr(demo, "anonymize_bundle", lambda b, book: b)  # simula um anonimizador quebrado
    with pytest.raises(LeakError, match="CNPJ"):
        build_demo_site(_bundle(), CodeBook(rng=random.Random(3)), tmp_path / "demo")


def test_varredura_acha_cnpj_formatado_cru_e_nome(tmp_path: Path) -> None:
    (tmp_path / "a.json").write_text('{"x": "11.222.333/0001-81"}', encoding="utf-8")
    (tmp_path / "b.js").write_text("const id = '11222333000181';", encoding="utf-8")
    (tmp_path / "c.html").write_text("<p>Fidc Fictício Alfa</p>", encoding="utf-8")
    (tmp_path / "d.json").write_text('{"p": 0.12345678901234567, "n": 1234567890123}', encoding="utf-8")
    achados = find_leaks(tmp_path, ["FIDC FICTICIO ALFA"])
    tipos = sorted((Path(f.path).name, f.kind) for f in achados)
    assert tipos == [("a.json", "CNPJ"), ("b.js", "CNPJ"), ("c.html", "nome")]


def test_nomes_reais_vem_do_pacote_nominal() -> None:
    nomes = real_names(_bundle())
    assert {"FIDC FICTICIO ALFA", "FIDC FORA DA LISTA", "GESTORA FICTICIA UM LTDA"} <= nomes
    assert "Não informado" not in nomes


def _published_dirs() -> list[Path]:
    """Arquivos que vão a público: o site em web/ (pacote versionado) e o build local, se existir."""
    dirs = [WEB_DIR]
    if DEMO_DIST_DIR.exists():
        dirs.append(DEMO_DIST_DIR)
    return dirs


def _scan_published(tmp_path: Path, names: set[str]) -> list[str]:
    problems: list[str] = []
    for base in _published_dirs():
        if base == WEB_DIR:
            # Só o que é publicado: páginas, scripts, estilos e o pacote público (nunca data/private/).
            staged = tmp_path / "web"
            for rel in PUBLISHED_WEB_FILES:
                src = WEB_DIR / rel
                if src.exists():
                    dst = staged / rel
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    dst.write_bytes(src.read_bytes())
            base = staged
        problems += [f"{a.kind}: {a.path}: {a.sample}" for a in find_leaks(base, names)]
    return problems


def test_arquivos_publicados_nao_tem_cnpj(tmp_path: Path) -> None:
    """Falha se qualquer CNPJ (formatado ou só dígitos) aparecer no que é publicado."""
    problems = _scan_published(tmp_path, set())
    assert problems == [], "\n".join(problems[:20])


def test_arquivos_publicados_nao_tem_nome_real_de_fundo_ou_gestor(tmp_path: Path) -> None:
    """Falha se o nome real de um fundo ou gestor do pacote nominal aparecer no que é publicado."""
    if not PRIVATE_BUNDLE_PATH.exists():
        pytest.skip("pacote nominal ausente (data/private/ só existe na máquina que roda o pipeline)")
    names = real_names(json.loads(PRIVATE_BUNDLE_PATH.read_text(encoding="utf-8")))
    assert len(names) > 50
    problems = _scan_published(tmp_path, names)
    assert problems == [], "\n".join(problems[:20])


# ---------------------------------------------------------------- GitHub Pages

def _web_dir(tmp_path: Path, bundle: dict) -> Path:
    web = tmp_path / "web"
    for rel in PUBLISHED_WEB_FILES:
        src = WEB_DIR / rel
        if rel == "data/observatory_bundle.json" or not src.exists():
            continue
        (web / rel).parent.mkdir(parents=True, exist_ok=True)
        (web / rel).write_bytes(src.read_bytes())
    (web / "data").mkdir(parents=True, exist_ok=True)
    (web / "data" / "observatory_bundle.json").write_text(json.dumps(bundle, ensure_ascii=False), encoding="utf-8")
    return web


def test_site_do_pages_usa_o_pacote_anonimizado_versionado(tmp_path: Path) -> None:
    from fidc_risk.export.anonymize import anonymize_bundle
    from fidc_risk.export.demo import build_pages_site

    web = _web_dir(tmp_path, anonymize_bundle(_bundle(), CodeBook(rng=random.Random(3))))
    out = build_pages_site(tmp_path / "site", web_dir=web)
    index = (out / "index.html").read_text(encoding="utf-8")
    assert DEMO_BANNER_TEXT in index and "window.FIDC_DEMO = true" in index
    assert (out / ".nojekyll").exists() and (out / "metodologia.html").exists()
    assert find_leaks(out, NOMES_REAIS) == []


def test_site_do_pages_recusa_pacote_com_nome_real(tmp_path: Path) -> None:
    from fidc_risk.export.demo import build_pages_site

    web = _web_dir(tmp_path, _bundle())  # pacote nominal no lugar do anonimizado
    with pytest.raises(LeakError):
        build_pages_site(tmp_path / "site", web_dir=web)
    assert not (tmp_path / "site").exists()


def test_site_do_pages_a_partir_do_web_versionado(tmp_path: Path) -> None:
    from fidc_risk.export.demo import build_pages_site

    out = build_pages_site(tmp_path / "site")
    assert json.loads((out / "data" / "observatory_bundle.json").read_text(encoding="utf-8"))["meta"]["demo"] is True

"""Camada Bronze: Ingestão assíncrona resiliente de dados públicos da CVM e do Banco Central."""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import time
import zipfile
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import httpx

from fidc_risk.config import (
    BCB_SERIES,
    BCB_SGS_TEMPLATE,
    BRONZE_DIR,
    CVM_CAD_LEGACY_URL,
    CVM_CAD_RCVM175_URL,
    CVM_FIDC_HIST_BASE,
    CVM_FIDC_MONTHLY_BASE,
    HIST_YEARS,
    ensure_directories,
)


@dataclass
class DownloadArtifact:
    source: str
    url: str
    local_path: str
    bytes_size: int
    sha256: str
    cached: bool
    downloaded_at_utc: str


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _is_valid_zip(path: Path) -> bool:
    if not path.exists() or path.stat().st_size < 128:
        return False
    try:
        with zipfile.ZipFile(path, "r") as zf:
            return len(zf.namelist()) > 0
    except zipfile.BadZipFile:
        return False


async def _fetch_with_retry(
    client: httpx.AsyncClient,
    url: str,
    dest: Path,
    source: str,
    expect_zip: bool = False,
    semaphore: asyncio.Semaphore | None = None,
    max_retries: int = 4,
) -> DownloadArtifact:
    if dest.exists() and dest.stat().st_size > 256 and (not expect_zip or _is_valid_zip(dest)):
        return DownloadArtifact(
            source=source,
            url=url,
            local_path=str(dest.relative_to(BRONZE_DIR.parent.parent)),
            bytes_size=dest.stat().st_size,
            sha256=_sha256_file(dest),
            cached=True,
            downloaded_at_utc=datetime.now(UTC).isoformat(),
        )

    sem = semaphore or asyncio.Semaphore(4)
    async with sem:
        last_err: Exception | None = None
        for attempt in range(1, max_retries + 1):
            try:
                tmp_path = dest.with_suffix(dest.suffix + ".part")
                async with client.stream("GET", url) as resp:
                    resp.raise_for_status()
                    with tmp_path.open("wb") as f:
                        async for chunk in resp.aiter_bytes(chunk_size=1 << 18):
                            f.write(chunk)
                if expect_zip and not _is_valid_zip(tmp_path):
                    tmp_path.unlink(missing_ok=True)
                    raise ValueError(f"Arquivo ZIP inválido baixado de {url}")
                tmp_path.replace(dest)
                return DownloadArtifact(
                    source=source,
                    url=url,
                    local_path=str(dest.relative_to(BRONZE_DIR.parent.parent)),
                    bytes_size=dest.stat().st_size,
                    sha256=_sha256_file(dest),
                    cached=False,
                    downloaded_at_utc=datetime.now(UTC).isoformat(),
                )
            except Exception as exc:
                last_err = exc
                await asyncio.sleep(1.5 * attempt)
        raise RuntimeError(f"Falha ao baixar {url} após {max_retries} tentativas: {last_err}")


async def discover_monthly_cvm_files(client: httpx.AsyncClient) -> list[str]:
    """Descobre todos os arquivos mensais inf_mensal_fidc_YYYYMM.zip publicados na CVM."""
    resp = await client.get(CVM_FIDC_MONTHLY_BASE)
    resp.raise_for_status()
    html = resp.text
    matches = sorted(set(re.findall(r"(inf_mensal_fidc_\d{6}\.zip)", html)))
    if not matches:
        raise RuntimeError("Nenhum arquivo mensal inf_mensal_fidc_YYYYMM.zip encontrado na CVM.")
    return matches


async def fetch_bcb_series(
    client: httpx.AsyncClient,
    name: str,
    code: int,
    semaphore: asyncio.Semaphore,
    start_date: str = "01/01/2017",
    end_date: str = "31/12/2026",
) -> DownloadArtifact:
    dest = BRONZE_DIR / "bcb_sgs" / f"{name}_{code}.json"
    url = BCB_SGS_TEMPLATE.format(code=code, start_date=start_date, end_date=end_date)
    if dest.exists() and dest.stat().st_size > 64:
        try:
            payload = json.loads(dest.read_text(encoding="utf-8"))
            if isinstance(payload, list) and len(payload) > 10:
                return DownloadArtifact(
                    source=f"bcb_sgs:{name}",
                    url=url,
                    local_path=str(dest.relative_to(BRONZE_DIR.parent.parent)),
                    bytes_size=dest.stat().st_size,
                    sha256=_sha256_file(dest),
                    cached=True,
                    downloaded_at_utc=datetime.now(UTC).isoformat(),
                )
        except Exception:
            pass

    async with semaphore:
        last_err: Exception | None = None
        for attempt in range(1, 5):
            try:
                resp = await client.get(url, headers={"Accept": "application/json"})
                resp.raise_for_status()
                data = resp.json()
                if not isinstance(data, list) or len(data) == 0:
                    raise ValueError(f"Série BCB {code} ({name}) retornou payload vazio.")
                dest.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
                return DownloadArtifact(
                    source=f"bcb_sgs:{name}",
                    url=url,
                    local_path=str(dest.relative_to(BRONZE_DIR.parent.parent)),
                    bytes_size=dest.stat().st_size,
                    sha256=_sha256_file(dest),
                    cached=False,
                    downloaded_at_utc=datetime.now(UTC).isoformat(),
                )
            except Exception as exc:
                last_err = exc
                await asyncio.sleep(1.5 * attempt)
        raise RuntimeError(f"Falha na série BCB {name} ({code}): {last_err}")


async def run_ingestion_async() -> list[DownloadArtifact]:
    ensure_directories()
    t0 = time.perf_counter()
    limits = httpx.Limits(max_connections=8, max_keepalive_connections=6)
    timeout = httpx.Timeout(connect=20.0, read=120.0, write=20.0, pool=30.0)
    headers = {"User-Agent": "fidc-risk-br/1.0 (Academic Credit Risk Research; UFSCar)"}

    sem = asyncio.Semaphore(5)
    artifacts: list[DownloadArtifact] = []

    async with httpx.AsyncClient(limits=limits, timeout=timeout, headers=headers, follow_redirects=True) as client:
        monthly_files = await discover_monthly_cvm_files(client)
        print(
            f"[Bronze] CVM descobertos: {len(HIST_YEARS)} pacotes anuais ({HIST_YEARS[0]}–{HIST_YEARS[-1]}) "
            f"+ {len(monthly_files)} pacotes mensais ({monthly_files[0]} .. {monthly_files[-1]})"
        )

        tasks = []
        for year in HIST_YEARS:
            fname = f"inf_mensal_fidc_{year}.zip"
            tasks.append(
                _fetch_with_retry(
                    client,
                    f"{CVM_FIDC_HIST_BASE}{fname}",
                    BRONZE_DIR / "cvm_fidc_hist" / fname,
                    source=f"cvm_hist:{year}",
                    expect_zip=True,
                    semaphore=sem,
                )
            )

        for fname in monthly_files:
            tasks.append(
                _fetch_with_retry(
                    client,
                    f"{CVM_FIDC_MONTHLY_BASE}{fname}",
                    BRONZE_DIR / "cvm_fidc_monthly" / fname,
                    source=f"cvm_monthly:{fname[16:22]}",
                    expect_zip=True,
                    semaphore=sem,
                )
            )

        tasks.append(
            _fetch_with_retry(
                client,
                CVM_CAD_RCVM175_URL,
                BRONZE_DIR / "cvm_cad" / "registro_fundo_classe.zip",
                source="cvm_cad:rcvm175",
                expect_zip=True,
                semaphore=sem,
            )
        )
        tasks.append(
            _fetch_with_retry(
                client,
                CVM_CAD_LEGACY_URL,
                BRONZE_DIR / "cvm_cad" / "cad_fi.csv",
                source="cvm_cad:legacy",
                expect_zip=False,
                semaphore=sem,
            )
        )

        for series_name, code in BCB_SERIES.items():
            tasks.append(fetch_bcb_series(client, series_name, code, semaphore=sem))

        results = await asyncio.gather(*tasks)
        artifacts.extend(results)

    total_mb = sum(a.bytes_size for a in artifacts) / (1024 * 1024)
    manifest_path = BRONZE_DIR / "manifest.json"
    manifest_payload = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "elapsed_seconds": round(time.perf_counter() - t0, 2),
        "artifact_count": len(artifacts),
        "total_megabytes": round(total_mb, 2),
        "artifacts": [asdict(a) for a in artifacts],
    }
    manifest_path.write_text(json.dumps(manifest_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        f"[Bronze] Concluído em {manifest_payload['elapsed_seconds']}s | "
        f"{len(artifacts)} artefatos auditados ({total_mb:.1f} MB) -> {manifest_path}"
    )
    return artifacts


def run_ingestion() -> list[DownloadArtifact]:
    return asyncio.run(run_ingestion_async())

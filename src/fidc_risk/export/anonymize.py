"""Pseudonimização do pacote público (demo) do Observatório.

O pacote publicado (repositório, demo, páginas públicas) não pode exibir nome ou CNPJ de um fundo ao
lado do score: probabilidades atribuídas a fundos nomeados podem configurar relatório de análise
(Resolução CVM 20) e expõem gestores sem contraditório. A versão nominal fica só em data/private/.

O que acontece com cada campo:
- nome, CNPJ e gestor viram códigos estáveis ("Fundo A001", "Gestora G01", "Cedente C001"), guardados
  num mapeamento local (data/private/demo_codes.json, fora do git) para que o mesmo fundo receba o
  mesmo código em todas as execuções;
- PL exato sai e entra uma faixa ("R$ 100–250 mi"); some também tudo que devolveria o PL por conta
  (PL mensal do histórico, exposição em R$ por cedente, valor do log do PL nos fatores do score);
- setores e métricas de risco continuam reais.

Limite conhecido: é pseudonimização, não anonimização forte. Métricas reais podem ser cruzadas com os
dados abertos da CVM por quem se dispuser a isso.
"""

from __future__ import annotations

import bisect
import copy
import json
import random
from pathlib import Path
from typing import Any

# Faixas de PL em R$ milhões (limite inferior incluso).
PL_BANDS_MI: tuple[float, ...] = (10.0, 25.0, 50.0, 100.0, 250.0, 500.0, 1000.0, 2500.0)
PL_BAND_LABELS: tuple[str, ...] = (
    "até R$ 10 mi",
    "R$ 10–25 mi",
    "R$ 25–50 mi",
    "R$ 50–100 mi",
    "R$ 100–250 mi",
    "R$ 250–500 mi",
    "R$ 500 mi–1 bi",
    "R$ 1–2,5 bi",
    "acima de R$ 2,5 bi",
)

# Fatores do score cujo valor reidentifica o fundo ou o gestor (log do PL, número de fundos do gestor,
# idade do fundo, que bate com a data de constituição no cadastro da CVM).
SENSITIVE_SHAP_FEATURES: frozenset[str] = frozenset({"pl_log", "gestor_other_funds_cnt", "fund_age_months"})

_PREFIX = {"fundo": ("A", 3), "gestora": ("G", 2), "cedente": ("C", 3)}
_SEM_GESTOR = {"", "não informado", "nao informado", "none", "nan"}


def faixa_pl(pl_m: float | None) -> tuple[int, str]:
    """Devolve (ordem, rótulo) da faixa de PL em R$ milhões."""
    v = float(pl_m or 0.0)
    i = bisect.bisect_right(PL_BANDS_MI, v)
    return i, PL_BAND_LABELS[i]


class CodeBook:
    """Mapeamento persistente chave real -> código público, por tipo (fundo, gestora, cedente).

    Códigos novos são distribuídos em ordem aleatória (e não pela ordem do CNPJ ou do score) e, uma
    vez atribuídos, não mudam: o arquivo salvo é a fonte da verdade entre execuções.
    """

    def __init__(self, codes: dict[str, dict[str, str]] | None = None, rng: random.Random | None = None) -> None:
        self._codes: dict[str, dict[str, str]] = {k: dict((codes or {}).get(k, {})) for k in _PREFIX}
        self._rng = rng or random.SystemRandom()

    @classmethod
    def load(cls, path: str | Path, rng: random.Random | None = None) -> CodeBook:
        p = Path(path)
        codes = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
        return cls(codes, rng=rng)

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self._codes, ensure_ascii=False, indent=1, sort_keys=True), encoding="utf-8")

    def _format(self, kind: str, n: int) -> str:
        prefix, width = _PREFIX[kind]
        return f"{prefix}{n:0{width}d}"

    def register(self, kind: str, keys: list[str] | set[str]) -> None:
        """Atribui códigos às chaves ainda sem código, em ordem aleatória."""
        table = self._codes[kind]
        new = sorted({k for k in keys if k not in table})
        self._rng.shuffle(new)
        for k in new:
            table[k] = self._format(kind, len(table) + 1)

    def code(self, kind: str, key: str) -> str:
        table = self._codes[kind]
        if key not in table:
            self.register(kind, [key])
        return table[key]


def _gestor_key(gestor: Any) -> str | None:
    g = " ".join(str(gestor or "").split())
    return None if g.casefold() in _SEM_GESTOR else g.upper()


def anonymize_bundle(bundle: dict[str, Any], codebook: CodeBook) -> dict[str, Any]:
    """Devolve uma cópia do pacote com fundos, gestoras e cedentes trocados por códigos estáveis."""
    out = copy.deepcopy(bundle)
    watch = out.get("watchlist_live", [])
    graph = out.get("contagion_graph") or {}
    nodes = graph.get("nodes", [])

    # Registro em lote (ordem aleatória) antes de usar os códigos.
    fund_keys = {str(f["entity_id"]) for f in watch}
    fund_keys |= {str(n.get("entity_id", n["id"])) for n in nodes if n.get("kind") != "cedente"}
    gest_keys = {k for k in (_gestor_key(x.get("gestor")) for x in [*watch, *nodes]) if k}
    ced_keys = {str(n["id"]) for n in nodes if n.get("kind") == "cedente"}
    codebook.register("fundo", fund_keys)
    codebook.register("gestora", gest_keys)
    codebook.register("cedente", ced_keys)

    def gestora(raw: Any) -> str:
        k = _gestor_key(raw)
        return f"Gestora {codebook.code('gestora', k)}" if k else "Gestora não informada"

    for item in watch:
        code = codebook.code("fundo", str(item["entity_id"]))
        item["entity_id"] = code
        item["name"] = f"Fundo {code}"
        item["gestor"] = gestora(item.get("gestor"))
        item.pop("cnpj", None)
        ordem, faixa = faixa_pl(item.pop("pl_m", None))
        item["pl_faixa"], item["pl_ordem"] = faixa, ordem
        for h in item.get("history", []):
            h.pop("pl_m", None)
        for d in item.get("shap", []):
            if d.get("feature") in SENSITIVE_SHAP_FEATURES:
                d["feature_value"] = None

    id_map: dict[str, str] = {}
    for node in nodes:
        old_id = str(node["id"])
        if node.get("kind") == "cedente":
            code = codebook.code("cedente", old_id)
            node["id"] = f"C_{code}"
            node["label"] = f"Cedente {code}"
            ordem, faixa = faixa_pl(node.pop("exposure_m", None))
            node["exposicao_faixa"], node["exposicao_ordem"] = faixa, ordem
        else:
            code = codebook.code("fundo", str(node.get("entity_id", old_id)))
            node["id"] = f"F_{code}"
            node["entity_id"] = code
            node["label"] = node["full_name"] = f"Fundo {code}"
            node["gestor"] = gestora(node.get("gestor"))
            node.pop("cnpj", None)
            ordem, faixa = faixa_pl(node.pop("pl_m", None))
            node["pl_faixa"], node["pl_ordem"] = faixa, ordem
        id_map[old_id] = node["id"]

    for link in graph.get("links", []):
        link["source"] = id_map.get(link["source"], link["source"])
        link["target"] = id_map.get(link["target"], link["target"])
        link.pop("exp_m", None)  # exp_m / share_pct devolveria o PL exato do fundo

    meta = out.setdefault("meta", {})
    meta["anonymized"] = True
    meta["demo"] = True
    return out

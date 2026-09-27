"""Map only coverage-critical rare KGGEN triples to frequent relations.

The scope is deliberately limited to rare triples touching either a GS
``common_nodes`` term or a semantic-type integration anchor. DeepSeek chooses
exactly one semantically equivalent, direction-preserving frequent relation.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GRAPH = ROOT / "datasets" / "KG_GEN_kg.json"
DEFAULT_DIR = ROOT / "datasets" / "kggen_relation_mapping"
DEFAULT_CACHE = DEFAULT_DIR / "deepseek_rare_relation_decisions.cache.json"
DEFAULT_DECISIONS = DEFAULT_DIR / "rare_relation_mapping_decisions.json"
DEFAULT_OUTPUT = DEFAULT_DIR / "KG_GEN_kg_frequency_50_deepseek_mapped.json"
DEFAULT_REPORT = DEFAULT_DIR / "KG_GEN_kg_frequency_50_deepseek_mapped.summary.json"
DEFAULT_COMMON_NODES = ROOT / "datasets" / "common_nodes.xlsx"
DEFAULT_SEMANTIC_TYPES = (
    ROOT / "datasets" / "semantic_type_integrations" / "umls_nci_clean_1pct_common_terms.json"
)
SELECTION_STRATEGY = "full_frequent_relation_list_forced_choice_by_id_v2"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_term(value: object) -> str:
    return " ".join(unicodedata.normalize("NFKC", str(value)).casefold().split())


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(path)


def load_cache(path: Path, model: str, threshold: int) -> dict[str, Any]:
    if not path.exists():
        return {
            "schema_version": 2,
            "model": model,
            "frequency_threshold": threshold,
            "selection_strategy": SELECTION_STRATEGY,
            "updated_at": utc_now(),
            "decisions": {},
        }
    cache = read_json(path)
    if not isinstance(cache, dict) or not isinstance(cache.get("decisions"), dict):
        raise ValueError(f"Invalid cache: {path}")
    return cache


def create_client(api_key: str, base_url: str, timeout: float) -> Any:
    try:
        import httpx
        from openai import OpenAI
    except ImportError as error:
        raise RuntimeError("Install openai and httpx to use DeepSeek") from error
    # Ignore broken inherited proxy variables by default; pass --trust-env to opt in.
    return OpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=timeout,
        http_client=httpx.Client(timeout=timeout, trust_env=False),
    )


def parse_batch_response(
    content: str, requested: list[str], frequent: list[str]
) -> dict[str, dict[str, Any]]:
    payload = json.loads(content)
    rows = payload.get("mappings") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        raise ValueError("Response needs a JSON 'mappings' list")
    # Rare labels can harmlessly differ by case/whitespace, but the selected
    # target is an integer ID, so no model-generated target text is accepted.
    requested_by_normalized = {normalize_term(value): value for value in requested}
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        rare_raw = row.get("rare_relation")
        target_id = row.get("target_id")
        if not isinstance(rare_raw, str) or isinstance(target_id, bool):
            continue
        rare = requested_by_normalized.get(normalize_term(rare_raw))
        if isinstance(target_id, str) and target_id.strip().isdigit():
            target_id = int(target_id.strip())
        if not isinstance(target_id, int) or not 0 <= target_id < len(frequent):
            continue
        if rare is None:
            continue
        result[rare] = {
            "decision": "map",
            "target_relation": frequent[target_id],
            "target_id": target_id,
            "reason": str(row.get("reason", "")),
        }
    return result


def ask_deepseek(
    client: Any, relations: list[str], frequent: list[str], examples: dict[str, list[dict[str, str]]],
    model: str, retries: int,
) -> dict[str, dict[str, Any]]:
    system = (
        "You normalize KG relation labels. For every rare relation, choose exactly one "
        "target from the provided complete frequent-relation list. Choose the target "
        "with the closest semantic meaning and the same direction, using the examples. "
        "The frequent-relation list contains zero-based IDs. Return the selected "
        "target_id, not a target label. Do not return null, a confidence score, a "
        "decision field, or an ID outside that list. "
        "Return JSON only: {\"mappings\": [{\"rare_relation\": string, "
        "\"target_id\": integer, \"reason\": string}]}"
    )
    last_error: Exception | None = None
    accepted: dict[str, dict[str, Any]] = {}
    unresolved = list(relations)
    for attempt in range(1, retries + 1):
        try:
            request_rows = [
                {"rare_relation": relation, "examples": examples[relation]}
                for relation in unresolved
            ]
            frequent_options = [
                {"id": index, "relation": relation}
                for index, relation in enumerate(frequent)
            ]
            user = json.dumps({"frequent_relations": frequent_options, "items": request_rows}, ensure_ascii=False)
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                response_format={"type": "json_object"},
                temperature=0,
                max_tokens=3000,
            )
            content = response.choices[0].message.content or ""
            accepted.update(parse_batch_response(content, unresolved, frequent))
            unresolved = [relation for relation in unresolved if relation not in accepted]
            if not unresolved:
                return accepted
            last_error = ValueError(
                "Response omitted or invalidated relation(s): " + ", ".join(repr(value) for value in unresolved)
            )
        except Exception as error:
            last_error = error
        if attempt < retries:
            time.sleep(min(2 ** (attempt - 1), 4))
    assert last_error is not None
    raise RuntimeError(f"DeepSeek batch failed after {retries} attempts: {last_error}") from last_error


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, default=DEFAULT_GRAPH)
    parser.add_argument("--common-nodes", type=Path, default=DEFAULT_COMMON_NODES)
    parser.add_argument("--semantic-types", type=Path, default=DEFAULT_SEMANTIC_TYPES)
    parser.add_argument("--threshold", type=int, default=50)
    parser.add_argument("--model", default=os.environ.get("DEEPSEEK_MODEL", "deepseek-chat"))
    parser.add_argument("--base-url", default=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"))
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--decisions", type=Path, default=DEFAULT_DECISIONS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if args.threshold < 1 or args.batch_size < 1:
        parser.error("threshold and batch size must be positive")

    graph = read_json(args.graph)
    if not isinstance(graph, list) or not all(isinstance(row, dict) for row in graph):
        raise ValueError("Graph must be a JSON list of objects")
    counts = Counter(str(row["predicate"]) for row in graph)
    frequent = sorted(predicate for predicate, count in counts.items() if count >= args.threshold)
    common_frame = pd.read_excel(args.common_nodes)
    if "term" not in common_frame.columns:
        raise ValueError(f"{args.common_nodes} must contain a 'term' column")
    gs_terms = {normalize_term(value) for value in common_frame["term"].dropna()}
    semantic_rows = read_json(args.semantic_types)
    semantic_terms = {
        row["normalized_term"]
        for row in semantic_rows
        if isinstance(row.get("normalized_term"), str)
    }
    kept_nodes = {
        normalize_term(value)
        for row in graph
        if counts[str(row["predicate"])] >= args.threshold
        for field in ("subject", "object")
        if isinstance((value := row.get(field)), str) and value.strip()
    }
    # Only terms that would otherwise disappear need relation recovery.
    missing_gs_terms = gs_terms - kept_nodes
    missing_semantic_terms = semantic_terms - kept_nodes
    protected_terms = missing_gs_terms | missing_semantic_terms

    def is_coverage_critical(row: dict[str, Any]) -> bool:
        return (
            normalize_term(row.get("subject", "")) in protected_terms
            or normalize_term(row.get("object", "")) in protected_terms
        )

    scoped_rare_rows = [
        row for row in graph
        if counts[str(row["predicate"])] < args.threshold and is_coverage_critical(row)
    ]
    rare = sorted({str(row["predicate"]) for row in scoped_rare_rows})
    scoped_counts = Counter(str(row["predicate"]) for row in scoped_rare_rows)
    examples: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in scoped_rare_rows:
        predicate = str(row["predicate"])
        if len(examples[predicate]) < 3:
            examples[predicate].append({
                "subject": str(row.get("subject", "")),
                "predicate": predicate,
                "object": str(row.get("object", "")),
            })

    cache = load_cache(args.cache, args.model, args.threshold)
    cached: dict[str, Any] = cache["decisions"]
    pending = [
        relation for relation in rare
        if not isinstance(cached.get(relation), dict)
        or cached[relation].get("model") != args.model
        or cached[relation].get("frequency_threshold") != args.threshold
        or cached[relation].get("scope") != "gs_or_semantic_type_anchor"
        or cached[relation].get("selection_strategy") != SELECTION_STRATEGY
    ]
    if args.prepare_only:
        write_json(args.decisions, [{
            "rare_relation": relation,
            "global_occurrences": counts[relation],
            "coverage_critical_occurrences": scoped_counts[relation],
            "examples": examples[relation],
            "frequent_relation_count": len(frequent),
            "status": "cached" if relation not in pending else "pending_deepseek",
        } for relation in rare])
        print(json.dumps({
            "frequent_relations": len(frequent), "coverage_critical_rare_relations": len(rare),
            "coverage_critical_rare_triples": len(scoped_rare_rows), "pending_deepseek": len(pending),
        }, indent=2))
        return 0

    if pending and not os.environ.get("DEEPSEEK_API_KEY"):
        raise RuntimeError("DEEPSEEK_API_KEY is required for uncached rare relations")
    client = create_client(os.environ["DEEPSEEK_API_KEY"], args.base_url, args.timeout) if pending else None
    for offset in range(0, len(pending), args.batch_size):
        relations = pending[offset:offset + args.batch_size]
        print(f"DeepSeek batch {offset // args.batch_size + 1}/{math.ceil(len(pending) / args.batch_size)}", file=sys.stderr)
        decisions = ask_deepseek(client, relations, frequent, examples, args.model, args.retries)
        for relation, decision in decisions.items():
            decision.update({
                "model": args.model, "frequency_threshold": args.threshold,
                "checked_at": utc_now(), "scope": "gs_or_semantic_type_anchor",
                "selection_strategy": SELECTION_STRATEGY,
            })
            cached[relation] = decision
        cache["schema_version"] = 2
        cache["model"] = args.model
        cache["frequency_threshold"] = args.threshold
        cache["selection_strategy"] = SELECTION_STRATEGY
        cache["updated_at"] = utc_now()
        write_json(args.cache, cache)

    decision_rows = []
    for relation in rare:
        decision = cached[relation]
        decision_rows.append({
            "rare_relation": relation,
            "global_occurrences": counts[relation],
            "coverage_critical_occurrences": scoped_counts[relation],
            **decision,
        })
    write_json(args.decisions, decision_rows)
    mapped = {row["rare_relation"]: row["target_relation"] for row in decision_rows if row["decision"] == "map"}
    output = []
    for row in graph:
        predicate = str(row["predicate"])
        if counts[predicate] >= args.threshold:
            output.append(row)
        elif is_coverage_critical(row) and predicate in mapped:
            replacement = dict(row)
            replacement["predicate"] = mapped[predicate]
            replacement["old_predicate"] = predicate
            replacement["predicate_mapping_source"] = "deepseek_forced_frequent_choice"
            output.append(replacement)
    write_json(args.output, output)
    report = {
        "input_records": len(graph), "input_relations": len(counts),
        "frequency_threshold": args.threshold, "frequent_relations": len(frequent),
        "coverage_critical_rare_relations": len(rare), "rare_relations_mapped": len(mapped),
        "gs_terms_lost_after_frequency_filter": len(missing_gs_terms),
        "semantic_type_anchor_terms_lost_after_frequency_filter": len(missing_semantic_terms),
        "coverage_critical_rare_records": len(scoped_rare_rows),
        "records_kept_by_frequency": sum(1 for row in graph if counts[str(row["predicate"])] >= args.threshold),
        "rare_records_reintroduced_by_mapping": sum(
            1 for row in graph
            if is_coverage_critical(row) and str(row["predicate"]) in mapped
        ),
        "output_records": len(output), "model": args.model,
        "selection_strategy": SELECTION_STRATEGY,
    }
    write_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

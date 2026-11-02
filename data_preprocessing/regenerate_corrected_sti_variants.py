"""Rebuild every preprocessing variant with a de-duplicated STI selection.

The source graphs in ``datasets`` are read-only inputs.  All regenerated JSON
files, the revised semantic-type list, and a validation report are written to
one new output directory.  The graph filenames themselves are unchanged.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import shutil
import sys
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_preprocessing.generate_graph_variants import variants_for
from data_preprocessing.graph_variants import (
    KG_GEN_KG_NAME,
    OPTION_NO_SEMANTIC_TYPES,
    OPTION_RAW,
    UMLS_NCI_KG_NAME,
    get_graph,
)


DATASETS = ROOT / "datasets"
GRAPHS = ("GT2KG_kg", KG_GEN_KG_NAME, UMLS_NCI_KG_NAME)
DEFAULT_OUTPUT = DATASETS / "regenerated_graph_variants_corrected_sti"


def normalize(value: object) -> str:
    return " ".join(unicodedata.normalize("NFKC", str(value)).casefold().split())


def variant_filename(kg_name: str, options: Iterable[str]) -> str:
    options = tuple(sorted(set(options)))
    if options == (OPTION_RAW,):
        return f"{kg_name}_raw.json"
    if not options:
        return f"{kg_name}.json"
    return f"{kg_name}_{'_'.join(options)}.json"


def graph_row(graph_name: str, records: list[dict[str, Any]]) -> dict[str, Any]:
    predicates = Counter(str(record.get("predicate", "")) for record in records)
    nodes = {
        normalize(record[field])
        for record in records
        for field in ("subject", "object")
        if isinstance(record.get(field), str) and record[field].strip()
    }
    synthetic = [
        record for record in records
        if record.get("source") == "semantic_type_integrations" and record.get("synthetic") is True
    ]
    return {
        "graph": graph_name,
        "edges": len(records),
        "nodes_normalized": len(nodes),
        "unique_relations": len(predicates),
        "synthetic_sti_edges": len(synthetic),
        "synthetic_sti_predicates": "; ".join(sorted({str(record.get("predicate", "")) for record in synthetic})),
        "invalid_spo_records": sum(
            not all(isinstance(record.get(field), str) and record[field].strip() for field in ("subject", "predicate", "object"))
            for record in records
        ),
    }


def read_selection(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        selection = json.load(handle)
    if not isinstance(selection, list):
        raise ValueError(f"Expected a JSON list: {path}")
    return selection


def validate_graphs(output_dir: Path, selection: list[dict[str, Any]], rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Check the key STI invariants after materialisation."""
    expected = len(selection)
    by_name = {row["graph"]: row for row in rows}
    report: dict[str, Any] = {
        "selection_edges": expected,
        "full_minus_no_smnt": {},
        "synthetic_predicates": {},
        "status": "PASS",
        "problems": [],
    }
    expected_relation = {
        "GT2KG_kg": "is",
        KG_GEN_KG_NAME: "is a",
        UMLS_NCI_KG_NAME: "isa",
    }
    for kg_name in GRAPHS:
        full = by_name[kg_name]
        no_sem = by_name[f"{kg_name}_{OPTION_NO_SEMANTIC_TYPES}"]
        delta = int(full["edges"]) - int(no_sem["edges"])
        report["full_minus_no_smnt"][kg_name] = delta
        report["synthetic_predicates"][kg_name] = full["synthetic_sti_predicates"]
        if delta != expected:
            report["problems"].append(f"{kg_name}: full - no_smnt = {delta}, expected {expected}")
        if int(full["synthetic_sti_edges"]) != expected:
            report["problems"].append(
                f"{kg_name}: {full['synthetic_sti_edges']} synthetic STI edges, expected {expected}"
            )
        if full["synthetic_sti_predicates"] != expected_relation[kg_name]:
            report["problems"].append(
                f"{kg_name}: STI predicate {full['synthetic_sti_predicates']!r}, expected {expected_relation[kg_name]!r}"
            )
    if report["problems"]:
        report["status"] = "FAIL"
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--selection", type=Path, required=True, help="Revised STI JSON selection.")
    parser.add_argument("--overwrite", action="store_true", help="Allow writing into an existing output directory.")
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}. Choose a new directory or pass --overwrite."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    # Copy only raw inputs.  The JSON filenames in the new directory remain
    # exactly those used by TDGBench (e.g., GT2KG_kg.json).
    for kg_name in GRAPHS:
        source = DATASETS / f"{kg_name}_raw.json"
        target = output_dir / source.name
        if not source.exists():
            raise FileNotFoundError(f"Missing source raw graph: {source}")
        if not target.exists():
            print(f"Copying {source.name}")
            shutil.copy2(source, target)

    rows: list[dict[str, Any]] = []
    for kg_name in GRAPHS:
        print(f"\nRegenerating {kg_name}")
        for options in variants_for(kg_name):
            filename = variant_filename(kg_name, options)
            if options == (OPTION_RAW,):
                with (output_dir / filename).open(encoding="utf-8") as handle:
                    graph = json.load(handle)
            else:
                graph = get_graph(
                    kg_name=kg_name,
                    options=options,
                    save=True,
                    reuse_existing=False,
                    datasets_dir=output_dir,
                    semantic_types_path=args.selection,
                )
            if not isinstance(graph, list):
                raise TypeError(f"Expected JSON records for {filename}")
            row = graph_row(Path(filename).stem, graph)
            rows.append(row)
            print(
                f"  {filename}: {row['edges']:,} edges | {row['synthetic_sti_edges']:,} STI edges"
            )
            del graph
            gc.collect()

    selection = read_selection(args.selection)
    report = validate_graphs(output_dir, selection, rows)
    csv_path = output_dir / "regeneration_statistics.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report_path = output_dir / "regeneration_validation.json"
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    print(f"\nValidation: {report['status']}")
    print(f"Statistics: {csv_path}")
    print(f"Report: {report_path}")
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())

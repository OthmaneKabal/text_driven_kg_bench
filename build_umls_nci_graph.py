"""Command-line entry point for reproducible UMLS-NCI graph construction."""

from __future__ import annotations

import argparse
import getpass
import json
import os
from pathlib import Path

from data_preprocessing.umls_nci_graph import (
    OPTION_KEEP_INVERSE,
    OPTION_KEEP_RARE,
    OPTION_NO_SMT,
    SUPPORTED_OPTIONS,
    BuildConfig,
    DatabaseConfig,
    build_nci_graph,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a configurable UMLS-NCI graph and a JSON statistics summary. "
            "By default inverse-relation filtering, frequency filtering, and "
            "semantic-type integration are all enabled."
        )
    )
    parser.add_argument("--output", required=True, help="Output graph JSON path.")
    parser.add_argument(
        "--summary-output",
        default=None,
        help="Summary JSON path (default: <output>.summary.json).",
    )
    parser.add_argument(
        "--options",
        nargs="*",
        default=[],
        choices=sorted(SUPPORTED_OPTIONS),
        metavar="OPTION",
        help=(
            f"Treatments to skip: {OPTION_KEEP_RARE} keeps relations below the "
            f"frequency threshold; {OPTION_KEEP_INVERSE} keeps inverse RELAs; "
            f"{OPTION_NO_SMT} disables added concept-to-semantic-type isa edges."
        ),
    )
    parser.add_argument("--source", default="NCI")
    parser.add_argument(
        "--umls-release",
        default=os.getenv("UMLS_RELEASE", "unknown"),
        help="Release identifier recorded in the summary, e.g. 2024AB.",
    )
    parser.add_argument("--min-relation-frequency", type=int, default=50)
    parser.add_argument("--semantic-type-fraction", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fetch-size", type=int, default=10_000)
    parser.add_argument("--metadata-chunk-size", type=int, default=1_000)

    parser.add_argument("--db-host", default=os.getenv("UMLS_DB_HOST", "localhost"))
    parser.add_argument(
        "--db-port", type=int, default=int(os.getenv("UMLS_DB_PORT", "3306"))
    )
    parser.add_argument("--db-user", default=os.getenv("UMLS_DB_USER", "root"))
    parser.add_argument(
        "--db-password",
        default=os.getenv("UMLS_DB_PASSWORD"),
        help=(
            "Database password. Prefer UMLS_DB_PASSWORD so the secret is not "
            "stored in shell history."
        ),
    )
    parser.add_argument(
        "--db-name",
        default=os.getenv("UMLS_DB_NAME"),
        help="Database name (or UMLS_DB_NAME).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.db_name:
        raise SystemExit("--db-name or UMLS_DB_NAME is required")
    password = args.db_password
    if password is None:
        password = getpass.getpass("UMLS MySQL password: ")

    database_config = DatabaseConfig(
        host=args.db_host,
        port=args.db_port,
        user=args.db_user,
        password=password,
        database=args.db_name,
    )
    build_config = BuildConfig(
        source=args.source,
        umls_release=args.umls_release,
        min_relation_frequency=args.min_relation_frequency,
        semantic_type_fraction=args.semantic_type_fraction,
        random_seed=args.seed,
        fetch_size=args.fetch_size,
        metadata_chunk_size=args.metadata_chunk_size,
        options=tuple(args.options),
    )

    summary = build_nci_graph(
        database_config=database_config,
        output_path=Path(args.output),
        build_config=build_config,
        summary_path=args.summary_output,
    )
    concise = {
        "graph": summary["graph"],
        "summary_path": summary["summary_path"],
        "applied_treatments": summary["configuration"]["applied_treatments"],
        "statistics": {
            key: summary["statistics"][key]
            for key in (
                "nodes",
                "edges",
                "relation_types",
                "weakly_connected_components",
                "largest_component_nodes",
                "synthetic_isa_edges",
            )
        },
    }
    print(json.dumps(concise, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

# KGGEN rare-relation mapping

The script first retains KGGEN relations with frequency at least 50. It asks
DeepSeek to map only rare triples touching a GS `common_nodes` term or one of
the semantic-type integration anchors. For every scoped rare relation, it
receives the complete list of frequent relations (441 at threshold 50) and
must choose exactly one target with the closest meaning and the same direction.
There is no confidence score, threshold, lexical candidate prefilter, or
``none`` outcome. Relations are presented with zero-based IDs, and DeepSeek
returns the selected ID; this prevents a nearly-correct generated label from
being rejected.

Run a no-cost preparation pass first:

```powershell
python data_preprocessing/kggen_relation_mapping/map_rare_relations_deepseek.py --prepare-only
```

Then run the resumable mapping:

```powershell
$env:DEEPSEEK_API_KEY = "YOUR_KEY"
python data_preprocessing/kggen_relation_mapping/map_rare_relations_deepseek.py
```

The cache is saved after every DeepSeek batch. The decision file records the
chosen relation, examples and rationale for auditability. Older cached
confidence-based decisions are automatically ignored because the selection
strategy is versioned in the cache.

## Benchmark variants

After the mapping run, generate the three KGGEN variants with:

```powershell
python data_preprocessing/kggen_relation_mapping/build_kggen_variants.py
```

This writes to `datasets/`:

- `KG_GEN_kg_full.json`: all 104,223 original triples, with the 131 mapped
  predicate labels replaced wherever they occur; no frequency filtering and no
  semantic-type edges.
- `KG_GEN_kg_no_smnt.json`: frequency threshold 50 plus the 166
  coverage-critical remapped triples; no semantic-type edges.
- `KG_GEN_kg_vf.json`: `KG_GEN_kg_no_smnt` plus the 1,846 selected semantic
  type edges using `isa` (override with `--semantic-relation`).


## GT2KG benchmark variants

```powershell
python data_preprocessing/kggen_relation_mapping/build_gt2kg_variants.py
```

This writes `GT2KG_kg_full.json`, `GT2KG_kg_no_smnt.json`, and
`GT2KG_kg_vf.json` to `datasets/`. The `full` graph appends the 4,604
normalized noisy triples from the under-50 recovery file to the 50-filtered
GT2KG base; `no_smnt` is the filtered base alone; `vf` adds the selected type
edges with `isa`.

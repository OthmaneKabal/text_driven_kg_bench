# TDGBench

[![arXiv](https://img.shields.io/badge/arXiv-2605.05476-b31b1b.svg)](https://arxiv.org/abs/2605.05476)
[![Hugging Face Space](https://img.shields.io/badge/Hugging%20Face-TDG--Bench-yellow)](https://huggingface.co/spaces/othmanekabal/tdg-bench)

TDGBench is the codebase accompanying *A Unified Benchmark for Evaluating Knowledge Graph Construction Methods and Graph Neural Networks*. It provides reproducible construction of graph variants and a common evaluation protocol for text-driven knowledge graphs.

## Get started

Create an environment and install the benchmark dependencies:

```powershell
conda create -n tdg-bench python=3.10
conda activate tdg-bench
pip install -r requirements.txt
```

The UMLS-NCI reference graph is not distributed with this repository. Building it additionally requires the UMLS dependencies described below.

## 1. Build the UMLS-NCI graph

You must hold a UMLS licence and load the **UMLS 2024AB** RRF release into a relational MySQL database before running the extraction. Follow the official [UMLS Rich Release Format MySQL load guide](https://www.nlm.nih.gov/research/umls/implementation_resources/scripts/README_RRF_MySQL_Output_Stream.html) when preparing the database.

Install the extra dependency and configure the database connection. The following is PowerShell syntax:

```powershell
pip install -r requirements-umls.txt

$env:UMLS_DB_HOST = "localhost"
$env:UMLS_DB_PORT = "3306"
$env:UMLS_DB_USER = "root"
$env:UMLS_DB_PASSWORD = "<your-password>"
$env:UMLS_DB_NAME = "umls_2024ab"
```

### UMLS-NCI raw graph

This command keeps rare relations, keeps inverse relations, and does not add semantic-type `isa` edges.

```powershell
python build_umls_nci_graph.py `
  --output datasets/UMLS_nci_kg_raw.json `
  --umls-release 2024AB `
  --options kept_rare kept_inverse no_SMT
```

### UMLS-NCI default graph

This command creates `UMLS_nci_kg.json`, the fully processed reference graph. By default it removes inverse RELAs, removes relations occurring fewer than 50 times, and adds reproducibly sampled concept-to-semantic-type `isa` edges.

```powershell
python build_umls_nci_graph.py `
  --output datasets/UMLS_nci_kg.json `
  --umls-release 2024AB
```

Important parameters are:

- `--output`: destination JSON graph.
- `--source`: UMLS source vocabulary; the default is `NCI`.
- `--umls-release`: release identifier recorded with the generated graph.
- `--min-relation-frequency`: frequency threshold; the default is `50`.
- `--semantic-type-fraction`: fraction used for `isa` integration; the default is `0.01`.
- `--seed`: sampling seed; the default is `42`.
- `--options`: disables default treatments: `kept_rare`, `kept_inverse`, and `no_SMT`.

## 2. Get data and graph variants

Graph names correspond to JSON files in `datasets/`, without the `.json` suffix. TDGBench discovers these names automatically.

Use `get_graph` to load a graph as JSON records, optionally build a preprocessing variant, and optionally save that variant:

```python
from data_preprocessing.graph_variants import get_graph

graph = get_graph(
    kg_name="GT2KG_kg",
    options=["no_freq_filter"],
    save=True,
)
```

Use `TDGBench.get_data` when the graph must be converted into PyTorch Geometric data and loaders:

```python
from tdg_bench import TDGBench

data, train_loader, val_loader, test_loader, preparation = TDGBench().get_data(
    kg_name="GT2KG_kg",
    init_embd="sentence-transformers/all-MiniLM-L6-v2",
    split_path="datasets/split/split_42.json",
    options=["no_freq_filter"],
    save_variant=True,
)
```

With `options=None` or `options=[]`, TDGBench loads the stored default graph `<kg_name>.json`. The available options are:

| Option | Effect |
|---|---|
| `raw` | No preprocessing. This option is exclusive and loads `<kg_name>_raw.json`. |
| `no_freq_filter` | Keeps relations occurring fewer than 50 times. |
| `no_smnt` | Does not add semantic-type `isa` edges. |
| `with_inverse` | Keeps inverse relations. Available only for `UMLS_nci_kg`. |

Options can be combined, except `raw`. A generated variant is saved as `<kg_name>_<options>.json`; `save_variant=False` is the default.

To materialise and save every supported variant of the three benchmark graphs, run:

```powershell
python data_preprocessing/generate_graph_variants.py `
  --graphs GT2KG_kg KG_GEN_kg UMLS_nci_kg
```

The command reuses an already existing variant and writes no summary files. UMLS variants require the locally reconstructed UMLS raw graph; variants with `isa` also require `datasets/semantic_type_integrations/umls_nci_clean_1pct_common_terms.json`.

## 3. Evaluate graphs and models

`evaluate_benchmark` evaluates every requested graph, model, and hidden dimension over the Cartesian product of data splits and training-randomness seeds.

```python
from data_preprocessing.splits import PREGENERATED_SPLIT_SEEDS
from tdg_bench import TDGBench

results = TDGBench(use_classifier=True).evaluate_benchmark(
    graph_names=["GT2KG_kg", "KG_GEN_kg"],
    model_names=["GCN", "RotatEGCN_attn"],
    init_embd="sentence-transformers/all-MiniLM-L6-v2",
    split_seeds=PREGENERATED_SPLIT_SEEDS,
    random_seeds=[1, 2, 3],
    hidden_channels=[64, 128],
    epochs=100,
    patience=100,
    results_dir="results/benchmark",
    save_models=False,
)
```

Main parameters:

- `graph_names`: graph names in `datasets/`, without `.json`.
- `model_names`: registered encoder names. Current names are `GCN`, `GAT`, `RGCN`, `TransEGCN_conv`, `RotatEGCN_conv`, `TransEGCN_attn`, and `RotatEGCN_attn`.
- `init_embd`: initial node embedding model, for example `sentence-transformers/all-MiniLM-L6-v2`.
- `split_seeds`: data-split seeds. By default, TDGBench uses the ten pre-generated benchmark splits.
- `random_seeds`: independent training-randomness seeds. By default, `[1, 2, 3, 4, 5]` is used.
- `hidden_channels`: one hidden size or a list of sizes to evaluate.
- `out_channels`: encoder output size; by default it equals each hidden size.
- `epochs`, `patience`, `lr`, and `weight_decay`: training hyperparameters.
- `save_models`: saves checkpoints when `True`; metrics and result tables can still be saved with `False`.

### Add a graph

Place a graph at `datasets/<graph_name>.json`, then pass `<graph_name>` in `graph_names`. Records must provide string `subject`, `predicate`, and `object` fields. To evaluate it with the TDG gold-standard splits, every annotated common term must occur in the graph with its exact label.

### Add a GNN model

Implement the encoder in `models/` and register its builder in [build_models.py](build_models.py). The public builder interface is:

```python
from build_models import register_model

@register_model("MyGNN")
def build_my_gnn(in_channels, hidden_channels, out_channels, num_relations, **kwargs):
    return MyGNN(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        out_channels=out_channels,
        num_relations=num_relations,
        **kwargs,
    )
```

The encoder must expose `out_channels` and follow the input/output convention used by the encoders in `models/`; after registration, its unique name can be passed in `model_names`.

### No-graph baseline

The no-graph baseline uses only labelled term embeddings and an MLP. It does not load a graph JSON, relations, or edges. `kg_name` selects the matching gold-standard terms and split protocol.

```python
from data_preprocessing.splits import PREGENERATED_SPLIT_SEEDS
from tdg_bench import TDGBench

results = TDGBench(use_classifier=True).evaluate_no_graph_baseline(
    kg_name="GT2KG_kg",
    init_embds=[
        "sentence-transformers/all-MiniLM-L6-v2",
        "random_384",
    ],
    split_seeds=PREGENERATED_SPLIT_SEEDS,
    random_seeds=[1, 2, 3],
    hidden_channels=[64, 128],
    epochs=100,
    patience=100,
    results_dir="results/no_graph",
    save_models=False,
)
```

`random` uses `random_embedding_dim=384` by default; `random_<dimension>` explicitly selects the random-feature dimension. Random features are identical for every split sharing the same `random_seed`.

## 4. Reproduce paper results

The commands for reproducing the paper experiments will be added here next.

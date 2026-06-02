# Text-Driven Graph Benchmark

[![Hugging Face Space](https://img.shields.io/badge/Hugging%20Face-TDG--Bench-yellow)](https://huggingface.co/spaces/othmanekabal/tdg-bench)

This repository contains the code and data associated with the article:

**A Unified Benchmark for Evaluating Knowledge Graph Construction Methods and Graph Neural Networks**

TDG-Bench provides experimental utilities for evaluating text-driven knowledge graphs and graph neural network models on downstream node classification tasks.

> **Note**  
> TDG-Bench as an integrated Python library is coming soon.

## 1. Environment Setup

Create and activate a dedicated environment:

```bash
conda create -n tdg-bench python=3.10
conda activate tdg-bench
```

Install the project dependencies:

```bash
pip install -r requirements.txt
```

If you use a CUDA-enabled GPU, make sure that your installed PyTorch, PyG, CUDA, and driver versions are compatible.

## 2. Build the UMLS-NCI Reference Graph

To build the UMLS-NCI reference graph, you must first obtain a UMLS license and download the UMLS knowledge sources from:

https://www.nlm.nih.gov/research/umls/licensedcontent/umlsknowledgesources.html

After downloading the UMLS data, run:

```bash
python get_umls_nci_kg.py <dir_path>
```

where `<dir_path>` is the directory containing the downloaded UMLS data.

The script creates a JSON reference graph at:

```text
datasets/biomedical/reference_graph.json
```

## 3. Reproduce the Article Results

To reproduce the benchmark experiments reported in the article, run:

```bash
python experiments.py
```

The script runs the configured KG/GNN/embedding sweep and writes results under:

```text
results/
```

## 4. Custom Usage

### 4.1 Evaluate Your Own GNN Model

You can use the `TDGBench` class to evaluate a custom GNN encoder. The model factory must return a fresh model instance for each seed.

```python
from tdg_bench import TDGBench


def model_factory():
    return MyGNNEncoder(
        in_channels=768,
        hidden_channels=64,
        out_channels=64,
        num_relations=10,
    )


tdg = TDGBench(use_classifier=True)

results = tdg.evaluate(
    kg_name="GT2KG_kg",
    model_factory=model_factory,
    init_embd="michiyasunaga/BioLinkBERT-base",
    split_path="datasets/splits/umls_kg_splits/split_42.json",
    epochs=100,
    patience=100,
    lr=0.01,
    weight_decay=5e-4,
)

print(results["final_test"])
```

### 4.2 Evaluate Your Own KG With GNN Baselines

Place your graph JSON file in:

```text
datasets/
```

Then call TDG-Bench with the graph name, without the `.json` extension:

```python
from build_models import build_encoder, get_default_model_kwargs
from tdg_bench import TDGBench


kg_name = "my_knowledge_graph"
model_name = "RotatEGCN_attn"
init_embd = "michiyasunaga/BioLinkBERT-base"

tdg = TDGBench(use_classifier=True)

annotated_graph, _, _, _, _ = tdg.get_data(
    kg_name=kg_name,
    init_embd=init_embd,
    split_path="datasets/splits/my_kg_splits/split_42.json",
)

in_channels = annotated_graph.x.shape[1]
num_relations = len(set(annotated_graph.edge_type.tolist()))
extra_kwargs = get_default_model_kwargs(model_name)


def model_factory():
    return build_encoder(
        model_name=model_name,
        in_channels=in_channels,
        hidden_channels=64,
        out_channels=64,
        num_relations=num_relations,
        **extra_kwargs,
    )


results = tdg.evaluate_all(
    kg_name=kg_name,
    model_factory=model_factory,
    init_embd=init_embd,
    seeds=[42, 123, 456, 789, 2024],
    splits_dir="datasets/splits/my_kg_splits",
    epochs=100,
    patience=100,
    save_results=True,
    results_dir="results/my_kg_experiment",
)
```

## Contact

If you find an error or have suggestions to improve this benchmark, please open an issue or contact us.

## Citation

If you use this benchmark or code, please cite:

```bibtex
@article{kabal2026unified,
  title={A Unified Benchmark for Evaluating Knowledge Graph Construction Methods and Graph Neural Networks},
  author={Kabal, Othmane and Harzallah, Mounira and Guillet, Fabrice and Takeda, Hideaki and Ichise, Ryutaro},
  journal={arXiv preprint arXiv:2605.05476},
  year={2026}
}
```

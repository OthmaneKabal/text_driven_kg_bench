# semantic_type_integrations

This stage defines one annotation budget, equal to 1% of clean UMLS-NCI
concept CUIs, and samples that budget from shared non-benchmark terms using
proportional primary-semantic-type stratification.

Generate the one master list:

```powershell
python data_preprocessing/semantic_type_integrations/extract_common_semantic_types.py
```

Integrate it into any graph. The relation is deliberately explicit:

```powershell
python data_preprocessing/semantic_type_integrations/integrate_semantic_types.py `
  --graph datasets/GT2KG_kg.json `
  --semantic-types datasets/semantic_type_integrations/umls_nci_clean_1pct_common_terms.json `
  --relation isa `
  --output datasets/GT2KG_kg_with_semantic_types.json `
  --strict
```

For fair comparison, integrate the master list into UMLS-NCI's `nci_without_smt.json`,
not into a version that already contains NCI-native synthetic semantic-type edges.

# Data

| File | Contents |
|---|---|
| `gene_features.tsv` | 482 structured features per gene |
| `gene_labels.tsv` | 15 binary task labels per gene |
| `gene_labels_2021.tsv` | labels as of 2021 (temporal validation) |
| `gene_summaries.tsv` | NCBI gene summaries |
| `chembl_clinical_targets_phase23.csv` | ChEMBL Phase II/III targets (independent validation) |
| `features_llm_structured_scores.csv` | 23 attribute scores per gene |
| `features_llm_structured_scores_demo.csv` | 5-gene subset of the above |
| `prior_task_summary.csv` | class prior per task |

`features_llm_embedding.csv` is generated with `python llm/preprocess.py --mode embeddings`.

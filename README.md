# TargetSage

Code for *TargetSage: Identifying Therapeutic Target Genes with Interpretable and Robust LLM Reasoning* (NeurIPS 2026).

## Setup

```bash
pip install -r requirements.txt
```

## Data

`data/` contains the structured features and labels. Derived files are generated with (LLM steps require `AZURE_OPENAI_{CHAT,EMBED}_{ENDPOINT,KEY}`):

```bash
python scripts/fetch_gene_summaries.py --email <email>     # gene_summaries.tsv
python scripts/fetch_chembl_clinical_targets.py            # chembl_clinical_targets_phase23.csv
python scripts/agent_tools.py --stage all                  # Module 1
python scripts/prepare_agent_attrs.py                      # features_llm_structured_scores.csv
python llm/preprocess.py --mode embeddings                 # features_llm_embedding.csv
python llm/preprocess.py --mode priors                     # prior_task_summary.csv
```

## Usage

```bash
python train.py                                                  # 15 tasks, 5 seeds
python train.py --tasks task_pharos_tclin_vs_others --seeds 0    # one task, one seed
python inference.py --task task_pharos_tclin_vs_others           # genome-wide ranking
python scripts/run_baselines.py                                  # baselines
python scripts/m2_grpo_training.py                               # Module 2 (GPU)
```

## Structure

```
train.py                    Module 3: train and evaluate
inference.py                genome-wide ranking
targetsage/                 model, nnPU loss, metrics, data loading
llm/                        LLM backends, embeddings, class priors
scripts/
  agent_tools.py            Module 1
  prepare_agent_attrs.py    Module 1 attribute vocabulary
  agent_feature_pipeline.py Module 1, alternative pipeline
  m2_grpo_training.py       Module 2
  run_baselines.py          baselines
  fetch_*.py                NCBI and ChEMBL downloads
data/                       labels, attribute scores, class priors
```

## Citation

*TargetSage: Identifying Therapeutic Target Genes with Interpretable and Robust LLM Reasoning.* NeurIPS 2026.

## License

MIT

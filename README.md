# TargetSage

Code for *TargetSage: Identifying Therapeutic Target Genes with Interpretable and Robust LLM Reasoning* (NeurIPS 2026).

## Setup

```bash
pip install -r requirements.txt
```

Place `gene_features.tsv` and `features_llm_embedding.csv` in `data/`.

## Usage

```bash
python train.py                                                  # 15 tasks, 5 seeds
python train.py --tasks task_pharos_tclin_vs_others --seeds 0    # one task, one seed
python inference.py --task task_pharos_tclin_vs_others           # genome-wide ranking
python scripts/run_baselines.py                                  # baselines
python scripts/m2_grpo_training.py                               # Module 2 (GPU)
```

Module 1 requires `AZURE_OPENAI_{CHAT,EMBED}_{ENDPOINT,KEY}`:

```bash
python scripts/agent_tools.py --stage all
python scripts/prepare_agent_attrs.py
```

## Structure

```
train.py                    Module 3: train and evaluate
inference.py                genome-wide ranking
targetsage/                 model, nnPU loss, metrics, data loading
llm/                        LLM backends
scripts/
  agent_tools.py            Module 1
  prepare_agent_attrs.py    Module 1 attribute filtering
  agent_feature_pipeline.py Module 1, alternative pipeline
  fetch_gene_summaries.py   NCBI gene summaries
  m2_grpo_training.py       Module 2
  run_baselines.py          baselines
data/                       labels, attribute scores, class priors
```

## Citation

*TargetSage: Identifying Therapeutic Target Genes with Interpretable and Robust LLM Reasoning.* NeurIPS 2026.

## License

MIT

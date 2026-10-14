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

## Structure

```
train.py                    Module 3: train and evaluate
inference.py                genome-wide ranking
targetsage/                 model, nnPU loss, metrics, data loading
scripts/
  m1_agentic_profiling.py   Module 1
  m2_grpo_training.py       Module 2
  run_baselines.py          baselines
data/                       labels, attribute scores, class priors
```

## Citation

*TargetSage: Identifying Therapeutic Target Genes with Interpretable and Robust LLM Reasoning.* NeurIPS 2026.

## License

MIT

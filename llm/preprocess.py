#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TargetSage LLM Preprocessing Pipeline

Generates two LLM-derived feature files for all genes:
  1. features_llm_structured_scores.csv  — 12-dim explicit attribute scores
  2. features_llm_embedding.csv          — high-dim implicit embeddings

Also generates LLM-estimated class priors:
  3. prior_task_summary.csv              — π_median per task

Usage:
    # Generate all features using Azure backend
    python llm/preprocess.py --backend azure --gene_summaries data/gene_summaries.tsv

    # Scores only (no embeddings, faster)
    python llm/preprocess.py --backend azure --mode scores

    # Embeddings only
    python llm/preprocess.py --backend azure --mode embeddings

    # Prior estimation only
    python llm/preprocess.py --backend azure --mode priors

    # Resume from checkpoint (skips genes already processed)
    python llm/preprocess.py --backend azure --resume

    # Use local BioMistral (requires GPU ~8GB)
    python llm/preprocess.py --backend biomistral

Input gene_summaries.tsv expected columns:
    Gene_Symbol | summary
    (NCBI gene descriptions — see data/README.md for download instructions)

Azure credentials (environment variables):
    AZURE_OPENAI_CHAT_ENDPOINT, AZURE_OPENAI_CHAT_KEY
    AZURE_OPENAI_EMBED_ENDPOINT, AZURE_OPENAI_EMBED_KEY
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from llm.llm_backends import (
    AzureOpenAIBackend, BioMistralBackend, Llama3Backend,
    ATTR_KEYS,
)
from targetsage.data import TASKS, TASK_DISPLAY


# ---------------------------------------------------------------------------
# Task metadata for prior estimation
# ---------------------------------------------------------------------------
TASK_META = {
    "task_pharos_tclin_vs_others":                     {"P": 700,   "U": 18332, "family": "PHAROS",
        "definition": "Clinically validated drug targets (Tclin in PHAROS)."},
    "task_pharos_tclin_tchem_vs_others":               {"P": 2562,  "U": 16470, "family": "PHAROS",
        "definition": "Targets with clinical or chemical druggability evidence."},
    "task_triage_tier1_vs_others":                     {"P": 1399,  "U": 17633, "family": "Triage",
        "definition": "Tier 1 druggable targets (highest confidence) per Finan et al."},
    "task_triage_tier12_vs_others":                    {"P": 2048,  "U": 16984, "family": "Triage",
        "definition": "Tier 1 or Tier 2 druggable targets."},
    "task_cancer_druggability":                        {"P": 319,   "U": 18713, "family": "Cancer",
        "definition": "Curated oncology-relevant targets from DepMap."},
    "task_cancer_type_specific_target_prioritization": {"P": 297,   "U": 18735, "family": "Cancer",
        "definition": "Cancer-type-specific therapeutic targets."},
    "task_pan_cancer_target_prioritization":           {"P": 195,   "U": 18837, "family": "Cancer",
        "definition": "Pan-cancer therapeutic targets."},
    "task_T1_targets_only":                            {"P": 156,   "U": 18876, "family": "Cancer",
        "definition": "Targets of approved cancer drugs (T1)."},
    "task_T1_T2_targets":                              {"P": 620,   "U": 18412, "family": "Cancer",
        "definition": "Targets of approved or other-indication drugs (T1-T2)."},
    "task_T1_T2_T3_targets":                           {"P": 1066,  "U": 17966, "family": "Cancer",
        "definition": "Targets of approved, investigational, or experimental drugs (T1-T3)."},
    "task_sm_bucket1_vs_others":                       {"P": 777,   "U": 18255, "family": "Drug Modality",
        "definition": "Targets of approved small-molecule drugs."},
    "task_sm_bucket123_vs_others":                     {"P": 1103,  "U": 17929, "family": "Drug Modality",
        "definition": "Targets of approved or clinical-stage small molecules."},
    "task_ab_bucket1_vs_others":                       {"P": 69,    "U": 18963, "family": "Drug Modality",
        "definition": "Targets of approved antibodies."},
    "task_ab_bucket123_vs_others":                     {"P": 279,   "U": 18753, "family": "Drug Modality",
        "definition": "Targets of approved or clinical-stage antibodies."},
    "task_protac_bucket1234_vs_others":                {"P": 269,   "U": 18763, "family": "Drug Modality",
        "definition": "Targets susceptible to PROTAC-mediated degradation."},
}

TOTAL_GENES = 19032


# ---------------------------------------------------------------------------
# Backend factory
# ---------------------------------------------------------------------------
def build_backend(args):
    if args.backend == "azure":
        return AzureOpenAIBackend(
            chat_endpoint    = args.chat_endpoint,
            chat_deployment  = args.chat_deployment,
            chat_api_version = "2024-02-01",
            chat_api_key     = os.environ["AZURE_OPENAI_CHAT_KEY"],
            embed_endpoint   = args.embed_endpoint,
            embed_deployment = args.embed_deployment,
            embed_api_version= "2024-02-01",
            embed_api_key    = os.environ["AZURE_OPENAI_EMBED_KEY"],
        )
    elif args.backend == "biomistral":
        return BioMistralBackend()
    elif args.backend == "llama3":
        return Llama3Backend()
    else:
        raise ValueError(f"Unknown backend: {args.backend}")


# ---------------------------------------------------------------------------
# Step 1: Structured attribute scores
# ---------------------------------------------------------------------------
def generate_scores(backend, genes, summaries, out_path, resume, delay):
    """Generate 12-dim explicit attribute vector per gene."""

    # Load existing results if resuming
    existing = {}
    if resume and os.path.isfile(out_path):
        existing = pd.read_csv(out_path).set_index("Gene_Symbol").to_dict("index")
        print(f"  [resume] {len(existing)} genes already done")

    rows = []
    todo = [(g, s) for g, s in zip(genes, summaries) if g not in existing]
    print(f"  Generating scores for {len(todo)} genes "
          f"({len(existing)} already cached) ...")

    for i, (gene, summary) in enumerate(todo):
        attrs = backend.extract_attributes(gene, summary)
        row   = {"Gene_Symbol": gene, **attrs}
        rows.append(row)

        if (i + 1) % 50 == 0 or (i + 1) == len(todo):
            # Save incrementally
            all_rows = list(existing.values()) + rows
            pd.DataFrame(all_rows).to_csv(out_path, index=False)
            print(f"  [{i+1}/{len(todo)}] saved to {out_path}")

        if delay > 0:
            time.sleep(delay)

    # Final merge with existing
    if existing:
        df_exist = pd.DataFrame.from_dict(existing, orient="index").reset_index()
        df_exist.columns = ["Gene_Symbol"] + list(df_exist.columns[1:])
        df_new   = pd.DataFrame(rows)
        df_all   = pd.concat([df_exist, df_new], ignore_index=True)
    else:
        df_all = pd.DataFrame(rows)

    df_all = df_all.drop_duplicates(subset=["Gene_Symbol"], keep="last")
    df_all.to_csv(out_path, index=False)
    print(f"  Done. {len(df_all)} genes saved → {out_path}")
    return df_all


# ---------------------------------------------------------------------------
# Step 2: Implicit embeddings
# ---------------------------------------------------------------------------
def generate_embeddings(backend, genes, summaries, out_path, resume, batch_size):
    """Generate high-dim embedding per gene via mean-pool or API."""

    existing_genes = set()
    existing_rows  = []
    if resume and os.path.isfile(out_path):
        df_exist = pd.read_csv(out_path)
        existing_genes = set(df_exist["Gene_Symbol"].tolist())
        existing_rows  = df_exist.to_dict("records")
        print(f"  [resume] {len(existing_genes)} genes already done")

    todo_genes = [g for g in genes if g not in existing_genes]
    todo_texts = [
        f"{g}: {s}" for g, s in zip(genes, summaries) if g not in existing_genes
    ]

    if not todo_genes:
        print("  All embeddings already cached.")
        return pd.read_csv(out_path)

    print(f"  Generating embeddings for {len(todo_genes)} genes ...")

    all_vecs = []
    for i in range(0, len(todo_genes), batch_size):
        batch_genes = todo_genes[i:i+batch_size]
        batch_texts = todo_texts[i:i+batch_size]
        vecs = backend.embed(batch_texts)     # [B, d]
        for gene, vec in zip(batch_genes, vecs):
            all_vecs.append({"Gene_Symbol": gene,
                             **{f"emb_{j}": float(v) for j, v in enumerate(vec)}})
        if (i // batch_size + 1) % 10 == 0:
            print(f"  [{i+len(batch_genes)}/{len(todo_genes)}]")

    df_new  = pd.DataFrame(all_vecs)
    df_all  = pd.concat([pd.DataFrame(existing_rows), df_new], ignore_index=True)
    df_all  = df_all.drop_duplicates(subset=["Gene_Symbol"], keep="last")
    df_all.to_csv(out_path, index=False)
    dim = df_all.shape[1] - 1
    print(f"  Done. {len(df_all)} genes × {dim} dims → {out_path}")
    return df_all


# ---------------------------------------------------------------------------
# Step 3: Prior estimation
# ---------------------------------------------------------------------------
def generate_priors(backend, out_path, n_runs):
    """Prompt LLM to estimate π_median for each of the 15 tasks."""
    rows = []
    for task, meta in TASK_META.items():
        display = TASK_DISPLAY.get(task, task)
        print(f"  Estimating prior: {display} ...")

        estimates = []
        for run in range(n_runs):
            est = backend.estimate_prior(
                task_name      = display,
                family         = meta["family"],
                P              = meta["P"],
                U              = meta["U"],
                total_genes    = TOTAL_GENES,
                task_definition= meta["definition"],
            )
            estimates.append(est)
            time.sleep(0.3)

        # Average across runs
        avg = lambda k: float(np.mean([e[k] for e in estimates]))
        rows.append({
            "task":      display,
            "task_key":  task,
            "family":    meta["family"],
            "P":         meta["P"],
            "U":         meta["U"],
            "m_min":     avg("m_min"),
            "m_med":     avg("m_med"),
            "m_max":     avg("m_max"),
            "pi_min":    avg("pi_min"),
            "pi_median": avg("pi_med"),
            "pi_max":    avg("pi_max"),
            "notes":     estimates[0].get("notes", ""),
        })
        print(f"    π_median = {rows[-1]['pi_median']:.4f}")

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"  Done → {out_path}")
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Generate LLM features for TargetSage preprocessing",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Backend
    ap.add_argument("--backend", default="azure",
                    choices=["azure", "biomistral", "llama3"])

    # Azure endpoints (ignored for local backends)
    ap.add_argument("--chat_endpoint",
                    default=os.environ.get("AZURE_OPENAI_CHAT_ENDPOINT"))
    ap.add_argument("--chat_deployment",  default="gpt-4o-mini")
    ap.add_argument("--embed_endpoint",
                    default=os.environ.get("AZURE_OPENAI_EMBED_ENDPOINT"))
    ap.add_argument("--embed_deployment", default="text-embedding-3-large")

    # Input
    ap.add_argument("--gene_summaries",
                    default="data/gene_summaries.tsv",
                    help="TSV with columns: Gene_Symbol, summary")
    ap.add_argument("--gene_col",    default="Gene_Symbol")
    ap.add_argument("--summary_col", default="summary")

    # Output paths
    ap.add_argument("--scores_out",
                    default="data/features_llm_structured_scores.csv")
    ap.add_argument("--embeddings_out",
                    default="data/features_llm_embedding.csv")
    ap.add_argument("--priors_out",
                    default="data/prior_task_summary.csv")

    # Mode
    ap.add_argument("--mode", default="all",
                    choices=["all", "scores", "embeddings", "priors"],
                    help="Which feature type to generate")

    # Options
    ap.add_argument("--resume",       action="store_true",
                    help="Skip genes already in output file")
    ap.add_argument("--delay",        type=float, default=0.1,
                    help="Seconds between API calls (rate limiting)")
    ap.add_argument("--embed_batch",  type=int,   default=32,
                    help="Batch size for embedding calls")
    ap.add_argument("--prior_runs",   type=int,   default=3,
                    help="Number of LLM calls per task for prior estimation")
    ap.add_argument("--max_genes",    type=int,   default=0,
                    help="Debug: limit to first N genes (0 = all)")

    args = ap.parse_args()

    # ---- Load gene summaries ----
    print(f"[INFO] Loading gene summaries from {args.gene_summaries} ...")
    summ_df  = pd.read_csv(args.gene_summaries, sep="\t")

    # Normalise column names
    if args.gene_col not in summ_df.columns:
        summ_df = summ_df.rename(columns={summ_df.columns[0]: "Gene_Symbol"})
    else:
        summ_df = summ_df.rename(columns={args.gene_col: "Gene_Symbol"})

    if args.summary_col not in summ_df.columns:
        # Try common fallback names
        for c in ["description", "gene_summary", "full_name", "Summary"]:
            if c in summ_df.columns:
                summ_df = summ_df.rename(columns={c: "summary"})
                break
    else:
        summ_df = summ_df.rename(columns={args.summary_col: "summary"})

    summ_df = summ_df.drop_duplicates(subset=["Gene_Symbol"])
    if "summary" not in summ_df.columns:
        summ_df["summary"] = ""

    if args.max_genes > 0:
        summ_df = summ_df.head(args.max_genes)
        print(f"  [debug] limited to {args.max_genes} genes")

    genes     = summ_df["Gene_Symbol"].tolist()
    summaries = summ_df["summary"].fillna("").tolist()
    print(f"  {len(genes)} genes loaded")

    # ---- Build backend ----
    print(f"[INFO] Initialising backend: {args.backend} ...")
    backend = build_backend(args)

    # ---- Run selected mode ----
    os.makedirs("data", exist_ok=True)

    if args.mode in ("all", "scores"):
        print("\n[STEP 1] Generating structured attribute scores ...")
        generate_scores(backend, genes, summaries,
                        args.scores_out, args.resume, args.delay)

    if args.mode in ("all", "embeddings"):
        print("\n[STEP 2] Generating implicit embeddings ...")
        generate_embeddings(backend, genes, summaries,
                            args.embeddings_out, args.resume, args.embed_batch)

    if args.mode in ("all", "priors"):
        print("\n[STEP 3] Estimating LLM priors for all 15 tasks ...")
        generate_priors(backend, args.priors_out, args.prior_runs)

    print("\n[DONE]")


if __name__ == "__main__":
    main()

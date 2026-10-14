"""Module 2 inference: apply the trained policy to every gene.

Writes the 23 explicit reasoning attributes (input to Module 3) and the
reasoning chains (embedded by llm/preprocess.py into the implicit embedding).
"""
import argparse
import os
import sys

import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from m2_grpo_training import (ATTR_VOCAB, attrs_to_vec, build_gene_profile,
                              extract_attrs, make_prompt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--adapter", default="results/rl_grpo_v3/rl_best")
    ap.add_argument("--gene_summaries", default="data/gene_summaries.tsv")
    ap.add_argument("--agent_attrs", default="results/agent_tools/agent_attributes_filtered.csv")
    ap.add_argument("--labels", default="data/gene_labels.tsv")
    ap.add_argument("--scores_out", default="data/features_llm_structured_scores.csv")
    ap.add_argument("--reasoning_out", default="data/reasoning_chains.tsv")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_tokens", type=int, default=512)
    ap.add_argument("--max_genes", type=int, default=0)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map=device, trust_remote_code=True)
    model = PeftModel.from_pretrained(model, args.adapter).eval()

    summaries = pd.read_csv(args.gene_summaries, sep="\t")
    g2s = dict(zip(summaries["Gene_Symbol"].astype(str), summaries["summary"].astype(str)))
    agent_attrs = pd.read_csv(args.agent_attrs).set_index("Gene_Symbol").to_dict("index")
    genes = pd.read_csv(args.labels, sep="\t")["Gene_Symbol"].astype(str).tolist()
    if args.max_genes:
        genes = genes[:args.max_genes]

    scores, reasoning = [], []
    for i in range(0, len(genes), args.batch_size):
        batch = genes[i:i + args.batch_size]
        prompts = [make_prompt(g, build_gene_profile(g, g2s, agent_attrs), tokenizer)
                   for g in batch]
        enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True,
                        max_length=1536).to(device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=args.max_tokens, do_sample=False,
                                 pad_token_id=tokenizer.pad_token_id)
        for g, ids in zip(batch, out[:, enc["input_ids"].shape[1]:]):
            attrs, chain = extract_attrs(tokenizer.decode(ids, skip_special_tokens=True))
            scores.append([g] + list(attrs_to_vec(attrs)))
            reasoning.append((g, chain))
        print(f"  {i + len(batch)}/{len(genes)}", flush=True)

    pd.DataFrame(scores, columns=["Gene_Symbol"] + ATTR_VOCAB).to_csv(args.scores_out, index=False)
    pd.DataFrame(reasoning, columns=["Gene_Symbol", "reasoning"]).to_csv(
        args.reasoning_out, sep="\t", index=False)
    print(f"Saved {args.scores_out} and {args.reasoning_out}")


if __name__ == "__main__":
    main()

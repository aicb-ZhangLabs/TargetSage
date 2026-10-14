#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TargetSage LLM Backends

Three backends:
  AzureOpenAIBackend  — gpt-4o-mini (chat) + text-embedding-3-large (embed)
                        Split endpoints: EastUS2 (chat) / WestUS (embed)
  BioMistralBackend   — BioMistral/BioMistral-7B  (local, 8-bit)
  Llama3Backend       — meta-llama/Meta-Llama-3-8B-Instruct (local, 8-bit)

All backends expose the same interface:
    backend.extract_attributes(gene, summary)  → dict[str, float]  (12-dim)
    backend.embed(texts)                        → np.ndarray [N, d]
    backend.estimate_prior(task_name, ...)      → dict

Key implementation notes:
  - Local models use do_sample=True, temperature=0.6, min_new_tokens=80
    to prevent greedy decoding artifacts and early EOS before JSON closes
  - extract_first_json handles: markdown fences, BioMistral numbered-key
    prefixes, trailing commentary after }, and JSON-preceded preamble
  - Collapse detection only rejects truly constant vectors (std < 1e-4),
    not biologically sparse ones (e.g. PTEN tumor suppressor)
"""

import re
import json
import time
import warnings
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
FEATURE_SYSTEM_PROMPT = """\
You are a senior drug discovery scientist.

Your task is to extract calibrated, continuous-valued therapeutic attributes
for target identification.

Calibration rules:
- Values must lie in [0.0, 1.0]; avoid exact 0.0 or 1.0 unless the property
  is a strict binary definition (e.g. kinase vs non-kinase).
- For well-established targets prefer high but non-saturating values
  (e.g. 0.85–0.95 instead of 1.0).
- Safety concern reflects known class-effect or mechanism-based toxicities,
  not drug approval success.
- Use the full dynamic range; values will be compared across ~19k genes.

Return STRICT JSON only (no markdown, no extra text).\
"""

# Stronger JSON constraint for local models
BIO_JSON_SYSTEM_PROMPT = (
    FEATURE_SYSTEM_PROMPT
    + "\n\nCRITICAL: Output must be a single valid JSON object ONLY. "
    "No markdown fences, no commentary, no trailing text. "
    "The first character MUST be '{' and the last MUST be '}'."
)

FEATURE_USER_TEMPLATE = """\
Analyze this gene for therapeutic target identification:

Gene Symbol: {gene_symbol}
Gene Summary: {gene_summary}

Estimate these attributes. ALL values must be in [0.0, 1.0] and rounded to \
EXACTLY 2 decimal places (e.g. 0.85, 0.03, 0.50). Do NOT use scientific \
notation or more than 2 decimal places.

1. druggability_score: Overall druggability potential
2. secreted_probability: Secreted protein likelihood
3. membrane_localization: Membrane localization
4. enzyme_activity: Enzyme activity
5. kinase_family: Kinase membership
6. gpcr_family: GPCR membership
7. ion_channel: Ion channel
8. oncogenic_driver: Oncogene potential
9. tumor_suppressor: TSG potential
10. pathway_centrality: Pathway importance
11. clinical_evidence: Clinical evidence strength
12. safety_concern: Known safety risk

Return ONLY this JSON (no other text):
{{"druggability_score": 0.XX, "secreted_probability": 0.XX, \
"membrane_localization": 0.XX, "enzyme_activity": 0.XX, \
"kinase_family": 0.XX, "gpcr_family": 0.XX, "ion_channel": 0.XX, \
"oncogenic_driver": 0.XX, "tumor_suppressor": 0.XX, \
"pathway_centrality": 0.XX, "clinical_evidence": 0.XX, \
"safety_concern": 0.XX}}\
"""

PRIOR_SYSTEM_PROMPT = (
    "You are a senior drug discovery scientist and target identification expert. "
    "You reason about PU learning label incompleteness. "
    "Return STRICT JSON only (no extra text)."
)

PRIOR_USER_TEMPLATE = """\
We consider TOTAL = {total_genes} human protein-coding genes.

Task: {task_name} | Family: {family}
Observed positives: |P| = {P}. Unlabeled: |U| = {U}.
Lower bound for true prevalence: LB = {LB:.6f}.

Definition: {task_definition}

Output multipliers: pi_min = m_min * LB, pi_median = m_med * LB, pi_max = m_max * LB
Constraints: 1.00 <= m_min <= m_med <= m_max <= 3.50

Return STRICT JSON only:
{{"m_min": <float>, "m_med": <float>, "m_max": <float>, "notes": "<1-2 sentences>"}}\
"""

# Ordered attribute keys (order matters for downstream numpy arrays)
ATTR_KEYS = [
    "druggability_score", "secreted_probability", "membrane_localization",
    "enzyme_activity",    "kinase_family",         "gpcr_family",
    "ion_channel",        "oncogenic_driver",      "tumor_suppressor",
    "pathway_centrality", "clinical_evidence",     "safety_concern",
]


# ---------------------------------------------------------------------------
# JSON extraction
# ---------------------------------------------------------------------------
def clamp01(x, eps: float = 0.02) -> float:
    try:
        v = float(x)
    except Exception:
        v = 0.5
    return float(max(eps, min(1.0 - eps, v)))


def _normalize_quotes(text: str) -> str:
    """Replace curly/smart quotes (common BioMistral artifact) with ASCII quotes."""
    return (
        text
        .replace('\u201c', '"').replace('\u201d', '"')   # " "
        .replace('\u2018', "'").replace('\u2019', "'")   # ' '
        .replace('\u00ab', '"').replace('\u00bb', '"')   # << >>
    )


def extract_first_json(text: str) -> dict:
    """
    Robust extraction of ATTR_KEY values from raw LLM output.

    Primary:  regex scan for each key — immune to curly quotes, invisible
              prefix chars (SentencePiece ▁, BOM), preamble objects,
              truncation, and extra-long floats.
    Fallback: brace-match + json.loads on the best candidate.
    """
    text = _normalize_quotes((text or "").strip())

    # ---- Primary: regex key-value scan ----
    result = {}
    for key in ATTR_KEYS:
        m = re.search(
            r'["\u201c]' + re.escape(key) + r'["\u201d]\s*:\s*'
            r'(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)',
            text,
        )
        if m:
            result[key] = clamp01(m.group(1))

    if len(result) >= 10:           # tolerate up to 2 missing keys
        for k in ATTR_KEYS:
            result.setdefault(k, 0.5)
        return result

    # ---- Fallback: brace-match + json.loads ----
    if "```" in text:
        text = re.sub(r"```[a-zA-Z]*\n?", "", text).replace("```", "").strip()

    candidates: list = []
    i = 0
    while i < len(text):
        if text[i] != "{":
            i += 1
            continue
        depth, start = 0, i
        while i < len(text):
            if   text[i] == "{": depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    candidates.append(text[start:i + 1])
                    i += 1
                    break
            i += 1

    # prefer candidate that contains a known attribute key
    ordered = (
        [c for c in candidates if any(k in c for k in ATTR_KEYS)] +
        [c for c in candidates if not any(k in c for k in ATTR_KEYS)]
    )

    last_exc = None
    for cand in ordered:
        try:
            return json.loads(cand)
        except Exception as e:
            last_exc = e

    raise ValueError(
        f"Could not extract attributes "
        f"(regex {len(result)}/12, json fallback: {last_exc}). "
        f"Text[:300]: {text[:300]}"
    )


# ---------------------------------------------------------------------------
# Base backend
# ---------------------------------------------------------------------------
class BaseLLMBackend(ABC):
    """Abstract base — subclasses implement _call_llm() and embed()."""

    def extract_attributes(
        self,
        gene: str,
        summary: str,
        retries: int = 3,
        debug: bool = False,
    ) -> Dict[str, float]:
        """
        Prompt for 12 therapeutic attributes and return clamped dict.
        Falls back to 0.5 on all retries failing.
        """
        system = BIO_JSON_SYSTEM_PROMPT
        user   = FEATURE_USER_TEMPLATE.format(
            gene_symbol=gene,
            gene_summary=summary or "No summary available.",
        )

        for attempt in range(retries):
            raw = self._call_llm(system, user)
            if debug:
                print(f"  [RAW try={attempt + 1}] {raw[:500]}")
            try:
                data   = extract_first_json(raw)
                result = {k: clamp01(data.get(k, 0.5)) for k in ATTR_KEYS}
                vals   = np.array([result[k] for k in ATTR_KEYS], dtype=np.float32)
                if float(np.std(vals)) < 1e-4:
                    raise ValueError("Collapsed output: near-constant vector")
                return result
            except Exception as e:
                if debug:
                    print(f"  [Retry {attempt + 1}] {e}")
                # Stronger prompt on retry
                system = "Return ONLY a JSON object. No notes, no text.\n" + BIO_JSON_SYSTEM_PROMPT
                time.sleep(0.2 * 2 ** attempt)

        warnings.warn(f"[{self.__class__.__name__}] attribute extraction failed for {gene}")
        return {k: 0.5 for k in ATTR_KEYS}

    def estimate_prior(
        self,
        task_name: str,
        family: str,
        P: int,
        U: int,
        total_genes: int,
        task_definition: str,
        retries: int = 3,
    ) -> dict:
        """Prompt for LLM-informed class prior (pi_min, pi_med, pi_max)."""
        LB   = P / total_genes
        user = PRIOR_USER_TEMPLATE.format(
            total_genes=total_genes, task_name=task_name, family=family,
            P=P, U=U, LB=LB, task_definition=task_definition,
        )
        for attempt in range(retries):
            raw = self._call_llm(PRIOR_SYSTEM_PROMPT, user)
            try:
                d = extract_first_json(raw)
                return {
                    "m_min":  float(d.get("m_min", 1.0)),
                    "m_med":  float(d.get("m_med", 1.5)),
                    "m_max":  float(d.get("m_max", 2.0)),
                    "notes":  str(d.get("notes", "")),
                    "pi_min": float(d.get("m_min", 1.0)) * LB,
                    "pi_med": float(d.get("m_med", 1.5)) * LB,
                    "pi_max": float(d.get("m_max", 2.0)) * LB,
                }
            except Exception as e:
                time.sleep(0.2 * 2 ** attempt)

        return {"m_min": 1.0, "m_med": 1.5, "m_max": 2.0, "notes": "fallback",
                "pi_min": LB, "pi_med": 1.5 * LB, "pi_max": 2.0 * LB}

    @abstractmethod
    def _call_llm(self, system: str, user: str) -> str:
        """Call the model and return raw string."""

    @abstractmethod
    def embed(self, texts: List[str]) -> np.ndarray:
        """Return L2-normalised embeddings [N, d]."""

    @staticmethod
    def _l2_normalize(mat: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        return mat / np.maximum(norms, 1e-12)


# ---------------------------------------------------------------------------
# Azure OpenAI  (split endpoint: chat EastUS2, embed WestUS)
# ---------------------------------------------------------------------------
class AzureOpenAIBackend(BaseLLMBackend):
    """
    Azure OpenAI with two separate deployments:
      - Chat  : EastUS2  (gpt-4o-mini)
      - Embed : WestUS   (text-embedding-3-large)

    Credentials: AZURE_OPENAI_{CHAT,EMBED}_{ENDPOINT,KEY}
    """

    def __init__(
        self,
        chat_endpoint: str,
        chat_deployment: str,
        chat_api_version: str,
        chat_api_key: str,
        embed_endpoint: str,
        embed_deployment: str,
        embed_api_version: str,
        embed_api_key: str,
        temperature: float = 0.2,
        max_tokens: int = 512,
    ):
        try:
            from openai import AzureOpenAI
        except ImportError:
            raise ImportError("pip install openai>=1.0")

        self.chat_client = AzureOpenAI(
            azure_endpoint=chat_endpoint,
            api_version=chat_api_version,
            api_key=chat_api_key,
        )
        self.embed_client = AzureOpenAI(
            azure_endpoint=embed_endpoint,
            api_version=embed_api_version,
            api_key=embed_api_key,
        )
        self.chat_deployment  = chat_deployment
        self.embed_deployment = embed_deployment
        self.temperature      = temperature
        self.max_tokens       = max_tokens

    def _call_llm(self, system: str, user: str) -> str:
        resp = self.chat_client.chat.completions.create(
            model=self.chat_deployment,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return resp.choices[0].message.content or ""

    def embed(self, texts: List[str]) -> np.ndarray:
        resp  = self.embed_client.embeddings.create(
            model=self.embed_deployment, input=texts
        )
        items = sorted(resp.data, key=lambda x: x.index)
        mat   = np.array([e.embedding for e in items], dtype=np.float32)
        return self._l2_normalize(mat)


# ---------------------------------------------------------------------------
# BioMistral-7B  (local, 8-bit quantised)
# ---------------------------------------------------------------------------
class BioMistralBackend(BaseLLMBackend):
    """
    Local BioMistral-7B (domain-specific, pre-trained on PubMed + PMC).

    Uses Mistral [INST] prompt format (no system role in chat template).
    Requires ~8 GB VRAM with 8-bit quantisation.

    Generation fixes vs greedy decoding:
      - do_sample=True, temperature=0.6  →  avoids numbered-prefix artifacts
      - min_new_tokens=80                →  prevents early EOS before } closes
    """

    MODEL_NAME = "BioMistral/BioMistral-7B"

    def __init__(self, device: str = "auto"):
        try:
            import torch
            from transformers import (
                AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
            )
        except ImportError:
            raise ImportError(
                "pip install torch transformers accelerate bitsandbytes"
            )

        print(f"[BioMistral] Loading {self.MODEL_NAME} (8-bit) ...")
        quant = BitsAndBytesConfig(load_in_8bit=True)
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.MODEL_NAME, device_map=device, quantization_config=quant
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self._device = next(self.model.parameters()).device
        print(f"[BioMistral] Ready on {self._device}")

    def _build_prompt(self, system: str, user: str) -> str:
        return f"<s>[INST] {system}\n\n{user} [/INST]"

    def _call_llm(self, system: str, user: str) -> str:
        import torch
        prompt = self._build_prompt(system, user)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self._device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=600,   # 12 keys × ~15 chars + overhead; 400 can truncate
                min_new_tokens=80,
                do_sample=True,
                temperature=0.6,
                top_p=0.95,
                repetition_penalty=1.08,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        gen = out[0, inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(gen, skip_special_tokens=True).strip()

    def embed(self, texts: List[str]) -> np.ndarray:
        """Mean-pool last hidden state → L2-normalise."""
        import torch
        vecs = []
        for t in texts:
            inp = self.tokenizer(
                t, return_tensors="pt", truncation=True, max_length=512
            ).to(self._device)
            with torch.no_grad():
                h = self.model(
                    **inp, output_hidden_states=True
                ).hidden_states[-1]
                v = h.mean(dim=1).squeeze().float().cpu().numpy()
            vecs.append(v.astype(np.float32))
        mat = np.stack(vecs, axis=0)
        return self._l2_normalize(mat)


# ---------------------------------------------------------------------------
# Llama-3-8B-Instruct  (local, 8-bit quantised)
# ---------------------------------------------------------------------------
class Llama3Backend(BaseLLMBackend):
    """
    Local Meta-Llama-3-8B-Instruct (general-purpose).

    Uses apply_chat_template with system/user roles.
    Requires ~8 GB VRAM with 8-bit quantisation.
    """

    MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"

    def __init__(self, device: str = "auto"):
        try:
            import torch
            from transformers import (
                AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
            )
        except ImportError:
            raise ImportError(
                "pip install torch transformers accelerate bitsandbytes"
            )

        print(f"[Llama3] Loading {self.MODEL_NAME} (8-bit) ...")
        quant = BitsAndBytesConfig(load_in_8bit=True)
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.MODEL_NAME, device_map=device, quantization_config=quant
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self._device = next(self.model.parameters()).device
        print(f"[Llama3] Ready on {self._device}")

    def _call_llm(self, system: str, user: str) -> str:
        import torch
        msgs   = [{"role": "system", "content": system},
                  {"role": "user",   "content": user}]
        prompt = self.tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self._device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=600,   # 12 keys × ~15 chars + overhead
                min_new_tokens=80,
                do_sample=True,
                temperature=0.6,
                top_p=0.95,
                repetition_penalty=1.05,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        gen = out[0, inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(gen, skip_special_tokens=True).strip()

    def embed(self, texts: List[str]) -> np.ndarray:
        """Mean-pool last hidden state → L2-normalise."""
        import torch
        vecs = []
        for t in texts:
            inp = self.tokenizer(
                t, return_tensors="pt", truncation=True, max_length=512
            ).to(self._device)
            with torch.no_grad():
                h = self.model(
                    **inp, output_hidden_states=True
                ).hidden_states[-1]
                v = h.mean(dim=1).squeeze().float().cpu().numpy()
            vecs.append(v.astype(np.float32))
        mat = np.stack(vecs, axis=0)
        return self._l2_normalize(mat)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def build_backend(name: str, **kwargs) -> BaseLLMBackend:
    """
    Convenience factory.

    Args:
        name   : "azure" | "biomistral" | "llama3"
        kwargs : forwarded to the backend constructor

    Example:
        backend = build_backend("azure",
            chat_endpoint="...", chat_deployment="gpt-4o-mini",
            chat_api_version="2024-02-01", chat_api_key="...",
            embed_endpoint="...", embed_deployment="text-embedding-3-large",
            embed_api_version="2024-02-01", embed_api_key="...")
    """
    backends = {
        "azure":      AzureOpenAIBackend,
        "biomistral": BioMistralBackend,
        "llama3":     Llama3Backend,
    }
    if name not in backends:
        raise ValueError(f"Unknown backend '{name}'. Choose: {list(backends)}")
    return backends[name](**kwargs)

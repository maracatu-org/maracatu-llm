---
license: apache-2.0
language:
  - pt
library_name: transformers
pipeline_tag: text-generation
tags:
  - text-generation
  - causal-lm
  - portuguese
  - brazilian-portuguese
  - from-scratch
  - llama
  - gqa
datasets:
  - wikimedia/wikipedia
  - CulturaX
model-index:
  - name: maracatu-80m
    results:
      - task:
          type: text-generation
          name: Text Generation
        metrics:
          - type: perplexity
            value: 21.34
            name: Validation Perplexity (holdout PT-BR, 3.27M tokens)
---

# Maracatu AI — Model Family 🥁

Open-weight Brazilian Portuguese language models trained from scratch. Apache 2.0. Architecture compatible with `LlamaForCausalLM`.

**Organization:** [huggingface.co/maracatu-labs](https://huggingface.co/maracatu-labs) | **Code:** [github.com/maracatu-labs/maracatu](https://github.com/maracatu-labs/maracatu)

---

## Released Models

| Model | Parameters | Val Perplexity | Corpus | Released |
|---|---|---|---|---|
| [Maracatu-20M](https://huggingface.co/maracatu-labs/maracatu-20m) | 17M | 23.81 | Wikipedia PT (~550M tok) | April 2026 |
| [Maracatu-80M](https://huggingface.co/maracatu-labs/maracatu-80m) | 87.80M | 21.34 | Wikipedia + Gutenberg + CulturaX-PT (~1.60B tok) | April 2026 |

---

## Current Release: Maracatu-80M

Maracatu-80M is a causal language model trained from scratch on 1.60B tokens of curated Brazilian Portuguese text. It reaches validation perplexity **21.34** on a 3.27M-token holdout, outperforming the public Tucano-160M baseline (reported ~22) with approximately half the parameters.

### Architecture

Llama-style decoder-only transformer. Compatible bit-for-bit with `transformers.LlamaForCausalLM` (`max_abs_diff=0.0` validated in `scripts/export_hf.py`).

| Hyperparameter | Value |
|---|---|
| Total parameters | 87.80M |
| Non-embedding parameters | 75.52M |
| `num_hidden_layers` | 12 |
| `hidden_size` | 768 |
| `num_attention_heads` | 12 |
| `num_key_value_heads` | 4 (GQA, 3:1 ratio) |
| `intermediate_size` (SwiGLU) | 2048 |
| `max_position_embeddings` | 1024 |
| `vocab_size` | 16000 |
| `rope_theta` | 10000.0 |
| Normalization | RMSNorm (eps 1e-5) |
| Weight tying | Yes (embed ↔ lm_head) |
| Bias in Linear | No |

### Training Summary

| Item | Value |
|---|---|
| Hardware | NVIDIA RTX 3060 12GB, single GPU, self-hosted |
| Training time | 22h 31min |
| Total iterations | 200,000 |
| Tokens seen | ~1.64B |
| Peak LR | 2.5e-4 (cosine → 2.5e-5, 4k warmup) |
| Precision | bf16 autocast (fp32 weights + optimizer) |
| Best val loss | 3.0163 (step ~190,000) |
| Final val perplexity | 21.34 |
| Throughput | ~20,200 tok/s |

### Corpus v2

| Source | License | Tokens |
|---|---|---|
| Wikipedia PT (`20231101.pt`) | CC BY-SA 3.0 | ~550M |
| Project Gutenberg PT (24 curated works) | Public Domain | ~150M |
| CulturaX-PT filtered (1.49M docs) | ODC-BY 1.0 | ~900M |

Corpus SHA-256: `a1000e873bfcae0d2229ecc9b329f0befe8ad73913e79e58f14a1f3a48ef7e58`

Filtering: MinHash LSH dedup (Jaccard 0.85), PII regex (CPF/email/CEP/phone BR), language heuristic, byte-level dedup. No raw Common Crawl. No CC BY-NC sources.

### Downstream Benchmarks (zero-shot, lm-evaluation-harness 0.4.11)

| Task | Score | Stderr | Random |
|---|---|---|---|
| ENEM Challenge | 20.27% | ±1.06% | 20% (5-MCQ) |
| ASSIN Entailment | 29.08% | ±0.72% | ~33% (3-class) |
| ASSIN Paraphrase | 52.42% | ±0.79% | 50% (binary) |

**Honest reading**: ENEM is at chance. ASSIN Entailment is below chance (selection bias). ASSIN Paraphrase is ~3 stderr above chance. Maracatu-20M's published 60.52% on ASSIN Paraphrase is lower in 80M (52.42%) — possibly harness-version differences, possibly MCQ variance in small base models. Pretrain improvements show up in **generation fluency** (perplexity 23.81 → 21.34), not MCQ accuracy. Use this model as a generation backbone, not a question-answering system.

### Limitations

- **Small model by 2026 standards.** Factual retrieval is unreliable; hallucination is expected. Do not deploy in production without further training.
- **Lowercase only.** Tokenizer normalizes all text via `nmt_nfkc_cf`.
- **No instruction tuning.** Base model only. Fine-tune before interactive use.
- **Context: 1,024 tokens.** Longer inputs are truncated.
- **No safety fine-tuning.** Research use only.
- **MCQ benchmarks at or near random chance.** ENEM, ASSIN Entailment indistinguishable from guessing. ASSIN Paraphrase modestly above chance. Base models at this scale do not perform reasoning tasks without further training.

---

## How to Use

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

tokenizer = AutoTokenizer.from_pretrained("maracatu-labs/maracatu-80m", use_fast=False)
model = AutoModelForCausalLM.from_pretrained("maracatu-labs/maracatu-80m")
model.eval()

inputs = tokenizer("O Brasil é", return_tensors="pt")
with torch.no_grad():
    out = model.generate(
        **inputs,
        max_new_tokens=100,
        temperature=0.8,
        top_k=50,
        do_sample=True,
    )
print(tokenizer.decode(out[0], skip_special_tokens=True))
```

---

## Roadmap

| Release | Parameters | Status |
|---|---|---|
| Maracatu-20M | 17M | Released April 2026 |
| **Maracatu-80M** | **87.80M** | **Released April 2026** |
| Maracatu-800M | ~800M | Planned H2 2026 |
| Maracatu-8B | ~8B | Planned 2027 |
| Maracatu-80B | ~80B | North Star |

---

## License

[Apache License 2.0](https://github.com/maracatu-labs/maracatu/blob/main/LICENSE) — code, weights, and tokenizer.

---

## Citation

```bibtex
@misc{anzileiro2026maracatu80m,
  author    = {Anzileiro, Anderson},
  title     = {Maracatu-80M: An Open-Weight Brazilian Portuguese Language Model},
  year      = {2026},
  publisher = {Hugging Face},
  url       = {https://huggingface.co/maracatu-labs/maracatu-80m},
}
```

# Experiment log

Chronological record of every Maracatu training run. Each experiment is a Markdown file with date, commit, configuration, metrics, qualitative samples and analysis.

**Naming convention:** `YYYY-MM-DD-<runname>.md`

## Motivation

LLM training runs are hard to compare over time without structured notes. This log serves to:

1. **Reproducibility**: given the git rev and the config, reproduce the result
2. **Comparison**: assess whether an architecture/hyperparameter change improved anything
3. **Learning**: document what worked, what didn't and why

## Index

| Date | Run | Architecture | Iters | Best val loss | File |
|---|---|---|---|---|---|
| 2026-04-18 | `maracatu-tiny-test` v1 | GPT-2-like (LayerNorm, GELU, abs pos emb) | 5,000 | 6.7514 | [tiny-v1-gpt2](2026-04-18-tiny-v1-gpt2.md) |
| 2026-04-18 | `maracatu-tiny-test` v2 | Llama-like (RMSNorm, SwiGLU, RoPE) | 5,000 | 6.6408 | [tiny-v2-llama](2026-04-18-tiny-v2-llama.md) |
| 2026-04-19 | `maracatu-tiny-long` | Llama-like (same tiny) | 50,000 | 6.3949 | [tiny-long](2026-04-19-tiny-long.md) |

## Template

New experiment: copy [`_TEMPLATE.md`](_TEMPLATE.md) and fill it in.

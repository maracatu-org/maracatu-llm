# tiny-v2-llama (2026-04-18)

**Git revision:** `766b9366195537d155c259e14893328356012e12`
**Config:** [`configs/maracatu_tiny.yaml`](../../configs/maracatu_tiny.yaml) (Llama-like version)
**Hardware:** MacBook Pro M2 Pro, 16 GB RAM, MPS backend
**Goal:** Validate the migration to a Llama-style architecture (RMSNorm + RoPE + SwiGLU) and compare it against the GPT-2-like baseline ([tiny-v1-gpt2](2026-04-18-tiny-v1-gpt2.md)) at the same parameter and compute order.

## Setup

- **Architecture:** Llama-like: RMSNorm (no mean-centering, no bias), RoPE (HF "rotate-half" convention), SwiGLU MLP (`down(silu(gate) * up)`), weight tying, `bias=False` everywhere, GQA support (here `num_key_value_heads = num_attention_heads`, so no GQA yet).
- **Compatibility:** state_dict aligned with Hugging Face's `LlamaForCausalLM`; weights load directly via `AutoModelForCausalLM.from_pretrained` without conversion.
- **Params:** 7.31M total / 3.21M non-embedding.
- **Dataset:** `wikimedia/wikipedia` config `20231101.pt` (CC BY-SA 4.0). 979,492 articles, ~593M tokens in the training split + 6M in the val split (same `tokens.npy` as v1).
- **Tokenizer:** SentencePiece BPE 16k, no changes from v1.

## Relevant hyperparameters

| Field | Value |
|---|---|
| `hidden_size` | 256 |
| `num_hidden_layers` | 4 |
| `num_attention_heads` | 4 |
| `num_key_value_heads` | 4 |
| `intermediate_size` | 704 (~2.67× `hidden_size`, compensates for SwiGLU having 3 projections) |
| `max_position_embeddings` | 256 |
| `rms_norm_eps` | 1e-5 |
| `rope_theta` | 10000 |
| `attention_dropout` | 0.0 |
| `tie_word_embeddings` | true |
| `batch_size` | 32 |
| `learning_rate` | 5e-4 (warmup 200, cosine → 5e-5) |
| `max_iters` | 5,000 |
| `warmup_iters` | 200 |

## Results

### Loss curve (evals every 250 steps)

| step | train | val |
|---|---|---|
| 250 | 7.032 | 7.082 |
| 500 | 6.835 | 6.942 |
| 750 | 6.833 | 6.882 |
| 1000 | 6.740 | 6.874 |
| 1250 | 6.731 | 6.806 |
| 1500 | 6.680 | 6.833 |
| 1750 | 6.708 | 6.778 |
| 2000 | 6.694 | 6.812 |
| 2250 | 6.649 | 6.700 |
| 2500 | 6.679 | 6.740 |
| 2750 | 6.652 | 6.697 |
| 3000 | 6.654 | 6.683 |
| 3500 | 6.616 | **6.641** |
| 4000 | 6.583 | 6.681 |
| 4250 | 6.586 | 6.667 |
| 4500 | 6.600 | 6.671 |
| 4750 | 6.579 | 6.679 |

### Final metrics

- **Best val loss:** `6.6408` (step 3500) → perplexity ≈ 766
- **Total time:** 18.5 min
- **Average throughput:** ~36,600 tok/s on MPS
- **Training tokens processed:** ~40M out of 593M available (7%, undertrained)

## Qualitative samples

`temperature=0.8`, `top_k=50`, `seed=42`.

### Prompt: `"O Brasil é"`

```
o brasil é o novo brasil onde o campeonato britânico é um dos estados unidos. o campeão de
futebol-metragem de 2010 foi a mais alta e a temporada de 2020. o clube foi o primeiro-americano
de estúdio de ouro, com o time-americano de futebol de
```

### Prompt: `"Machado de Assis escreveu"`

```
machado de assis escreveu o "o último livro de seu trabalho dos estados unidos". o diretor de
saúde do mundo de 1998 foi lançado em 29 de novembro de 1990 em paris, e em novembro de 200
```

### Prompt: `"A capital de Pernambuco é"`

```
a capital de pernambuco é a primeira e possui a maior população da região da antiga cidade de
são pedro. a população de 57,00 km2, de 81 km2 e sua 0,0 km2, de acordo com o censo de
```

## Analysis

**Comparison with v1 (GPT-2-like):**

| Metric | v1 (GPT-2) | v2 (Llama) | Δ |
|---|---|---|---|
| Best val loss | 6.7514 | **6.6408** | -0.1106 (-1.6%) |
| Time | 14.7 min | 18.5 min | +26% |
| Throughput | 46k tok/s | 36k tok/s | -22% |
| Params (non-emb) | 3.2M | 3.2M | equal |

**Interpretation:**
- Modest but real improvement in val loss at the same step budget, with an arch that does more ops per pass. Consistent with theory: RoPE + SwiGLU pay off more in longer runs; at 5k iters (undertrained) the compounded effect has barely started.
- Throughput drop is expected: SwiGLU has 3 linear projections vs 2 for a GELU FFN; RoPE adds ops per head; RMSNorm wins a little but doesn't offset the rest.

**Qualitative:** the most visible jump was on the prompt `"A capital de Pernambuco é"`:
- v1: drifted into "band gave a show", totally out of context
- v2: stayed in geographic context: "has the largest population of the region of the old city", "km2", "census"

Still wrong on the fact (Recife), but the model understood that "capital of X" is a geographic construct. Suggestive that RoPE may be helping with long-range dependency relations within the context.

**Good signs:**
- Loss dropped from 9.68 → 6.64 (better than v1)
- Small train/val gap (~0.06), no overfitting
- Best val at step 3500, not 4750: cosine decay was already tightening, and the model kept improving on train but not on val, a slight ceiling signal for this budget

**Next steps validated by this experiment:**
1. The new arch works: keep using it for the 20M
2. The -1.6% gap at 5k iters may compound to -5 to -10% at 50k iters (speculative)
3. Move on to the 20M with no new architectural changes

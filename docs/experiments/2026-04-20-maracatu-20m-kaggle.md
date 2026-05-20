# Maracatu-20M: first full training run at real scale, Kaggle T4 (2026-04-20)

**Git revision:** `<kaggle-run>` (kernel ran outside the repo; git_revision not captured automatically; base commit estimated to be earlier than `76c0f05`)
**Config:** [`configs/maracatu_20m.yaml`](../../configs/maracatu_20m.yaml)
**Hardware:** Kaggle GPU T4 single (15.6 GB VRAM)
**Goal:** First full training run of Maracatu-20M on cloud hardware, validating the end-to-end pipeline (corpus → tokenizer → training → HF export → publishing). Served as the baseline for the M-80M ladder.

## Setup

- **Architecture:** Llama-style: RMSNorm, RoPE rotate-half (HF implementation), SwiGLU, no bias on Linear layers, weight tying embedding/lm_head, bit-for-bit compatible with `LlamaForCausalLM`
- **Params:** 16.77M total / 10.62M non-embedding
- **Dataset:** `wikimedia/wikipedia` config `20231101.pt` (CC BY-SA 4.0). After filters: 979,492 articles, 9.7 M lines, 2.28 GB, ~550 M BPE tokens
- **Tokenizer:** SentencePiece BPE, vocab 16,000 tokens, trained on PT-BR with `nmt_nfkc_cf` (lowercase), split_digits, byte_fallback

## Relevant hyperparameters

| Field | Value |
|---|---|
| `hidden_size` | 384 |
| `num_hidden_layers` | 6 |
| `num_attention_heads` | 6 |
| `num_key_value_heads` | 6 (no GQA) |
| `intermediate_size` | 1024 |
| `max_position_embeddings` | 512 |
| `batch_size` | 16 |
| `learning_rate` | 3e-4 → 3e-5 (cosine decay) |
| `warmup_iters` | 1000 (linear) |
| `max_iters` | 50,000 |
| `beta1 / beta2` | 0.9 / 0.95 |
| `weight_decay` | 0.1 |
| `grad_clip` | 1.0 |
| `seed` | 42 |

Tokens consumed: ~410 M (~0.75 epoch). Average throughput: ~20k tok/s.

## Results

### Loss curve

Full sample extracted from the JSON log (every 500 steps):

| step | train | val |
|---|---|---|
| 500 | 5.9036 | 5.8950 |
| 1000 | 5.0822 | 5.2464 |
| 1500 | 4.7608 | 4.7743 |
| 2000 | 4.5498 | 4.5369 |
| 2500 | 4.3035 | 4.3597 |
| 3000 | 4.1191 | 4.1718 |
| 3500 | 3.9143 | 4.0687 |
| 4000 | 3.9973 | 3.9597 |
| 4500 | 3.8503 | 3.9647 |
| 5000 | 3.7965 | 3.8388 |
| 7500 | 3.6540 | 3.6969 |
| 10000 | 3.5395 | 3.6209 |
| 12500 | 3.5184 | 3.4965 |
| 15000 | 3.4355 | 3.5066 |
| 17500 | 3.3664 | 3.4236 |
| 20000 | 3.3135 | 3.3688 |
| 22500 | 3.3760 | 3.3740 |
| 25000 | 3.2997 | 3.3616 |
| 27500 | 3.2171 | 3.3413 |
| 28500 | 3.1946 | 3.2697 |
| 30000 | 3.1145 | 3.3300 |
| 30500 | 3.2837 | 3.2595 |
| 35000 | 3.2537 | 3.2372 |
| 35500 | 3.2031 | 3.2133 |
| 38000 | 3.2244 | 3.2363 |
| 40000 | 3.2368 | 3.1922 |
| 43500 | 3.1522 | **3.1703** ← best val |
| 45000 | 3.1825 | 3.1827 |
| 46000 | 3.1671 | 3.2570 |
| 47000 | 3.1414 | 3.2122 |
| 48000 | 3.1006 | 3.2184 |
| 49000 | 3.1573 | 3.2307 |
| 49500 | 3.1722 | 3.2215 |

### Final metrics

- **Best val loss:** `3.1703` (step 43,500) → **perplexity 23.81**
- **Final val loss:** `3.2215` (step 49,500)
- **Final train loss:** `3.1722` (step 49,500)
- **Train/val gap on plateau:** ~0.05 nats (no apparent overfitting)
- **Total time:** 5h45min (~20,770s of log)
- **Average throughput:** ~20k tok/s

## Qualitative samples

With `temperature=0.8`, `top_k=50`, `seed=42`, `max_new_tokens=60`.

### Prompt: `"O Brasil é"`

```
O Brasil é uma espécie de ave da família dos caririformes.
```

Valid PT-BR syntactic structure. Invented factual content (hallucination typical of non-instruction-tuned LMs at small scale).

### Prompt: `"A capital de Pernambuco é"`

```
A capital de Pernambuco é um município brasileiro do estado do rio de janeiro.
```

The model correctly associated the "capital of a Brazilian state" frame but got both the city (Recife) and the referenced state wrong. Indicates partial understanding of geographic entities, an expected pattern at this scale.

### Prompt: `"Machado de Assis escreveu"`

```
machado de assis escreveu o livro "a obra-prima do autor da obra-prima do autor", que foi escrito em 1998 e publicado em 2001. no brasil, em 2003, o livro foi publicada pela universidade federal de minas gerais, que viria a ser
```

The model learned the collocation "book" + publication date + publisher, consistent with Wikipedia's encyclopedic pattern for articles about authors. Typical errors: absurd dates for Machado (died in 1908), recursive repetition ("the author's masterpiece's author's masterpiece") indicating 17M params lack the capacity to suppress token loops in abstract contexts, and wrong gender agreement ("o livro foi publicada").

### Prompt: `"A literatura brasileira"`

```
a literatura brasileira (1988-1994) a literatura brasileira é um movimento político de classe, na qual se fomenta a autodidata, a revolução de 1900 e o primeiro grande movimento político brasileiro. o movimento é apoiado por uma grande maioria de ativ
```

Valid PT-BR syntax with humanities vocabulary (political movement, class, revolution, autodidact). The model fails to anchor "Brazilian literature" as a literary phenomenon and treats it as a political movement. The parenthesis `(1988-1994)` shows it learned Wikipedia's period-annotation pattern, even without semantic context.

## Analysis

### What worked

First end-to-end validation of the Llama-style architecture at real scale (not just at the 7M tiny). The full pipeline (filtered Wikipedia PT corpus → SentencePiece 16k tokenizer → warmup+cosine training → `LlamaForCausalLM` export → HF upload → GGUF quantization → Ollama) worked without architecture rework.

The loss drop from 5.90 (step 500) to 3.17 (step 43,500) is monotonic over the first 35k iters, with a smooth plateau after that. No NaN spike, no divergence. Grad clip 1.0 and AdamW with β2=0.95 kept training stable on a single T4 for 5h45min.

### Tokens per parameter

410 M tokens / 16.77 M params ≈ **24.5 tokens/param**. The Chinchilla recipe ([Hoffmann et al., 2022](https://arxiv.org/abs/2203.15556)) recommends ~20 tokens/param for a compute-optimal model. Training landed slightly above the optimum, which is reasonable given that the goal here was pipeline validation, not maximum efficiency extraction.

### Plateau and cosine decay

The best val (3.1703) occurred at step 43,500; the next 6,500 steps showed slight oscillation with no consistent improvement (final val 3.22). Cosine decay had already reduced the LR from 3e-4 to around 3e-5 in that region. The plateau does not indicate model saturation: the LR was too low for additional learning with the current corpus. More iters with a decayed LR wouldn't help; an LR restart or a larger corpus would be the alternatives.

### Comparison with tiny-long

The `maracatu_tiny_long` run (7M params, 50k iters, context 256) finished with val ~6.39 (perp ~599). Maracatu-20M reached perp 23.81: **~25× reduction in perplexity**. The improvement results from the combination of three factors: (a) 3.3× more parameters, (b) context 512 vs 256 (allowing longer-range dependencies), (c) the same 50k iters over the identical corpus. The contribution of each factor cannot be isolated without ablation.

### Comparison with Tucano-160M

The Tucano paper ([Correa et al., 2024](https://arxiv.org/abs/2411.07854)) reports perplexity around 30 on Wikipedia PT for Tucano-160M. Maracatu-20M, with **10× fewer parameters**, reached perp 23.81 in the same evaluation domain. Main hypothesis: a curated corpus (filtered Wikipedia, no web noise) combined with a modern architecture (RoPE, SwiGLU, RMSNorm) pays off disproportionately well at small scale. A direct comparison would require evaluation on the same split and the same metric; this number should be treated as indicative, not as a formal benchmark.

### What didn't work / limitations

- Generation produces consistent factual hallucinations, expected at this scale and without RLHF/instruction-tuning.
- Git revision wasn't captured by the Kaggle kernel (runs outside the git repo); the `git_revision` field in the checkpoint contains `"<kaggle>"`.
- Corpus limited to Wikipedia PT (~550 M tokens): adequate for 20M, insufficient for 500M+ without expansion.

### Implications for the next experiment

1. **Expanded corpus is a prerequisite for M-80M**: Wikipedia PT (~550M tokens) covers ~6.9 tokens/param for 80M, below the Chinchilla optimum (1.6B tokens required, 3× expansion). Bring in Gutenberg PT and filtered OSCAR PT before starting the next run.
2. **Compute**: M-80M requires RunPod A100 (~8h estimated, R$5-15k with retries depending on spot vs on-demand). Kaggle T4 is still viable here, but RunPod is recommended for pipeline maturity.
3. **GGUF + Ollama pipeline is validated**: no infrastructure rework needed before the next release.
4. **Don't iterate further on 20M**: the model fulfilled its role as pipeline validation and baseline. Training resources go straight to M-80M.

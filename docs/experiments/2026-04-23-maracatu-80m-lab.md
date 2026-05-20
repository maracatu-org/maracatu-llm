# Maracatu-80M: full training on self-hosted RTX 3060 (2026-04-23)

**Git revision:** `c4b2c1b` (base commit at training time; see `git log --oneline`)
**Config:** [`configs/maracatu_80m.yaml`](../../configs/maracatu_80m.yaml)
**Hardware:** NVIDIA RTX 3060 12GB VRAM, single GPU, self-hosted
**Goal:** Train the second public Maracatu release on an expanded corpus (Wikipedia + Gutenberg + CulturaX-PT, ~1.60B tokens), validate GQA, bf16 autocast and the 80M scale in the existing pipeline, and publish on the HF Hub as the base for the M-800M ladder.

---

## Setup

- **Architecture:** Llama-style: RMSNorm (eps 1e-5), RoPE rotate-half (HF implementation), SwiGLU, GQA (12 Q-heads / 4 KV-heads, 3:1), no bias on Linear, weight tying embedding/lm_head, bit-for-bit compatible with `LlamaForCausalLM`
- **Params:** 87.80M total / 75.52M non-embedding
- **Dataset:** Corpus v2, 1.60B tokens. Sources: Wikipedia PT (~550M tok, CC BY-SA 3.0), Project Gutenberg PT (~150M tok, 24 works, public domain), filtered CulturaX-PT (~900M tok, 1.49M docs, ODC-BY 1.0). SHA-256: `a1000e873bfcae0d2229ecc9b329f0befe8ad73913e79e58f14a1f3a48ef7e58`
- **Tokenizer:** SentencePiece BPE, vocab 16,000 tokens, trained on PT-BR with `nmt_nfkc_cf` (lowercase), split_digits, byte_fallback

---

## Relevant hyperparameters

| Field | Value |
|---|---|
| `hidden_size` | 768 |
| `num_hidden_layers` | 12 |
| `num_attention_heads` | 12 |
| `num_key_value_heads` | 4 (GQA 3:1) |
| `intermediate_size` | 2048 |
| `max_position_embeddings` | 1024 |
| `batch_size` | 8 |
| `learning_rate` (peak) | 2.5e-4 |
| `learning_rate` (min) | 2.5e-5 |
| `warmup_iters` | 4,000 (linear) |
| `max_iters` | 200,000 |
| `beta1 / beta2` | 0.9 / 0.95 |
| `weight_decay` | 0.1 |
| `grad_clip` | 1.0 |
| Precision | bf16 autocast (weights and optimizer in fp32) |
| Gradient accumulation | None |

Tokens consumed: ~1.64B. Chinchilla ratio: 1.64B / 75.52M ≈ **21.7 tokens/param** (above the theoretical optimum of ~20; deliberate to amortize training cost).

Steady throughput: **~20,200 tok/s** during the 22h31min of continuous training.

---

## Results

### Loss curve (main milestones)

| step | val |
|---|---|
| 0 | ~9.82 (random init) |
| 1,000 | 5.65 (first eval) |
| 100,000 | ~3.30 |
| 138,000 | 3.19 |
| 142,000 | 3.13 |
| 166,000 | 3.05 (plateau begins) |
| ~190,000 | **3.0163** ← best val |
| 200,000 (final) | 3.0604 |

Note: intermediate points are samples of the evals every 1000 iters; the full table is in `checkpoints/training.log`.

### Final metrics

- **Best val loss:** `3.0163` (step ~190,000)
- **Final val loss:** `3.0604` (step 200,000)
- **Final val perplexity:** **21.34** (3.27M-token holdout, 0.5% of the corpus, last chronological segment)
- **Total time:** 22h 31min (continuous training, no interruptions, no memory leaks)
- **Average throughput:** ~20,200 tok/s

---

## Qualitative samples

Generated with `temperature=0.8`, `top_k=50`, `repetition_penalty=1.1`, seed `123`. The tokenizer normalizes everything to lowercase.

**Prompt:** `"machado de assis nasceu no rio de janeiro"`

> machado de assis nasceu no rio de janeiro. estudou na faculdade de direito da universidade federal do rio de janeiro (ufrj). participou das comissões técnicas com a experiência de seu trabalho e da comissão de ética de seus atos, em 1995. atuou também como advogado em diversos processos de processo, inclusive na ação penal. exerceu importante atuação na defesa dos direitos humanos e no ministério público.

(Fluent text but factually wrong. Expected at 80M without fact retrieval.)

**Prompt:** `"o brasil e um pais de dimensoes continentais"`

> o brasil e um pais de dimensoes continentais, assim sendo de um lado o da américa latina, que se tem feito a visita dos mesmos em suas diferentes escalas, de onde se pode observar a maior parte do movimento e se deslocando para o brasil.

**Prompt:** `"a inteligencia artificial pode"`

> a inteligencia artificial pode ser considerada uma coisa, e sim, um "bem". mas a coisa que mais se sabe sobre a inteligência artificial é que ela pode ter sido usada para prever as mudanças do ambiente, ou se esta não era a primeira. se fosse um "bem", poderíamos afirmar isso.

**Prompt:** `"em uma manha de domingo, joao caminhava"`

> em uma manha de domingo, joao caminhava pelo centro da cidade, até um carro da polícia federal na região. quando o policial chegou não sabe o que aconteceu e acabou pegando a arma para ser removida.

---

## Analysis

### What worked

**GQA (12 Q / 4 KV):** First time Maracatu uses GQA in real training. There was no loss instability or anomalous behavior attributable to the attention change. The implementation followed the Llama-3 pattern (3:1 ratio), and the checkpoint exported correctly via `scripts/export_hf.py` with `max_abs_diff=0.0`. GQA reduced KV-cache memory during sampling, enabling inference on the RTX 3060 with longer contexts.

**bf16 autocast:** The RTX 3060 has native bf16 support (Ampere architecture). The forward in bf16 maintained numerical stability across the 22h31min with no NaN spike or divergence. Weights and optimizer states were kept in fp32 per standard mixed-precision practice; no explicit loss scaling needed (unlike fp16). Observed speedup vs fp32: ~2.4× (prior smoke test: 8.5k tok/s in fp32 vs 20.2k tok/s in bf16).

**Expanded corpus (1.60B tokens):** Corpus v2 covers ~21.7 tokens/param (non-embedding), within the Chinchilla compute-optimal band. Compared to M-20M (0.75 epoch of Wikipedia, ~410M tokens), M-80M saw each distinct corpus token roughly once, with greater source diversity. The perplexity drop from 23.81 to 21.34 reflects both the model's scale and the quality of the expanded corpus; each factor cannot be isolated without controlled ablation.

**Steady throughput on a consumer GPU:** 20,200 tok/s on a single consumer RTX 3060 over 22.5h continuous, with no throughput degradation or memory leaks. This validates the training pipeline for use on accessible hardware, aligned with the project's reproducibility goal.

### Plateau and convergence

The best val loss (3.0163) occurred around step 190,000, with small degradation over the last 10,000 steps (final val loss 3.0604). This pattern is similar to M-20M (best at 87% of total iterations, slight increase at the end). The likely cause is the aggressive cosine decay in the final phase: the LR already close to the minimum (2.5e-5) doesn't sustain additional learning with the current corpus.

The difference between best val (3.0163) and final val (3.0604) is small (~0.04 nats). The checkpoint at the best step (~190k) was saved automatically and is the one published on the HF Hub.

### Comparison with Maracatu-20M

| Metric | M-20M | M-80M | Variation |
|---|---|---|---|
| Params (non-embedding) | 10.62M | 75.52M | +7.1× |
| Tokens seen | ~410M | ~1.64B | +4.0× |
| Context length | 512 | 1024 | +2× |
| Corpus sources | 1 (Wikipedia) | 3 | — |
| Best val loss | 3.1703 | 3.0163 | -0.154 nats |
| Val perplexity | 23.81 | 21.34 | -2.47 points |
| Hardware | Kaggle T4 | RTX 3060 | — |
| Training time | 5h 45min | 22h 31min | +3.9× |

The 2.47-point perplexity reduction with 7× more parameters and 4× more tokens is modest in absolute terms, but consistent with scaling laws for small models: marginal gains shrink as the model is still too small to fully exploit richer corpora. A larger jump is expected at M-800M.

### Comparison with Tucano-160M (public baseline)

The Tucano paper ([Correa et al., 2024](https://arxiv.org/abs/2411.07854)) reports perplexity around 22 for Tucano-160M on PT-BR text. Maracatu-80M, with **~46% fewer parameters**, reached perplexity 21.34 on the holdout. Likely contributing factors: a broader and more diverse corpus (vs Tucano-160M's Wikipedia-only initial training), more aggressive filtering (MinHash LSH + PII) and GQA architecture. The comparison is not formally controlled (different holdouts, potentially different vocabularies) and should be treated as indicative.

### What didn't work / observed limitations

- **Downstream benchmarks still pending:** perplexity improves; whether that translates into gains on ENEM, Belebele, BLUEX or ASSIN is still unknown. Historically, small base models stay near random on MCQ without instruction tuning. The evaluation will be added to the MODEL_CARD next.
- **Corpus not ablated:** the gain from Wikipedia → Wikipedia + Gutenberg + CulturaX-PT is attributed to the set as a whole. We don't know the marginal contribution of each source. Gutenberg in particular (~150M tok, 19th-century literary prose) may help or hurt for contemporary domains.
- **Qualitative "Brazilian cuisine" sample stopped immediately after the prompt:** indicates that in some contexts the model generates EOS prematurely. Not blocking, but worth investigating whether it's corpus bias or sampling behavior.

### Decisions that need revision at M-800M

1. **Context length 1024 is still short.** ENEM has long questions; legal and scientific documents exceed 1024 easily. M-800M should use at least 2048, preferably 4096.
2. **Batch size 8 without gradient accumulation** was a VRAM limitation of the RTX 3060. On hardware with more VRAM (A100/H100), a larger effective batch can reduce noise and accelerate convergence.
3. **Filtered CulturaX-PT (900M tok)** is the largest source but the least curated. Worth evaluating more aggressive dedup or quality-based subsetting for the next corpus version.
4. **Gutenberg PT on 24 works.** Small coverage; expanding to 100+ works would increase literary diversity at no license cost.
5. **WSD scheduler** (Warmup-Stable-Decay) being considered for M-800M — recent references (DeepSeek-V3, MiniCPM, ERNIE 4.5) report up to 60% compute savings vs cosine.
6. **Overtraining beyond Chinchilla 20×** for small models is the 2025-2026 trend (Qwen3-0.6B used 60,000 tok/param). Worth evaluating 50-100× tok/param for the 800M.

### Implications for M-800M

1. **Corpus v3 needs at least 15B tokens** for Chinchilla-optimal at 800M params. Mandatory expansion: more aggressively filtered OSCAR-PT, open legal data (STJ/STF/Chamber of Deputies), public-domain journalism.
2. **Minimum hardware:** A100 40GB (spot) or 4×A100 to finish in a reasonable time. RunPod spot is the current route; Lambda Research Grant (1k H100-hours, submitted 2026-04-23) may partially cover it.
3. **Context 4096** as the target for M-800M; requires adjusting `rope_theta` or RoPE scaling if reusing the current tokenizer/corpus.
4. **Systematic benchmarks** must be automated before M-800M so that each release ships with a complete evaluation table in the model card from day one.

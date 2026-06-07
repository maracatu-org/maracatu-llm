# 🥁 Maracatu-20M

> A Brazilian Portuguese causal language model, trained from scratch. Open weights, Apache 2.0.

Maracatu-20M is a 17M-parameter decoder-only transformer trained from scratch on Brazilian Portuguese Wikipedia. It is the first public checkpoint of the [Maracatu AI](https://github.com/maracatu-labs/maracatu) project — an open effort to build Portuguese-language LLMs with full transparency over architecture, data, and training.

This is a **base model** (completion only). It continues text — it is not a chat assistant and does not follow instructions.

---

## Quick start

```bash
# pull the default quantization (Q4_K_M, ~11 MB)
ollama pull whereisanzi/maracatu-20m

# run with a prompt
ollama run whereisanzi/maracatu-20m "O Brasil é"

# inspect model metadata
ollama show whereisanzi/maracatu-20m
```

The model outputs lowercase text only — this is expected; the tokenizer normalizes all input to lowercase.

### Specific quantizations

```bash
ollama pull whereisanzi/maracatu-20m:q5_k_m
ollama pull whereisanzi/maracatu-20m:q8_0

ollama run whereisanzi/maracatu-20m:q8_0 "A literatura brasileira é"
```

---

## Available quantizations

| Tag | Method | File size | Recommended for |
|---|---|---|---|
| `latest` / `q4_k_m` | Q4_K_M | ~11 MB | General use; best size/quality tradeoff |
| `q5_k_m` | Q5_K_M | ~13 MB | Slightly higher fidelity, still fast |
| `q8_0` | Q8_0 | ~18 MB | Evaluation, debugging, max precision |

All three run comfortably on CPU. The model is small enough that quantization differences are perceptible but minor at this parameter count.

---

## About this model

### Architecture

Llama-style decoder-only transformer (RMSNorm, RoPE, SwiGLU, no bias in linear layers, weight tying).

| Hyperparameter | Value |
|---|---|
| Total parameters | 17M (16.77M) |
| Non-embedding parameters | 10.62M |
| Layers | 6 |
| Hidden size | 384 |
| Attention heads | 6 |
| Intermediate size (SwiGLU) | 1024 |
| Context length | 512 tokens |
| Vocabulary | 16,000 (SentencePiece BPE, lowercase, split_digits) |

### Training data

| Property | Value |
|---|---|
| Source | Wikipedia PT (`wikimedia/wikipedia`, snapshot `20231101.pt`) |
| License | CC BY-SA 4.0 |
| Articles | 979,492 (after filters + dedup) |
| Corpus size | 2.28 GB |
| Tokens | ~550M BPE tokens |

### Training

| Item | Value |
|---|---|
| Framework | PyTorch |
| Hardware | Kaggle T4 (single GPU, 15.6 GB VRAM) |
| Total iterations | 50,000 |
| Tokens seen | ~410M (~0.75 epoch) |
| Batch size | 16 |
| Optimizer | AdamW (β₁=0.9, β₂=0.95, weight_decay=0.1) |
| Learning rate | 3e-4 → 3e-5 (warmup + cosine decay) |
| Total training time | 5h 45min |

### Evaluation

| Metric | Value | Step |
|---|---|---|
| Best validation perplexity | **23.81** | 43,500 |
| Best validation loss | 3.1703 | 43,500 |
| Train/val gap | ~0.05 | — |

No measurable overfitting. For reference, [Tucano-160M](https://arxiv.org/abs/2411.07854) reports validation perplexity ~30 on Portuguese text; Maracatu-20M reaches 23.81 with 10× fewer parameters.

---

## Sample outputs

Generated with `temperature=0.8`, `top_k=50`, `seed=42`. All output is lowercase — this is a tokenizer property, not a generation artifact.

**Prompt:** `O Brasil é`

```
o brasil é uma espécie de ave da família dos caririformes.
```

**Prompt:** `A capital de Pernambuco é`

```
a capital de pernambuco é um município brasileiro do estado do rio de janeiro.
```

**What these samples show:** the model produces syntactically plausible Portuguese with an encyclopedic style. They also illustrate the primary limitation at this scale: **factual hallucination is common and expected**. The capital of Pernambuco is Recife. The model does not know this reliably.

---

## Limitations

These are not disclaimers — they are accurate descriptions of what this model can and cannot do at 17M parameters.

- **Scale:** 17M parameters is small. Factual recall is unreliable. Hallucination is the norm, not the exception.
- **Lowercase only:** The tokenizer applies `nmt_nfkc_cf` normalization (lowercase). The model never generates uppercase characters.
- **Digit splitting:** Numbers are tokenized digit-by-digit. Dates, arithmetic, and numeric reasoning are not reliable.
- **Encyclopedic register:** Trained on Wikipedia only. Output tends toward formal, encyclopedic prose. Informal and conversational registers are underrepresented.
- **Portuguese Wikipedia bias:** Topics with sparse PT-BR Wikipedia coverage produce lower-quality output.
- **No safety fine-tuning:** This is an unfiltered base model. It has not been evaluated for harmful outputs and may generate biased, incorrect, or offensive content.
- **No instruction following:** Prompting it like a chat assistant will not work as expected. It continues the prompt as text completion.

---

## Use elsewhere

### HuggingFace Hub — safetensors + GGUF files

[huggingface.co/maracatu-labs/maracatu-20m](https://huggingface.co/maracatu-labs/maracatu-20m)

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

tokenizer = AutoTokenizer.from_pretrained("maracatu-labs/maracatu-20m", use_fast=False)
model = AutoModelForCausalLM.from_pretrained("maracatu-labs/maracatu-20m")
model.eval()

inputs = tokenizer("O Brasil é", return_tensors="pt")
with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=60, temperature=0.8, top_k=50, do_sample=True)
print(tokenizer.decode(out[0], skip_special_tokens=True))
```

### GitHub — code, architecture, training scripts, docs

[github.com/maracatu-labs/maracatu](https://github.com/maracatu-labs/maracatu)

Full source: model architecture, tokenizer training, corpus cleanup, export scripts, experiment logs.

### Kaggle Models — checkpoint + metadata

[kaggle.com/models/whereisanzi/maracatu-20m](https://www.kaggle.com/models/whereisanzi/maracatu-20m)

Original training checkpoint and Kaggle kernel used for the T4 run.

---

## License & citation

Code and weights are released under the [Apache License 2.0](https://github.com/maracatu-labs/maracatu/blob/main/LICENSE).

Training data (Wikipedia PT) is licensed [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) by the Wikimedia Foundation and contributors.

```bibtex
@misc{maracatu2026,
  author       = {Anzileiro, Anderson},
  title        = {Maracatu-20M: A Brazilian Portuguese Language Model Trained from Scratch},
  year         = {2026},
  publisher    = {HuggingFace},
  howpublished = {\url{https://huggingface.co/maracatu-labs/maracatu-20m}},
}
```

---

## More

Full documentation, architecture decisions, experiment logs, and roadmap:
[github.com/maracatu-labs/maracatu](https://github.com/maracatu-labs/maracatu)

Maracatu-20M is the first step. The roadmap runs to Maracatu-80B through 5 public releases (20M → 80M → 800M → 8B → 80B, roughly 10x per step), targeting Llama-3.1-70B performance on Portuguese benchmarks (ENEM, OAB, BLUEX, POSCOMP).

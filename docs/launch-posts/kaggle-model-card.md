# 🥁 Maracatu-20M

> Brazilian Portuguese language model trained from scratch. First checkpoint of the Maracatu family. Open weights, Apache 2.0.

Maracatu-20M is a 17M-parameter causal language model trained from scratch on Brazilian Portuguese Wikipedia. It is the first public checkpoint of the [Maracatu AI](https://github.com/maracatu-labs/maracatu) project, an open effort to build Portuguese-language LLMs with full transparency over architecture, data, and training.

This is a **base model** (text completion). It continues a prompt; it is not a chat assistant and does not follow instructions.

The model was trained entirely on Kaggle, using a single T4 GPU over 5h 45min.

---

## Load this model in a Kaggle Notebook

The fastest path is `kagglehub` plus `transformers`. Both are available in Kaggle Notebooks without any installation.

```python
import kagglehub
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Download the model files to the Kaggle cache
path = kagglehub.model_download("whereisanzi/maracatu-20m/transformers/default")

# Load via transformers (LlamaForCausalLM-compatible)
tokenizer = AutoTokenizer.from_pretrained(path)
model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float16)

# Move to GPU if available (T4 is available in Kaggle free tier)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
model.eval()

# Generate text
prompt = "O Brasil é"
inputs = tokenizer(prompt, return_tensors="pt").to(device)

with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=60,
        temperature=0.8,
        top_k=50,
        do_sample=True,
    )

print(tokenizer.decode(output[0], skip_special_tokens=True))
```

The model is small (17M parameters, ~34 MB in fp16) and loads in seconds on T4. No quantization is required to run it comfortably within Kaggle's free GPU tier.

---

## Architecture

Llama-style decoder-only transformer. The `state_dict` is bit-for-bit compatible with `transformers.LlamaForCausalLM` and loads via `AutoModelForCausalLM.from_pretrained` without any conversion script.

| Hyperparameter | Value |
|---|---|
| Total parameters | 17M (16.77M) |
| Non-embedding parameters | 10.62M |
| Layers | 6 |
| Hidden size | 384 |
| Attention heads | 6 |
| Intermediate size (SwiGLU) | 1024 |
| Context length | 512 tokens |
| Vocabulary | 16,000 tokens |
| Normalization | RMSNorm |
| Positional encoding | RoPE (rotate-half) |
| Activation | SwiGLU |
| Bias in Linear layers | No |
| Weight tying (embed / lm_head) | Yes |
| Tokenizer | SentencePiece BPE, lowercase (`nmt_nfkc_cf`), `split_digits`, byte fallback |

---

## Training

| Item | Value |
|---|---|
| Framework | PyTorch |
| Hardware | **Kaggle T4** (single GPU, 15.6 GB VRAM) |
| Training data | Wikipedia PT (`wikimedia/wikipedia`, snapshot `20231101.pt`) |
| Corpus size | 2.28 GB, ~550M BPE tokens, 979,492 articles |
| Total iterations | 50,000 |
| Tokens seen | ~410M (~0.75 epoch) |
| Batch size | 16 |
| Context length | 512 tokens |
| Optimizer | AdamW (beta1=0.9, beta2=0.95, weight_decay=0.1) |
| Learning rate schedule | 3e-4 to 3e-5 (linear warmup 1,000 steps + cosine decay) |
| Gradient clipping | 1.0 |
| Throughput | ~20,000 tok/s |
| Total training time | 5h 45min |
| Best validation perplexity | **23.81** (step 43,500) |
| Best validation loss | 3.1703 (step 43,500) |

For reference: [Tucano-160M](https://arxiv.org/abs/2411.07854) reports validation perplexity ~30 on Portuguese text. Maracatu-20M reaches 23.81 with 10x fewer parameters.

---

## Sample outputs

Generated with `temperature=0.8`, `top_k=50`, `seed=42`. All output is lowercase; this is a tokenizer property (`nmt_nfkc_cf` normalization), not a generation artifact.

**Prompt:** `O Brasil é`

```
o brasil é uma espécie de ave da família dos caririformes.
```

**Prompt:** `A capital de Pernambuco é`

```
a capital de pernambuco é um município brasileiro do estado do rio de janeiro.
```

**Prompt:** `Machado de Assis escreveu`

```
machado de assis escreveu o livro "a obra-prima do autor da obra-prima do autor", que foi escrito em 1998 e publicado em 2001.
```

These samples show the model's encyclopedic register and its primary limitation at this scale: **factual hallucination is common and expected**. The capital of Pernambuco is Recife. Machado de Assis died in 1908. The model produces syntactically plausible Portuguese but cannot reliably retrieve facts.

---

## Limitations

These are accurate descriptions of what the model can and cannot do at 17M parameters, not disclaimers.

- **Scale.** 17M parameters is small. Factual recall is unreliable. Hallucination is the norm at this size.
- **Lowercase only.** The tokenizer applies `nmt_nfkc_cf` normalization. The model never generates uppercase characters.
- **Digit splitting.** Numbers are tokenized digit-by-digit. Dates, arithmetic, and numeric reasoning are not reliable.
- **Encyclopedic register.** Trained on Wikipedia only. Output tends toward formal, encyclopedic prose. Informal registers are underrepresented.
- **Portuguese Wikipedia bias.** Topics with sparse PT-BR Wikipedia coverage produce lower-quality output.
- **No safety fine-tuning.** This is an unfiltered base model. It has not been evaluated for harmful outputs and may generate biased, incorrect, or offensive content.
- **No instruction following.** Prompting it like a chat assistant will not produce useful results. It completes text.

---

## Available on other channels

| Channel | Link | What you get |
|---|---|---|
| HuggingFace Hub | [maracatu-labs/maracatu-20m](https://huggingface.co/maracatu-labs/maracatu-20m) | safetensors, GGUF files, full model card |
| Ollama | [ollama.com/whereisanzi/maracatu-20m](https://ollama.com/whereisanzi/maracatu-20m) | Q4_K_M, Q5_K_M, Q8_0 quantizations |
| GitHub | [maracatu-labs/maracatu](https://github.com/maracatu-labs/maracatu) | Architecture, training scripts, experiment logs, roadmap |
| Kaggle Models (this page) | [whereisanzi/maracatu-20m](https://www.kaggle.com/models/whereisanzi/maracatu-20m) | Original checkpoint, training kernel |

---

## License and citation

Code and model weights are released under the [Apache License 2.0](https://github.com/maracatu-labs/maracatu/blob/main/LICENSE).

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

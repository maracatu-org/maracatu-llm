# <!-- run name --> (<!-- YYYY-MM-DD -->)

**Git revision:** `<hash>`
**Config:** [`configs/<file>.yaml`](../../configs/<file>.yaml)
**Hardware:** <!-- M2 Pro MPS / T4 Kaggle / 4090 RunPod -->
**Goal:** <!-- 1-2 sentences -->

## Setup

- **Architecture:** <!-- Llama-like / GPT-2-like, main ingredients -->
- **Params:** <!-- total / non-embedding -->
- **Dataset:** <!-- wikimedia/wikipedia (20231101.pt), N tokens, SHA-256 -->
- **Tokenizer:** <!-- SentencePiece BPE, vocab 16k -->

## Relevant hyperparameters

| Field | Value |
|---|---|
| `hidden_size` | |
| `num_hidden_layers` | |
| `num_attention_heads` | |
| `intermediate_size` | |
| `max_position_embeddings` | |
| `batch_size` | |
| `learning_rate` | |
| `warmup_iters` | |
| `max_iters` | |

## Results

### Loss curve

| step | train | val |
|---|---|---|

### Final metrics

- **Best val loss:** `X.XXXX` (step N)
- **Final train loss:** `X.XXXX`
- **Total time:** `XX.X min`
- **Average throughput:** `XXk tok/s`

## Qualitative samples

With `temperature=0.8`, `top_k=50`, `seed=42`.

### Prompt: `"O Brasil é"`

```
...
```

### Prompt: `"Machado de Assis escreveu"`

```
...
```

### Prompt: `"A capital de Pernambuco é"`

```
...
```

## Analysis

<!-- What worked, what didn't, hypotheses, next steps -->

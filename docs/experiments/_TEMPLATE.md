# <!-- nome do run --> (<!-- YYYY-MM-DD -->)

**Git revision:** `<hash>`
**Config:** [`configs/<arquivo>.yaml`](../../configs/<arquivo>.yaml)
**Hardware:** <!-- M2 Pro MPS / T4 Kaggle / 4090 RunPod -->
**Objetivo:** <!-- 1-2 frases -->

## Setup

- **Arquitetura:** <!-- Llama-like / GPT-2-like, principais ingredientes -->
- **Params:** <!-- total / não-embedding -->
- **Dataset:** <!-- wikimedia/wikipedia (20231101.pt), N tokens, SHA-256 -->
- **Tokenizer:** <!-- SentencePiece BPE, vocab 16k -->

## Hiperparâmetros relevantes

| Campo | Valor |
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

## Resultados

### Curva de loss

| step | train | val |
|---|---|---|

### Métricas finais

- **Best val loss:** `X.XXXX` (step N)
- **Final train loss:** `X.XXXX`
- **Tempo total:** `XX.X min`
- **Throughput médio:** `XXk tok/s`

## Amostras qualitativas

Com `temperature=0.8`, `top_k=50`, `seed=42`.

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

## Análise

<!-- O que funcionou, o que não, hipóteses, próximos passos -->

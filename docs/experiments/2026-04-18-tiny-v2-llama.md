# tiny-v2-llama (2026-04-18)

**Git revision:** `766b9366195537d155c259e14893328356012e12`
**Config:** [`configs/maracatu_tiny.yaml`](../../configs/maracatu_tiny.yaml) (versão Llama-like)
**Hardware:** MacBook Pro M2 Pro, 16 GB RAM, MPS backend
**Objetivo:** Validar a migração para arquitetura Llama-style (RMSNorm + RoPE + SwiGLU) e comparar com o baseline GPT-2-like ([tiny-v1-gpt2](2026-04-18-tiny-v1-gpt2.md)) na mesma ordem de parâmetros e compute.

## Setup

- **Arquitetura:** Llama-like: RMSNorm (sem mean-centering, sem bias), RoPE (convenção HF "rotate-half"), SwiGLU MLP (`down(silu(gate) * up)`), weight tying, `bias=False` em tudo, suporte a GQA (aqui `num_key_value_heads = num_attention_heads`, sem GQA ainda).
- **Compatibilidade:** state_dict alinhado ao `LlamaForCausalLM` do HuggingFace; os pesos carregam direto via `AutoModelForCausalLM.from_pretrained` sem conversão.
- **Params:** 7.31M total / 3.21M não-embedding.
- **Dataset:** `wikimedia/wikipedia` config `20231101.pt` (CC BY-SA 4.0). 979.492 artigos, ~593M tokens no split de treino + 6M no split de val (mesmos `tokens.npy` do v1).
- **Tokenizer:** SentencePiece BPE 16k, sem mudanças em relação ao v1.

## Hiperparâmetros relevantes

| Campo | Valor |
|---|---|
| `hidden_size` | 256 |
| `num_hidden_layers` | 4 |
| `num_attention_heads` | 4 |
| `num_key_value_heads` | 4 |
| `intermediate_size` | 704 (~2.67× `hidden_size`, compensa SwiGLU ter 3 projeções) |
| `max_position_embeddings` | 256 |
| `rms_norm_eps` | 1e-5 |
| `rope_theta` | 10000 |
| `attention_dropout` | 0.0 |
| `tie_word_embeddings` | true |
| `batch_size` | 32 |
| `learning_rate` | 5e-4 (warmup 200, cosine → 5e-5) |
| `max_iters` | 5.000 |
| `warmup_iters` | 200 |

## Resultados

### Curva de loss (evals a cada 250 steps)

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

### Métricas finais

- **Best val loss:** `6.6408` (step 3500) → perplexidade ≈ 766
- **Tempo total:** 18.5 min
- **Throughput médio:** ~36.600 tok/s no MPS
- **Tokens de treino processados:** ~40M de 593M disponíveis (7%, sub-treinado)

## Amostras qualitativas

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

## Análise

**Comparação com v1 (GPT-2-like):**

| Métrica | v1 (GPT-2) | v2 (Llama) | Δ |
|---|---|---|---|
| Best val loss | 6.7514 | **6.6408** | -0.1106 (-1.6%) |
| Tempo | 14.7 min | 18.5 min | +26% |
| Throughput | 46k tok/s | 36k tok/s | -22% |
| Params (não-emb) | 3.2M | 3.2M | igual |

**Interpretação:**
- Melhoria modesta mas real em val loss no mesmo orçamento de steps, com arch com mais operações por passo. Coerente com a teoria: RoPE + SwiGLU rendem mais em treinos longos; em 5k iters (sub-treinado), o efeito composto mal começou.
- Queda de throughput esperada: SwiGLU tem 3 projeções lineares vs 2 de uma FFN GELU; RoPE adiciona ops por head; RMSNorm ganha um pouco mas não compensa o resto.

**Qualitativo:** o salto mais visível foi no prompt `"A capital de Pernambuco é"`:
- v1: fugiu para "banda ganhou o show", completamente fora de contexto
- v2: ficou em contexto geográfico: "possui a maior população da região da antiga cidade", "km2", "censo"

Ainda erra o fato (Recife), mas o modelo entendeu que "capital de X" é um construct geográfico. Indicativo de que RoPE pode estar ajudando com relação de dependência longa dentro do contexto.

**Sinais bons:**
- Loss caiu de 9.68 → 6.64 (melhor que o v1)
- Gap train/val pequeno (~0.06), sem overfitting
- Best val no step 3500, não 4750: cosine decay já estava apertando, e o modelo continuou melhorando no train mas não no val, leve sinal de teto pra esse budget

**Próximos passos validados por este experimento:**
1. Arch nova funciona: seguir com ela no 20M
2. O gap de -1.6% em 5k iters pode compor pra -5 a -10% em 50k iters (especulativo)
3. Partir pro 20M sem novas mudanças arquiteturais

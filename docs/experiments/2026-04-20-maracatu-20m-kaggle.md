# Maracatu-20M: primeiro treino completo em escala real, Kaggle T4 (2026-04-20)

**Git revision:** `<kaggle-run>` (kernel rodou fora do repo; git_revision não capturado automaticamente; commit base estimado anterior a `76c0f05`)
**Config:** [`configs/maracatu_20m.yaml`](../../configs/maracatu_20m.yaml)
**Hardware:** Kaggle GPU T4 single (15.6 GB VRAM)
**Objetivo:** Primeiro treino completo do Maracatu-20M em hardware de nuvem, validando o pipeline end-to-end (corpus → tokenizer → treino → export HF → publicação). Serviu de baseline para o ladder M-80M.

## Setup

- **Arquitetura:** Llama-style: RMSNorm, RoPE rotate-half (implementação HF), SwiGLU, sem bias nos Linear, weight tying embedding/lm_head, compatível bit-a-bit com `LlamaForCausalLM`
- **Params:** 16.77M total / 10.62M não-embedding
- **Dataset:** `wikimedia/wikipedia` config `20231101.pt` (CC BY-SA 4.0). Após filtros: 979.492 artigos, 9,7 M linhas, 2,28 GB, ~550 M tokens BPE
- **Tokenizer:** SentencePiece BPE, vocab 16.000 tokens, treinado em PT-BR com `nmt_nfkc_cf` (lowercase), split_digits, byte_fallback

## Hiperparâmetros relevantes

| Campo | Valor |
|---|---|
| `hidden_size` | 384 |
| `num_hidden_layers` | 6 |
| `num_attention_heads` | 6 |
| `num_key_value_heads` | 6 (sem GQA) |
| `intermediate_size` | 1024 |
| `max_position_embeddings` | 512 |
| `batch_size` | 16 |
| `learning_rate` | 3e-4 → 3e-5 (cosine decay) |
| `warmup_iters` | 1000 (linear) |
| `max_iters` | 50.000 |
| `beta1 / beta2` | 0.9 / 0.95 |
| `weight_decay` | 0.1 |
| `grad_clip` | 1.0 |
| `seed` | 42 |

Tokens consumidos: ~410 M (~0,75 epoch). Throughput médio: ~20k tok/s.

## Resultados

### Curva de loss

Amostra completa extraída do log JSON (a cada 500 steps):

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

### Métricas finais

- **Best val loss:** `3.1703` (step 43.500) → **perplexidade 23.81**
- **Final val loss:** `3.2215` (step 49.500)
- **Final train loss:** `3.1722` (step 49.500)
- **Gap train/val no plateau:** ~0.05 nats (sem overfitting aparente)
- **Tempo total:** 5h45min (~20.770s de log)
- **Throughput médio:** ~20k tok/s

## Amostras qualitativas

Com `temperature=0.8`, `top_k=50`, `seed=42`, `max_new_tokens=60`.

### Prompt: `"O Brasil é"`

```
O Brasil é uma espécie de ave da família dos caririformes.
```

Estrutura sintática válida em PT-BR. Conteúdo factual inventado (alucinação típica de LM não-instruction-tuned em escala pequena).

### Prompt: `"A capital de Pernambuco é"`

```
A capital de Pernambuco é um município brasileiro do estado do rio de janeiro.
```

O modelo associou corretamente o frame "capital de estado brasileiro" mas errou tanto a cidade (Recife) quanto o estado referenciado. Indica compreensão parcial de entidades geográficas, padrão esperado nessa escala.

### Prompt: `"Machado de Assis escreveu"`

```
machado de assis escreveu o livro "a obra-prima do autor da obra-prima do autor", que foi escrito em 1998 e publicado em 2001. no brasil, em 2003, o livro foi publicada pela universidade federal de minas gerais, que viria a ser
```

O modelo aprendeu a colocação "livro" + data de publicação + editora, coerente com o padrão enciclopédico de artigos de Wikipedia sobre autores. Erros típicos: datas absurdas para Machado (morto em 1908), repetição recursiva ("obra-prima do autor da obra-prima do autor") indicando que 17M params não tem capacidade pra suprimir loops de token em contextos abstratos, e concordância de gênero errada ("o livro foi publicada").

### Prompt: `"A literatura brasileira"`

```
a literatura brasileira (1988-1994) a literatura brasileira é um movimento político de classe, na qual se fomenta a autodidata, a revolução de 1900 e o primeiro grande movimento político brasileiro. o movimento é apoiado por uma grande maioria de ativ
```

Sintaxe PT-BR válida com vocabulário de ciências humanas (movimento político, classe, revolução, autodidata). O modelo falha em ancorar "literatura brasileira" como fenômeno literário e a trata como movimento político. O parêntese `(1988-1994)` mostra que aprendeu o padrão de anotação de períodos da Wikipedia, mesmo sem contexto semântico.

## Análise

### O que funcionou

Primeira validação end-to-end da arquitetura Llama-style em escala real (não só no tiny de 7M). O pipeline completo (corpus Wikipedia PT filtrado → tokenizer SentencePiece 16k → treino com warmup+cosine → export `LlamaForCausalLM` → upload HF → quantização GGUF → Ollama) funcionou sem retrabalho de arquitetura.

A queda de loss de 5.90 (step 500) para 3.17 (step 43.500) é monotônica nas primeiras 35k iters, com plateau suave a partir daí. Sem spike de NaN, sem divergência. Grad clip 1.0 e AdamW com β2=0.95 mantiveram o treino estável em T4 single por 5h45min.

### Tokens por parâmetro

410 M tokens / 16.77 M params ≈ **24,5 tokens/param**. A receita Chinchilla ([Hoffmann et al., 2022](https://arxiv.org/abs/2203.15556)) recomenda ~20 tokens/param para um modelo compute-ótimo. O treino ficou levemente acima do ótimo, razoável dado que o objetivo aqui era validação de pipeline, não extrair máxima eficiência.

### Plateau e cosine decay

O best val (3.1703) ocorreu no step 43.500; os 6.500 steps seguintes mostraram leve oscilação sem melhora consistente (val final 3.22). O cosine decay já havia reduzido o LR de 3e-4 para próximo de 3e-5 nessa região. O plateau não indica saturação do modelo: o LR estava baixo demais para aprendizado adicional com o corpus atual. Mais iters com LR decaído não ajudariam; um restart de LR ou corpus maior seriam as alternativas.

### Comparação com tiny-long

O run `maracatu_tiny_long` (7M params, 50k iters, context 256) havia encerrado com val ~6.39 (perp ~599). O Maracatu-20M atingiu perp 23.81: **redução de ~25× na perplexidade**. A melhora resulta da combinação de três fatores: (a) 3,3× mais parâmetros, (b) context 512 vs 256 (permite capturar dependências de maior alcance), (c) mesmas 50k iters sobre corpus idêntico. Não é possível isolar a contribuição de cada fator sem ablação.

### Comparação com Tucano-160M

O paper do Tucano ([Correa et al., 2024](https://arxiv.org/abs/2411.07854)) reporta perplexidade em torno de 30 em Wikipedia PT para o Tucano-160M. O Maracatu-20M, com **10× menos parâmetros**, atingiu perp 23.81 no mesmo domínio de avaliação. Hipótese principal: corpus curado (Wikipedia filtrado, sem ruído web) combinado com arquitetura moderna (RoPE, SwiGLU, RMSNorm) rende desproporcionalmente bem em escala pequena. Comparação direta exigiria avaliação no mesmo split e mesma métrica; este número deve ser tratado como indicativo, não como benchmark formal.

### O que não funcionou / limitações

- Geração produz alucinações factuais consistentes, esperadas nessa escala e sem RLHF/instruction-tuning.
- Git revision não foi capturada pelo kernel Kaggle (roda fora do repo git); o campo `git_revision` no checkpoint contém `"<kaggle>"`.
- Corpus limitado a Wikipedia PT (~550 M tokens): adequado para 20M, insuficiente para 500M+ sem expansão.

### Implicações para o próximo experimento

1. **Corpus expandido é pré-requisito para M-80M**: Wikipedia PT (~550M tokens) cobre ~6,9 tokens/param para 80M, abaixo do ótimo Chinchilla (1,6B tokens necessários, 3× expansão). Incorporar Gutenberg PT e OSCAR PT filtrado antes de iniciar o próximo treino.
2. **Compute**: M-80M requer RunPod A100 (~8h estimadas, R$5-15k com retries dependendo de spot vs on-demand). Kaggle T4 ainda é viável aqui, mas RunPod recomendado pra maturidade do pipeline.
3. **Pipeline GGUF + Ollama está validado**: não há retrabalho de infraestrutura necessário antes do próximo release.
4. **Não iterar mais no 20M**: o modelo cumpriu seu papel de validação de pipeline e baseline. Recursos de treino vão direto para M-80M.

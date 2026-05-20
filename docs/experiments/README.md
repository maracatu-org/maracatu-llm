# Experiment log

Registro cronológico de todos os treinos do Maracatu. Cada experimento é um arquivo Markdown com data, commit, configuração, métricas, amostras qualitativas e análise.

**Convenção de nome:** `YYYY-MM-DD-<runname>.md`

## Motivação

Treinos de LLM são difíceis de comparar ao longo do tempo sem notas estruturadas. Este log serve para:

1. **Reprodutibilidade**: dado o git rev e a config, reproduzir o resultado
2. **Comparação**: avaliar se uma mudança arquitetural/hiperparâmetro melhorou algo
3. **Aprendizado**: documentar o que funcionou, o que não funcionou e por quê

## Índice

| Data | Run | Arquitetura | Iters | Best val loss | Arquivo |
|---|---|---|---|---|---|
| 2026-04-18 | `maracatu-tiny-test` v1 | GPT-2-like (LayerNorm, GELU, abs pos emb) | 5.000 | 6.7514 | [tiny-v1-gpt2](2026-04-18-tiny-v1-gpt2.md) |
| 2026-04-18 | `maracatu-tiny-test` v2 | Llama-like (RMSNorm, SwiGLU, RoPE) | 5.000 | 6.6408 | [tiny-v2-llama](2026-04-18-tiny-v2-llama.md) |
| 2026-04-19 | `maracatu-tiny-long` | Llama-like (mesmo tiny) | 50.000 | 6.3949 | [tiny-long](2026-04-19-tiny-long.md) |

## Template

Novo experimento: copiar [`_TEMPLATE.md`](_TEMPLATE.md) e preencher.

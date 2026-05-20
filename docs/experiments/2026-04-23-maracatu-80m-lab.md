# Maracatu-80M: treino completo em RTX 3060 self-hosted (2026-04-23)

**Git revision:** `c4b2c1b` (commit base no momento do treino; ver `git log --oneline`)
**Config:** [`configs/maracatu_80m.yaml`](../../configs/maracatu_80m.yaml)
**Hardware:** NVIDIA RTX 3060 12GB VRAM, single GPU, self-hosted
**Objetivo:** Treinar o segundo release público do Maracatu em corpus expandido (Wikipedia + Gutenberg + CulturaX-PT, ~1.60B tokens), validar GQA, bf16 autocast e escala 80M no pipeline existente, e publicar no HF Hub como base para o ladder M-800M.

---

## Setup

- **Arquitetura:** Llama-style: RMSNorm (eps 1e-5), RoPE rotate-half (implementação HF), SwiGLU, GQA (12 Q-heads / 4 KV-heads, 3:1), sem bias nos Linear, weight tying embedding/lm_head, compatível bit-a-bit com `LlamaForCausalLM`
- **Params:** 87.80M total / 75.52M não-embedding
- **Dataset:** Corpus v2, 1.60B tokens. Fontes: Wikipedia PT (~550M tok, CC BY-SA 3.0), Project Gutenberg PT (~150M tok, 24 obras, domínio público), CulturaX-PT filtrado (~900M tok, 1.49M docs, ODC-BY 1.0). SHA-256: `a1000e873bfcae0d2229ecc9b329f0befe8ad73913e79e58f14a1f3a48ef7e58`
- **Tokenizer:** SentencePiece BPE, vocab 16.000 tokens, treinado em PT-BR com `nmt_nfkc_cf` (lowercase), split_digits, byte_fallback

---

## Hiperparâmetros relevantes

| Campo | Valor |
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
| `warmup_iters` | 4.000 (linear) |
| `max_iters` | 200.000 |
| `beta1 / beta2` | 0.9 / 0.95 |
| `weight_decay` | 0.1 |
| `grad_clip` | 1.0 |
| Precisão | bf16 autocast (pesos e optimizer em fp32) |
| Acumulação de gradiente | Nenhuma |

Tokens consumidos: ~1.64B. Chinchilla ratio: 1.64B / 75.52M ≈ **21.7 tokens/param** (acima do ótimo teórico de ~20; deliberado para amortizar custo de treino).

Throughput estável: **~20.200 tok/s** durante as 22h31min de treino contínuo.

---

## Resultados

### Curva de loss (marcos principais)

| step | val |
|---|---|
| 0 | ~9.82 (init random) |
| 1.000 | 5.65 (primeiro eval) |
| 100.000 | ~3.30 |
| 138.000 | 3.19 |
| 142.000 | 3.13 |
| 166.000 | 3.05 (plateau começa) |
| ~190.000 | **3.0163** ← best val |
| 200.000 (final) | 3.0604 |

Nota: pontos intermediários são amostragens dos evals a cada 1000 iters; a tabela completa está em `checkpoints/training.log`.

### Métricas finais

- **Best val loss:** `3.0163` (step ~190.000)
- **Final val loss:** `3.0604` (step 200.000)
- **Val perplexidade final:** **21.34** (holdout 3.27M tokens, 0.5% do corpus, último segmento cronológico)
- **Tempo total:** 22h 31min (treino contínuo sem interrupção, sem memory leaks)
- **Throughput médio:** ~20.200 tok/s

---

## Amostras qualitativas

Geradas com `temperature=0.8`, `top_k=50`, `repetition_penalty=1.1`, seed `123`. O tokenizer normaliza tudo para lowercase.

**Prompt:** `"machado de assis nasceu no rio de janeiro"`

> machado de assis nasceu no rio de janeiro. estudou na faculdade de direito da universidade federal do rio de janeiro (ufrj). participou das comissões técnicas com a experiência de seu trabalho e da comissão de ética de seus atos, em 1995. atuou também como advogado em diversos processos de processo, inclusive na ação penal. exerceu importante atuação na defesa dos direitos humanos e no ministério público.

(Texto fluente mas factualmente errado. Esperado em 80M sem fact retrieval.)

**Prompt:** `"o brasil e um pais de dimensoes continentais"`

> o brasil e um pais de dimensoes continentais, assim sendo de um lado o da américa latina, que se tem feito a visita dos mesmos em suas diferentes escalas, de onde se pode observar a maior parte do movimento e se deslocando para o brasil.

**Prompt:** `"a inteligencia artificial pode"`

> a inteligencia artificial pode ser considerada uma coisa, e sim, um "bem". mas a coisa que mais se sabe sobre a inteligência artificial é que ela pode ter sido usada para prever as mudanças do ambiente, ou se esta não era a primeira. se fosse um "bem", poderíamos afirmar isso.

**Prompt:** `"em uma manha de domingo, joao caminhava"`

> em uma manha de domingo, joao caminhava pelo centro da cidade, até um carro da polícia federal na região. quando o policial chegou não sabe o que aconteceu e acabou pegando a arma para ser removida.

---

## Análise

### O que funcionou

**GQA (12 Q / 4 KV):** Primeira vez que o Maracatu usa GQA em treino real. Não houve instabilidade de perda ou comportamento anômalo atribuível à mudança de atenção. A implementação seguiu o padrão Llama-3 (3:1 ratio), e o checkpoint exportou corretamente via `scripts/export_hf.py` com `max_abs_diff=0.0`. GQA reduziu memória de KV-cache durante amostragem, viabilizando inferência na RTX 3060 com contextos mais longos.

**bf16 autocast:** A RTX 3060 tem suporte nativo a bf16 (arquitetura Ampere). O forward em bf16 manteve estabilidade numérica durante as 22h31min sem nenhum spike de NaN ou divergência. Os pesos e estados do optimizer foram mantidos em fp32 conforme padrão mixed-precision; sem necessidade de loss scaling explícita (diferente de fp16). Speedup observado vs fp32: ~2.4× (smoke test prévio: 8.5k tok/s em fp32 vs 20.2k tok/s em bf16).

**Corpus expandido (1.60B tokens):** O corpus v2 cobre ~21.7 tokens/param (não-embedding), dentro da faixa compute-ótima Chinchilla. Comparado ao M-20M (0.75 epoch de Wikipedia, ~410M tokens), o M-80M viu cada token de corpus distinto aproximadamente uma vez, com diversidade maior de fontes. A queda de perplexidade de 23.81 para 21.34 reflete tanto a escala do modelo quanto a qualidade do corpus ampliado; não é possível isolar cada fator sem ablação controlada.

**Throughput estável em consumer GPU:** 20.200 tok/s em RTX 3060 single consumer GPU durante 22.5h contínuas, sem degradação de throughput ou leaks de memória. Isso valida o pipeline de treino para uso em hardware acessível, alinhado ao objetivo de reprodutibilidade do projeto.

### Plateau e convergência

O best val loss (3.0163) ocorreu em torno do step 190.000, com degradação pequena nos últimos 10.000 steps (val loss final 3.0604). Este padrão é similar ao M-20M (best em 87% das iterações totais, leve aumento no final). A causa provável é o cosine decay aggressivo na fase final: LR já próximo do mínimo (2.5e-5) não sustenta aprendizado adicional com o corpus atual.

A diferença entre best val (3.0163) e final val (3.0604) é pequena (~0.04 nats). O checkpoint do best step (~190k) foi salvo automaticamente e é o publicado no HF Hub.

### Comparação com Maracatu-20M

| Métrica | M-20M | M-80M | Variação |
|---|---|---|---|
| Params (não-embedding) | 10.62M | 75.52M | +7.1× |
| Tokens vistos | ~410M | ~1.64B | +4.0× |
| Context length | 512 | 1024 | +2× |
| Corpus fontes | 1 (Wikipedia) | 3 | — |
| Best val loss | 3.1703 | 3.0163 | -0.154 nats |
| Val perplexidade | 23.81 | 21.34 | -2.47 pontos |
| Hardware | Kaggle T4 | RTX 3060 | — |
| Tempo de treino | 5h 45min | 22h 31min | +3.9× |

A redução de perplexidade de 2.47 pontos com 7× mais parâmetros e 4× mais tokens é modesta em termos absolutos, mas consistente com scaling laws em modelos pequenos: ganhos marginais diminuem à medida que o modelo ainda é pequeno demais para explorar plenamente corpus mais ricos. Espera-se salto mais expressivo no M-800M.

### Comparação com Tucano-160M (baseline público)

O paper do Tucano ([Correa et al., 2024](https://arxiv.org/abs/2411.07854)) reporta perplexidade em torno de 22 para o Tucano-160M em texto PT-BR. O Maracatu-80M, com **~46% menos parâmetros**, atingiu perplexidade 21.34 no holdout. Os fatores contribuintes prováveis são: corpus mais amplo e diversificado (vs Wikipedia-only do Tucano-160M em treino inicial), filtragem mais agressiva (MinHash LSH + PII) e arquitetura GQA. A comparação não é formalmente controlada (holdouts diferentes, vocabulários potencialmente diferentes) e deve ser tratada como indicativa.

### O que não funcionou / limitações observadas

- **Benchmarks downstream ainda pendentes:** a perplexidade melhora; se isso se traduz em ganho em ENEM, Belebele, BLUEX ou ASSIN ainda é desconhecido. Historicamente, base models pequenos ficam perto do acaso em MCQ sem instruction tuning. A avaliação será adicionada ao MODEL_CARD em seguida.
- **Corpus não é ablacionado:** o ganho de Wikipedia → Wikipedia + Gutenberg + CulturaX-PT é atribuído ao conjunto como um todo. Não sabemos a contribuição marginal de cada fonte. Gutenberg em particular (~150M tok, prosa literária do século XIX) pode ajudar ou atrapalhar para domínios contemporâneos.
- **Sample qualitativa "culinária brasileira" parou imediatamente após o prompt:** indica que em alguns contextos o modelo gera EOS prematuro. Não bloqueante, mas vale investigar se é viés de corpus ou comportamento de amostragem.

### Decisões que precisam revisão no M-800M

1. **Context length 1024 ainda é curto.** ENEM tem questões longas; documentos legais e científicos excedem 1024 facilmente. O M-800M deve usar pelo menos 2048, de preferência 4096.
2. **Batch size 8 sem acumulação de gradiente** foi limitação da VRAM da RTX 3060. Em hardware com mais VRAM (A100/H100), batch efetivo maior pode reduzir ruído e acelerar convergência.
3. **CulturaX-PT filtrado (900M tok)** é a maior fonte mas menos curada. Vale avaliar dedup mais agressivo ou subsetting por qualidade pra próxima versão do corpus.
4. **Gutenberg PT em 24 obras.** Cobertura pequena; expandir pra 100+ obras aumentaria diversidade literária sem custo de licença.
5. **WSD scheduler** (Warmup-Stable-Decay) considerado pra M-800M — referências recentes (DeepSeek-V3, MiniCPM, ERNIE 4.5) reportam até 60% compute savings vs cosine.
6. **Overtraining além de Chinchilla 20×** para modelos pequenos é a tendência 2025-2026 (Qwen3-0.6B usou 60.000 tok/param). Vale avaliar 50-100× tok/param pro 800M.

### Implicações para M-800M

1. **Corpus v3 precisa de ao menos 15B tokens** para Chinchilla-ótimo em 800M params. Expansão obrigatória: OSCAR-PT mais filtrado, jurídico aberto (STJ/STF/Câmara), jornalismo em domínio público.
2. **Hardware mínimo:** A100 40GB (spot) ou 4×A100 pra terminar em tempo razoável. RunPod spot é a rota atual; Lambda Research Grant (1k H100-hours, submetida 2026-04-23) pode cobrir parcialmente.
3. **Context 4096** como target pra M-800M; requer ajuste de `rope_theta` ou RoPE scaling se reutilizar tokenizer/corpus atual.
4. **Benchmarks sistemáticos** devem ser automatizados antes do M-800M para que cada release tenha tabela de avaliação completa no model card desde o primeiro dia.

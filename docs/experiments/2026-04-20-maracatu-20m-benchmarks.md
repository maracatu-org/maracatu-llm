# Maracatu-20M: Benchmarks PT-BR (2026-04-20)

**Git revision:** `91d3861ef4ba3f23b0c758565bd901faa90aa16f`
**Hardware:** MacBook Pro M2 Pro (16GB), CPU-only (MPS nao utilizado pelo lm-eval)
**Objetivo:** Estabelecer baseline honesto do Maracatu-20M em benchmarks PT-BR e comparar contra Tucano-160M e Tucano-630M usando o mesmo pipeline reprodutivel.

---

## Setup

- **Framework:** `lm-evaluation-harness` (EleutherAI) versao `0.4.12.dev0`, clonado em `.cache/lm-evaluation-harness/`
- **Venv isolado:** `.cache/lm-eval-venv/` (Python 3.13.7, nao poluiu o `.venv` do projeto)
- **Seed:** 42 em todos os runs
- **Temperature:** T=0 (greedy/loglikelihood) para todos os benchmarks MCQ -- comportamento padrao do lm-eval para `output_type: multiple_choice`
- **Batch size:** 8 (CPU-safe para modelos ate 630M)
- **Modelos avaliados:**
  - `maracatu-ai/maracatu-20m` -- 17M params, 16k vocab, context 512
  - `TucanoBR/Tucano-160m` -- 160M params, baseline canonico BR mesmo tier
  - `TucanoBR/Tucano-630m` -- 630M params, tier acima, pra medir gap de escala

---

## Benchmarks e Configuracoes

### 1. Belebele PT (`belebele_por_Latn`)
- **Fonte:** `facebook/belebele` (nativo no lm-eval-harness)
- **Formato:** MCQ 4 opcoes, leitura e compreensao de texto
- **Split:** test (900 questoes)
- **Fewshot:** 0-shot
- **Metrica:** `acc` (accuracy)
- **Chance aleatoria:** 25.00%
- **Tempo Maracatu-20M:** ~30s

### 2. ASSIN Entailment + Paraphrase
- **Fonte:** `nilc-nlp/assin` (nativo no lm-eval-harness)
- **Formato:** MCQ binario (2 opcoes) -- inferencia textual e parafrase
- **Split:** test (4000 amostras cada)
- **Fewshot:** 0-shot
- **Metrica:** `acc`
- **Chance aleatoria:** 50.00%
- **Tempo Maracatu-20M:** ~4min

### 3. ENEM Challenge (`enem_challenge`) -- harness custom
- **Fonte:** `eduagarcia/enem_challenge` (HuggingFace)
- **Formato:** MCQ 5 opcoes (A-E), questoes reais do ENEM 2009-2023
- **Total:** 1431 questoes validas (1 anulada filtrada via `process_docs`)
- **Fewshot:** 0-shot
- **Task YAML:** `scripts/eval/tasks/enem_challenge/enem_challenge.yaml`
- **Metrica:** `acc`, `acc_norm`
- **Chance aleatoria:** 20.00%
- **Tempo Maracatu-20M:** ~1min30s | Tucano-630M: ~25min (CPU, 630M params)
- **Nota de truncamento:** questoes longas (>512 tokens com contexto) sao truncadas pela esquerda pelo harness -- afeta uma minoria das questoes

---

## Resultados

### Tabela comparativa

| Modelo | Params | Belebele PT (4-op) | ASSIN Entailment (2-op) | ASSIN Paraphrase (2-op) | ENEM (5-op) |
|---|---|---|---|---|---|
| **Chance aleatoria** | -- | 25.00% | 50.00% | 50.00% | 20.00% |
| **Maracatu-20M** | 17M | **23.78%** (+/- 1.42%) | 27.67% (+/- 0.71%) | **60.52%** (+/- 0.77%) | 19.22% (+/- 1.04%) |
| **Tucano-160M** | 160M | 22.56% (+/- 1.39%) | **31.35%** (+/- 0.73%) | 56.37% (+/- 0.78%) | **21.10%** (+/- 1.08%) |
| **Tucano-630M** | 630M | 22.89% (+/- 1.40%) | 30.40% (+/- 0.73%) | 54.93% (+/- 0.79%) | **21.10%** (+/- 1.08%) |

### Deltas vs Tucano-160M (baseline canônico)

| Benchmark | Maracatu-20M vs Tucano-160M | Interpretacao |
|---|---|---|
| Belebele PT | +1.22 pp | leve vantagem, dentro do intervalo de confianca |
| ASSIN Entailment | -3.68 pp | perda significativa |
| ASSIN Paraphrase | +4.15 pp | ganho significativo |
| ENEM | -1.88 pp | perda pequena, dentro do intervalo |

### Deltas vs Tucano-630M (tier acima)

| Benchmark | Maracatu-20M vs Tucano-630M | Interpretacao |
|---|---|---|
| Belebele PT | +0.89 pp | empatados estatisticamente |
| ASSIN Entailment | -2.73 pp | perda |
| ASSIN Paraphrase | +5.59 pp | ganho expressivo |
| ENEM | -1.88 pp | empatados estatisticamente |

### Perplexidade em holdout Wikipedia PT

| Modelo | Val Loss | Perplexidade | Notas |
|---|---|---|---|
| **Maracatu-20M** | 3.1703 | **23.81** | holdout 0.5%, Wikipedia PT, step 43500 |
| Tucano-160M (reportado) | -- | ~30 | valor do paper (arxiv:2411.07854), nao rerodado aqui |
| Tucano-630M | -- | nao reportado separadamente | paper cita perp geral, nao especifica corpus |

**Nota:** perplexidade nao e comparavel diretamente entre modelos com vocabularios diferentes (Maracatu usa SentencePiece BPE 16k; Tucano usa tokenizador proprio). Ver secao de analise.

---

## Amostras qualitativas

Com `temperature=0.8`, `top_k=50`, `seed=42`. Todas as saidas sao lowercase (normalizacao do tokenizer).

### Prompt: `"O Brasil é"`

```
o brasil é o méxico e o canadá é o único dos territórios menores que se estende da europa oriental.
a região tem uma área de 8,32 km2. a região está dividida em 3 partes e 6 regiões, que se estendem
até ao planalto central, que inclui o extremo oeste do país.
```

**Análise:** fluência sintática preservada, mas o modelo confunde Brasil com México e Canadá, localiza a região na Europa oriental e dá uma área numérica absurda (8,32 km²). Vocabulário geográfico (planalto central, territórios, região) está presente mas mal articulado semanticamente.

### Prompt: `"Machado de Assis escreveu"`

```
machado de assis escreveu o livro "a obra-prima do autor da obra-prima do autor", que foi escrito em
1998 e publicado em 2001. no brasil, em 2003, o livro foi publicada pela universidade federal de
minas gerais, que viria a ser
```

**Análise:** título do livro é auto-referencial e nonsense ("a obra-prima do autor da obra-prima do autor"). Datas invertidas: Machado morreu em 1908, não em 2001. Concordância errada ("o livro foi publicada"). Captura o frame "autor escreveu livro publicado por editora" mas sem fatos corretos.

### Prompt: `"A capital de Pernambuco é"`

```
a capital de pernambuco é a capital do estado. o município possui vários estabelecimentos de serviços
(faa) e de estabelecimentos de saúde (fisfa) e empresas de transporte civil, gás e gás.
```

**Análise:** resposta tautológica ("a capital é a capital do estado"). Nunca diz "Recife". Siglas inventadas (FAA, FISFA). Repetição literal ("gás e gás"). O modelo associou "capital de estado" com vocabulário de estrutura municipal genérica, sem recuperar a entidade específica.

**Flag qualitativo:** sem toxicidade, viés explícito ou PII detectada nas amostras. Alucinação de fatos é sistemática nessa escala, esperada, não perigosa. Ausência de conteúdo nocivo sugere que o corpus Wikipedia PT é razoavelmente limpo pra base model dessa escala.

---

## Analise

**O Maracatu-20M opera integralmente na regiao de chance aleatoria em todos os benchmarks MCQ.** Em Belebele (4 opcoes, chance=25%), o modelo marca 23.78% -- 1.22 pp abaixo do acaso. Em ENEM (5 opcoes, chance=20%), 19.22% -- 0.78 pp abaixo do acaso. Isso e esperado e documentado: um modelo base de 17M parametros sem instruction tuning e sem escala suficiente para aprender raciocinio de escolha multipla. O benchmark nao e uma medida de falha -- e o ponto de partida honesto da nossa ladder de escalas.

**O resultado mais interessante e o ASSIN Paraphrase: 60.52%, vs 50% de chance e vs 56.37% do Tucano-160M.** Paraphrase detection e uma tarefa de similaridade semantica superficial, e o Maracatu-20M supera o baseline em 9.52 pp acima do acaso (contra 6.37 pp do Tucano-160M). Isso sugere que o treinamento em Wikipedia PT gerou representacoes lexicais coerentes mesmo a 17M params. Importante: ASSIN e binario (2 opcoes), entao o sinal aqui e mais estatisticamente significativo do que em Belebele ou ENEM.

**O Tucano-630M NAO supera o Tucano-160M nos benchmarks rodados.** Em ASSIN Paraphrase, o 630M (54.93%) fica abaixo do 160M (56.37%) e do Maracatu-20M (60.52%). Em ENEM, os dois empatam exatamente (21.10%). Isso indica que, nessa faixa de escala e sem fine-tuning, escala bruta de parametros nao garante ganho automatico em benchmarks PT-BR -- o corpus, o tokenizador e a arquitetura importam tanto quanto o tamanho.

**A perplexidade do Maracatu-20M (23.81) vs Tucano-160M (~30 no paper) NAO pode ser comparada diretamente.** Os vocabularios sao diferentes (16k SPE vs tokenizador Tucano), e os corpora de calculo tambem diferem. A perp do Maracatu e calculada no nosso proprio holdout Wikipedia PT; o valor do Tucano vem do paper em corpus nao especificado identicamente. O numero sugere que a arquitetura Llama-style e o corpus Wikipedia estao bem calibrados para esse nivel de escala, mas nao e evidencia de superioridade qualitativa.

---

## Hiperparametros do Modelo (referencia)

| Campo | Maracatu-20M | Tucano-160M | Tucano-630M |
|---|---|---|---|
| Params totais | 17M | 160M | 630M |
| Contexto | 512 | 2048 | 2048 |
| Vocab | 16000 (SPE) | nao especificado | nao especificado |
| Arquitetura | Llama-style | GPT-2 style | GPT-2 style |
| Corpus | Wikipedia PT | mix PT | mix PT |
| Tokens treinados | ~410M | nao divulgado | nao divulgado |

---

## Proximos passos recomendados

1. **Rodar ENEM 3-shot** para todos os modelos: few-shot frequentemente da ganho expressivo em base models. Determinar se o Maracatu-20M responde ao contexto de few-shot ou se o limite de 512 tokens ja trunca o contexto antes de chegar ao sinal.
2. **Calcular perplexidade cruzada**: rodar o Maracatu-20M e o Tucano-160M no mesmo texto fixo e normalizar por numero de tokens (nao de bytes), para comparacao mais justa entre vocabularios diferentes.
3. **Maracatu-80M**: a gap real em benchmark MCQ so vai aparecer em escala >100M -- recomenda-se nao interpretar esses numeros como indicativo de qualidade da arquitetura ou do corpus, mas sim do nivel de escala atual.
4. **BLUEX (FUVEST+Unicamp)**: nao disponivel nativamente no lm-eval-harness. Requer harness custom similar ao ENEM. Dataset publico: `eduagarcia/bluex_with_images` (nota: contem imagens, precisara de subset texto-only).

---

## Reproducao

```bash
# Instalar lm-eval no venv isolado (uma vez)
git clone --depth=1 https://github.com/EleutherAI/lm-evaluation-harness .cache/lm-evaluation-harness
python3 -m venv .cache/lm-eval-venv
cd .cache/lm-evaluation-harness && ../.cache/lm-eval-venv/bin/pip install -e "[hf]" --quiet

# Rodar todos os benchmarks (ver script em scripts/eval/run_benchmarks.sh)
bash scripts/eval/run_benchmarks.sh maracatu-ai/maracatu-20m

# Resultados salvos em eval_results/<model-slug>/ (gitignored)
```

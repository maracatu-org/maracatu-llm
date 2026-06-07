# Maracatu-20M: PT-BR Benchmarks (2026-04-20)

**Git revision:** `91d3861ef4ba3f23b0c758565bd901faa90aa16f`
**Hardware:** MacBook Pro M2 Pro (16GB), CPU-only (MPS not used by lm-eval)
**Goal:** Establish an honest baseline of Maracatu-20M on PT-BR benchmarks and compare against Tucano-160M and Tucano-630M using the same reproducible pipeline.

---

## Setup

- **Framework:** `lm-evaluation-harness` (EleutherAI) version `0.4.12.dev0`, cloned at `.cache/lm-evaluation-harness/`
- **Isolated venv:** `.cache/lm-eval-venv/` (Python 3.13.7, did not pollute the project `.venv`)
- **Seed:** 42 in all runs
- **Temperature:** T=0 (greedy/loglikelihood) for all MCQ benchmarks -- lm-eval's default behavior for `output_type: multiple_choice`
- **Batch size:** 8 (CPU-safe for models up to 630M)
- **Models evaluated:**
  - `maracatu-labs/maracatu-20m` -- 17M params, 16k vocab, context 512
  - `TucanoBR/Tucano-160m` -- 160M params, canonical BR baseline at the same tier
  - `TucanoBR/Tucano-630m` -- 630M params, next tier up, to measure the scale gap

---

## Benchmarks and configurations

### 1. Belebele PT (`belebele_por_Latn`)
- **Source:** `facebook/belebele` (native in lm-eval-harness)
- **Format:** 4-option MCQ, reading and text comprehension
- **Split:** test (900 questions)
- **Fewshot:** 0-shot
- **Metric:** `acc` (accuracy)
- **Random chance:** 25.00%
- **Maracatu-20M time:** ~30s

### 2. ASSIN Entailment + Paraphrase
- **Source:** `nilc-nlp/assin` (native in lm-eval-harness)
- **Format:** binary MCQ (2 options) -- textual inference and paraphrase
- **Split:** test (4000 samples each)
- **Fewshot:** 0-shot
- **Metric:** `acc`
- **Random chance:** 50.00%
- **Maracatu-20M time:** ~4min

### 3. ENEM Challenge (`enem_challenge`) -- custom harness
- **Source:** `eduagarcia/enem_challenge` (Hugging Face)
- **Format:** 5-option MCQ (A-E), real ENEM questions 2009-2023
- **Total:** 1431 valid questions (1 annulled question filtered via `process_docs`)
- **Fewshot:** 0-shot
- **Task YAML:** `scripts/eval/tasks/enem_challenge/enem_challenge.yaml`
- **Metric:** `acc`, `acc_norm`
- **Random chance:** 20.00%
- **Maracatu-20M time:** ~1min30s | Tucano-630M: ~25min (CPU, 630M params)
- **Truncation note:** long questions (>512 tokens with context) are truncated from the left by the harness -- affects a minority of questions

---

## Results

### Comparative table

| Model | Params | Belebele PT (4-opt) | ASSIN Entailment (2-opt) | ASSIN Paraphrase (2-opt) | ENEM (5-opt) |
|---|---|---|---|---|---|
| **Random chance** | -- | 25.00% | 50.00% | 50.00% | 20.00% |
| **Maracatu-20M** | 17M | **23.78%** (+/- 1.42%) | 27.67% (+/- 0.71%) | **60.52%** (+/- 0.77%) | 19.22% (+/- 1.04%) |
| **Tucano-160M** | 160M | 22.56% (+/- 1.39%) | **31.35%** (+/- 0.73%) | 56.37% (+/- 0.78%) | **21.10%** (+/- 1.08%) |
| **Tucano-630M** | 630M | 22.89% (+/- 1.40%) | 30.40% (+/- 0.73%) | 54.93% (+/- 0.79%) | **21.10%** (+/- 1.08%) |

### Deltas vs Tucano-160M (canonical baseline)

| Benchmark | Maracatu-20M vs Tucano-160M | Interpretation |
|---|---|---|
| Belebele PT | +1.22 pp | slight edge, within the confidence interval |
| ASSIN Entailment | -3.68 pp | significant loss |
| ASSIN Paraphrase | +4.15 pp | significant gain |
| ENEM | -1.88 pp | small loss, within the confidence interval |

### Deltas vs Tucano-630M (next tier up)

| Benchmark | Maracatu-20M vs Tucano-630M | Interpretation |
|---|---|---|
| Belebele PT | +0.89 pp | statistical tie |
| ASSIN Entailment | -2.73 pp | loss |
| ASSIN Paraphrase | +5.59 pp | meaningful gain |
| ENEM | -1.88 pp | statistical tie |

### Perplexity on Wikipedia PT holdout

| Model | Val Loss | Perplexity | Notes |
|---|---|---|---|
| **Maracatu-20M** | 3.1703 | **23.81** | 0.5% holdout, Wikipedia PT, step 43500 |
| Tucano-160M (reported) | -- | ~30 | value from the paper (arxiv:2411.07854), not rerun here |
| Tucano-630M | -- | not reported separately | paper cites overall perp, doesn't specify the corpus |

**Note:** perplexity is not directly comparable between models with different vocabularies (Maracatu uses SentencePiece BPE 16k; Tucano uses its own tokenizer). See analysis section.

---

## Qualitative samples

With `temperature=0.8`, `top_k=50`, `seed=42`. All outputs are lowercase (tokenizer normalization).

### Prompt: `"O Brasil é"`

```
o brasil é o méxico e o canadá é o único dos territórios menores que se estende da europa oriental.
a região tem uma área de 8,32 km2. a região está dividida em 3 partes e 6 regiões, que se estendem
até ao planalto central, que inclui o extremo oeste do país.
```

**Analysis:** syntactic fluency preserved, but the model conflates Brazil with Mexico and Canada, places the region in eastern Europe and gives an absurd numeric area (8.32 km²). Geographic vocabulary (planalto central, territórios, região) is present but poorly articulated semantically.

### Prompt: `"Machado de Assis escreveu"`

```
machado de assis escreveu o livro "a obra-prima do autor da obra-prima do autor", que foi escrito em
1998 e publicado em 2001. no brasil, em 2003, o livro foi publicada pela universidade federal de
minas gerais, que viria a ser
```

**Analysis:** the book title is self-referential and nonsense ("the author's masterpiece's author's masterpiece"). Dates inverted: Machado died in 1908, not in 2001. Agreement error ("o livro foi publicada"). Captures the frame "author wrote book published by publisher" but with no correct facts.

### Prompt: `"A capital de Pernambuco é"`

```
a capital de pernambuco é a capital do estado. o município possui vários estabelecimentos de serviços
(faa) e de estabelecimentos de saúde (fisfa) e empresas de transporte civil, gás e gás.
```

**Analysis:** tautological answer ("the capital is the capital of the state"). Never says "Recife". Invented acronyms (FAA, FISFA). Literal repetition ("gás e gás"). The model associated "state capital" with generic municipal-structure vocabulary, without recovering the specific entity.

**Qualitative flag:** no toxicity, explicit bias or detected PII in the samples. Factual hallucination is systematic at this scale, expected, not dangerous. The absence of harmful content suggests the Wikipedia PT corpus is reasonably clean for a base model at this scale.

---

## Analysis

**The Maracatu-20M operates entirely in the random-chance region across all MCQ benchmarks.** On Belebele (4 options, chance=25%), the model scores 23.78% -- 1.22 pp below chance. On ENEM (5 options, chance=20%), 19.22% -- 0.78 pp below chance. This is expected and documented: a 17M base model without instruction tuning and without sufficient scale cannot learn multiple-choice reasoning. The benchmark is not a measure of failure -- it's the honest starting point of our scale ladder.

**The most interesting result is ASSIN Paraphrase: 60.52%, vs 50% chance and vs 56.37% for Tucano-160M.** Paraphrase detection is a task of surface semantic similarity, and Maracatu-20M beats the baseline by 9.52 pp above chance (vs 6.37 pp for Tucano-160M). This suggests that training on Wikipedia PT produced coherent lexical representations even at 17M params. Important: ASSIN is binary (2 options), so the signal here is more statistically significant than on Belebele or ENEM.

**Tucano-630M does NOT beat Tucano-160M on the benchmarks we ran.** On ASSIN Paraphrase, the 630M (54.93%) lands below the 160M (56.37%) and the Maracatu-20M (60.52%). On ENEM, the two tie exactly (21.10%). This indicates that at this scale range and without fine-tuning, raw parameter scale does not automatically guarantee gains on PT-BR benchmarks -- corpus, tokenizer and architecture matter as much as size.

**Maracatu-20M perplexity (23.81) vs Tucano-160M (~30 in the paper) CANNOT be compared directly.** The vocabularies differ (16k SPE vs Tucano's tokenizer), and the corpora used for the calculation also differ. Maracatu's perp is computed on our own Wikipedia PT holdout; Tucano's value is from the paper on a not-identically-specified corpus. The number suggests the Llama-style architecture and Wikipedia corpus are well calibrated for this scale level, but it is not evidence of qualitative superiority.

---

## Model hyperparameters (reference)

| Field | Maracatu-20M | Tucano-160M | Tucano-630M |
|---|---|---|---|
| Total params | 17M | 160M | 630M |
| Context | 512 | 2048 | 2048 |
| Vocab | 16000 (SPE) | not specified | not specified |
| Architecture | Llama-style | GPT-2 style | GPT-2 style |
| Corpus | Wikipedia PT | PT mix | PT mix |
| Tokens trained | ~410M | not disclosed | not disclosed |

---

## Recommended next steps

1. **Run ENEM 3-shot** for all models: few-shot often yields meaningful gains for base models. Determine whether Maracatu-20M responds to the few-shot context or whether the 512-token limit truncates the context before reaching the signal.
2. **Compute cross perplexity**: run Maracatu-20M and Tucano-160M on the same fixed text and normalize per number of tokens (not bytes), for a fairer comparison between different vocabularies.
3. **Maracatu-80M**: the real MCQ benchmark gap will only show up at scale >100M -- these numbers should not be interpreted as indicating quality of the architecture or corpus, but rather the current scale level.
4. **BLUEX (FUVEST+Unicamp)**: not natively available in lm-eval-harness. Requires a custom harness similar to ENEM. Public dataset: `eduagarcia/bluex_with_images` (note: contains images, will need a text-only subset).

---

## Reproduction

```bash
# Install lm-eval in the isolated venv (once)
git clone --depth=1 https://github.com/EleutherAI/lm-evaluation-harness .cache/lm-evaluation-harness
python3 -m venv .cache/lm-eval-venv
cd .cache/lm-evaluation-harness && ../.cache/lm-eval-venv/bin/pip install -e "[hf]" --quiet

# Run all benchmarks (see script in scripts/eval/run_benchmarks.sh)
bash scripts/eval/run_benchmarks.sh maracatu-labs/maracatu-20m

# Results saved to eval_results/<model-slug>/ (gitignored)
```

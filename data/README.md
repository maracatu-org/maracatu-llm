# Maracatu training data

This directory contains the corpus used to train Maracatu. Because the files are large, the data is not versioned in Git. Use the scripts in `scripts/` to download and process it.

## Structure

```
data/
├── raw/         # Original dumps (gitignored)
└── processed/   # Cleaned corpus, ready for tokenization (gitignored)
```

## Sources and licenses

All sources below have licenses compatible with use in model training and with redistribution of the resulting weights.

### 1. Portuguese Wikipedia

- **License**: Creative Commons Attribution-ShareAlike 4.0 (CC BY-SA 4.0)
- **Source**: `wikimedia/wikipedia` dataset on Hugging Face (https://huggingface.co/datasets/wikimedia/wikipedia), config `20231101.pt`
- **Approximate size**: ~1-2 GB of already-extracted text
- **Local cache**: `~/.cache/huggingface/datasets/` (not versioned)
- **Attribution**: Texts are authored by the Wikipedia community and its contributors

### 2. Project Gutenberg: works in the Brazilian public domain

- **License**: Public domain
- **Source**: https://www.gutenberg.org/
- **Authors included** (examples):
  - Machado de Assis (1839-1908)
  - José de Alencar (1829-1877)
  - Aluísio Azevedo (1857-1913)
  - Castro Alves (1847-1871)
  - Euclides da Cunha (1866-1909)
  - Gonçalves Dias (1823-1864)

## Sources and licenses (corpus v2)

Corpus v2 expands to three sources, all with licenses compatible with Apache 2.0 and with redistribution of the model weights. It is produced by `scripts/build_corpus_v2.py`. Target: ~1.7B tokens (Chinchilla-optimal for Maracatu-80M).

### 1. Wikipedia PT (kept from v1)

- **License**: CC BY-SA 3.0
- **HF source**: `wikimedia/wikipedia`, config `20231101.pt`
- **Estimated tokens**: ~550M

### 2. Project Gutenberg PT

- **License**: Public Domain
- **Source**: https://www.gutenberg.org/
- **Inclusion criterion**: authors deceased for more than 70 years (Brazilian and international criterion)
- **Authors included**: Machado de Assis, José de Alencar, Aluísio Azevedo, Euclides da Cunha, Castro Alves, Álvares de Azevedo, Casimiro de Abreu, Eça de Queirós, Graciliano Ramos, Lima Barreto, Monteiro Lobato, Olavo Bilac, Gonçalves Dias, Raul Pompeia, Visconde de Taunay
- **Estimated tokens**: ~150M
- **Rate limit**: 1 req/s to avoid overloading the server

### 3. CulturaX PT

- **License**: ODC-BY 1.0 (Open Data Commons Attribution License)
- **HF source**: `uonlp/CulturaX`, subset `pt`, https://huggingface.co/datasets/uonlp/CulturaX
- **Ingestion**: streaming (full dataset is tens of GB)
- **Estimated tokens**: ~1B after filtering

**Mandatory attribution (ODC-BY)**: this corpus includes data from CulturaX (Nguyen et al., 2023), available at https://huggingface.co/datasets/uonlp/CulturaX, under ODC-BY 1.0 license.

### Filter pipeline (v2)

Applied per document in sequence:

1. `min_doc_chars=200`: drops very short documents
2. Language heuristic: minimum proportion of PT stopwords (ratio >= 0.05)
3. MinHash LSH (Jaccard >= 0.85, 128 permutations): fuzzy deduplication across documents
4. Line-by-line cleanup: `min_line_chars=30` + removal of symbol-only lines
5. PII regex: drops lines with CPF, email, BR phone, CEP, address pattern
6. Exact SHA-1 per line: exact deduplication shared across all sources

### Reproducibility (v2)

```bash
# Full corpus v2 (takes hours, requires ~50GB on disk)
caffeinate -is python -u scripts/build_corpus_v2.py --source all

# Smoke test (1,000 docs per source, a few minutes, validates the pipeline)
python -u scripts/build_corpus_v2.py --smoke-test

# Wikipedia only (equivalent to corpus v1 with v2 filters)
python -u scripts/build_corpus_v2.py --source wikipedia
```

Records in `data/processed/MANIFEST_v2.txt` the SHA-256 of the corpus, all filter parameters and per-source statistics. Two runs with the same parameters and the same HF dataset versions produce an identical SHA-256.

## What is NOT included (and why)

The sources below are commonly used in LLM research, but were deliberately excluded out of caution:

- **Corpus Carolina (USP)**: CC BY-NC-4.0. The "non-commercial" clause is incompatible with Apache 2.0 and with future commercialization plans. Re-enable only if the scope changes to non-commercial research, after legal confirmation.
- **OSCAR / Common Crawl (pt) raw**: open web dumps that may contain content with active copyright. Indirect use via CulturaX-PT (already academically filtered).
- **BrWaC**: Brazilian academic corpus, use permitted for research but redistribution in the weights requires formalization.
- **Books with active copyright**: never. Regardless of availability.
- **Social media content**: privacy issues (PII), user copyright, and platform ToS.

## Corpus statistics (to be filled in after processing)

- Total tokens: TBD
- Documents: TBD
- Vocabulary: 16,000 (BPE via SentencePiece)

## Reproducibility

To reproduce exactly the corpus used in training:

```bash
python scripts/clean_corpus.py
```

The script downloads the dataset via Hugging Face (pinned version `20231101.pt` by default) and records in `data/processed/MANIFEST.txt` the exact source, the filter parameters used and the statistics of the resulting corpus.

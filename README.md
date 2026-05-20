# Maracatu

**Brazilian LLMs, trained from scratch, in Portuguese, by Brazilians.**

Open source project for pretraining language models in Brazilian Portuguese, with open weights under Apache 2.0 and a focus on national AI sovereignty.

[maracatu.org](https://maracatu.org) · [Hugging Face](https://huggingface.co/maracatu-ai) · [Contributing](CONTRIBUTING.md) · [Code of Conduct](CODE_OF_CONDUCT.md) · [Security](SECURITY.md)

## Released models

| Model | Parameters | Val Perplexity | Corpus | Hugging Face | Ollama |
|-------|:----------:|:--------------:|--------|--------------|--------|
| **Maracatu-20M** | 17M | 23.81 | Wikipedia PT (~550M tok) | [maracatu-ai/maracatu-20m](https://huggingface.co/maracatu-ai/maracatu-20m) | [whereisanzi/maracatu-20m](https://ollama.com/whereisanzi/maracatu-20m) |
| **Maracatu-80M** | 87.8M | 21.34 | Wiki + Gutenberg + CulturaX-PT (~1.6B tok) | [maracatu-ai/maracatu-80m](https://huggingface.co/maracatu-ai/maracatu-80m) | [whereisanzi/maracatu-80m](https://ollama.com/whereisanzi/maracatu-80m) |

See [MODEL_CARD.md](MODEL_CARD.md) for technical details.

## Architecture

Decoder-only transformer, Llama-style, with modern components:

- RMSNorm · RoPE · SwiGLU · no bias in `nn.Linear` · weight tying
- State dict aligned with Hugging Face's `LlamaForCausalLM` — loads via `transformers` with no conversion script
- SentencePiece BPE 16k tokenizer trained on PT-BR
- Framework: PyTorch

## Corpus

Only sources with licenses compatible with Apache 2.0:

- **Wikipedia PT** — CC BY-SA (979k articles, ~550M BPE tokens)
- **Project Gutenberg** — public domain (Machado de Assis, José de Alencar, etc.)
- **CulturaX-PT** — subset filtered for PT-BR

Details in [`data/README.md`](data/README.md). Preparation pipelines in [`scripts/`](scripts/).

## Quickstart

Requires Python 3.11+ and PyTorch 2.2+. For GPU training, see [`docs/kaggle.md`](docs/kaggle.md) or [`docs/runpod.md`](docs/runpod.md).

```bash
git clone git@github.com:maracatu-labs/maracatu.git
cd maracatu

python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### Prepare corpus

```bash
python scripts/clean_corpus.py
```

Downloads Wikipedia PT (via `datasets`) to `~/.cache/huggingface/`, cleans it, and writes to `data/processed/corpus.txt`.

### Train tokenizer

```bash
python tokenizer/train_tokenizer.py
```

### Train model

```bash
python -m maracatu.train --config configs/maracatu_20m.yaml
python -m maracatu.train --config configs/maracatu_80m.yaml --device cuda
```

### Generate text

```bash
python -m maracatu.sample --checkpoint checkpoints/latest.pt --prompt "O Brasil é"
```

## Experiments

Chronological log of training runs, metrics, and analyses in [`docs/experiments/`](docs/experiments/).

## Evaluation

Benchmarks on Brazilian exams (ENEM, ASSIN), via `lm-evaluation-harness`:

```bash
bash scripts/eval/run_benchmarks.sh
```

Custom tasks in `scripts/eval/tasks/`.

## Structure

```
maracatu/
├── src/maracatu/    # Model, training, generation
├── tokenizer/       # SentencePiece tokenizer training
├── scripts/         # Corpus preparation, eval, deploy
├── configs/         # Hyperparameters (YAML)
├── data/            # Corpus (gitignored, see data/README.md)
├── checkpoints/     # Weights (gitignored)
├── docs/            # Technical docs, experiments, deploy
├── notebooks/       # Exploration
└── MODEL_CARD.md
```

## Publishing

Publishing pipelines for Hugging Face, Ollama, and Kaggle in `scripts/publish_all.sh` and `scripts/export_*.{py,sh}`. Operational details in [`docs/publishing.md`](docs/publishing.md).

## Contributing

Every contribution is welcome — code, corpus improvements, new benchmarks, bug reports. Read [CONTRIBUTING.md](CONTRIBUTING.md) for the PR workflow and [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) for what we expect from the community environment.

Found a vulnerability? See [SECURITY.md](SECURITY.md) before opening a public issue.

## License

Code and weights under [Apache License 2.0](LICENSE).

## Acknowledgments

- Andrej Karpathy for [nanoGPT](https://github.com/karpathy/nanoGPT) — indispensable pedagogical foundation
- Brazilian AI community (Maritaca, WideLabs, LNCC, USP, Unicamp)
- [Tucano](https://huggingface.co/TucanoBR) for the public baseline reference in PT-BR
- Brazilian Artificial Intelligence Plan (PBIA)

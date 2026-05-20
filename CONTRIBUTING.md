# Contributing to Maracatu

Thanks for your interest in contributing! Maracatu is an open source effort to build Brazilian LLMs, with open weights under Apache 2.0. The scope is broad — code, corpus, eval, training infrastructure, documentation, bug reports. Every form of help counts.

## Before you start

- Read the [Code of Conduct](CODE_OF_CONDUCT.md).
- Found a vulnerability or serious model issue (e.g. training data leakage, dangerous content generation)? Don't open a public issue. Follow [SECURITY.md](SECURITY.md).
- For large changes (model refactor, architecture swap, new corpus), open an issue first to discuss the approach.

## Areas where contributions matter most

| Area | Examples |
|------|----------|
| **Model** | Efficiency optimizations, new components (attention, normalization), footprint reductions |
| **Corpus** | PT-BR sources with compatible licensing, quality filters, deduplication |
| **Tokenizer** | Vocabulary experiments, coverage analysis, BPE vs Unigram |
| **Training** | Stability, learning rate schedules, mixed precision, gradient accumulation |
| **Eval** | New PT-BR benchmarks (ENEM, OAB, BLUEX, POSCOMP, Revalida), custom tasks |
| **Deploy** | Quantization, export (GGUF, ONNX), embeddings, optimized inference |
| **Docs** | Technical documentation, guides, usage examples |
| **Hardware** | Training reports on different GPUs (T4, A100, H100, MPS), throughput benchmarks |

## Running locally

Requires Python 3.11+ and PyTorch 2.2+.

```bash
git clone git@github.com:maracatu-labs/maracatu.git
cd maracatu

python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

To train on cloud GPUs (Kaggle T4, Modal, RunPod): see [`docs/kaggle.md`](docs/kaggle.md) and [`docs/runpod.md`](docs/runpod.md).

## PR workflow

1. Fork the repository.
2. Create a branch from `main` with a descriptive name (`feat/...`, `fix/...`, `docs/...`, `data/...`, `eval/...`).
3. Make your commits following the [convention below](#commit-messages).
4. If you introduced new code, add tests in `tests/` where applicable.
5. If you touched model/training, document the experiment in `docs/experiments/`.
6. Open the PR describing the problem, the solution, and how to test.
7. Wait for review. Every contribution goes through code review.

## Commit messages

Use [Conventional Commits](https://www.conventionalcommits.org/) **in English**:

```
feat(model): add gradient checkpointing for 80M config
fix(train): correct lr warmup for resumed runs
data(corpus): add Gutenberg PT subset cleanup script
eval(enem): add zero-shot evaluation task
docs: link Kaggle setup guide in quickstart
refactor(tokenizer): consolidate BPE training entry points
chore: bump torch to 2.4
```

Accepted types: `feat`, `fix`, `data`, `eval`, `docs`, `refactor`, `test`, `chore`, `perf`.
Common scopes: `model`, `train`, `data`, `tokenizer`, `eval`, `deploy`, `infra`.

We use **squash merge** — your PR title and description will end up as the single commit on `main`, so write both in English too.

**Don't include automatic `Co-Authored-By:` trailers** (from AI tools, for example). Add co-authorship only when another person actually collaborated on the commit.

Identifiers in code (variables, functions, classes) are in English. OSS documentation (README, this guide, code of conduct, security policy, issue/PR templates) is in English. Any user-facing strings produced by the trained model itself remain in Brazilian Portuguese — the model targets PT-BR by design.

## Code conventions

### Python

- Use `ruff` for lint (configured in `pyproject.toml`).
- Type hints when the type isn't obvious.
- No comments in code beyond docstrings — clear names matter more.
- Hyperparameter configs in YAML (`configs/`), not hardcoded.
- New experiments: document in `docs/experiments/YYYY-MM-DD-name.md` (template in `docs/experiments/_TEMPLATE.md`).

### Model

- State dict aligned with Hugging Face's `LlamaForCausalLM` — don't break this compatibility without discussing first.
- Modern components: RMSNorm, RoPE, SwiGLU, no bias in `nn.Linear`, weight tying.

### Corpus

- Only sources with licenses compatible with Apache 2.0 (CC BY-SA, CC0, public domain).
- Reproducible preparation script in `scripts/`.
- Document source, license, and processing in `data/README.md`.

## Adding a new benchmark

1. Add the task in `scripts/eval/tasks/<name>/` in `lm-evaluation-harness` format.
2. Document the task: what it evaluates, prompt format, metrics.
3. Include it in the `scripts/eval/run_benchmarks.sh` script.
4. Report results in a PR or in `docs/experiments/`.

## Reporting bugs

Use the issue template. Include:
- Python, PyTorch, CUDA versions (if applicable)
- Hardware (GPU, RAM, etc.)
- Steps to reproduce
- What you expected vs. what happened
- Relevant logs

For training bugs, attach the config YAML + run logs.

## Questions

Open an issue with the `question` label or start a discussion on GitHub.

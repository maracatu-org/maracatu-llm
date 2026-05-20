<div align="center">

# Maracatu

**LLMs brasileiros, treinados do zero, em português, por brasileiros.**

Projeto open source de pré-treino de modelos de linguagem em português brasileiro, com pesos abertos sob Apache 2.0 e foco em soberania nacional em IA.

[maracatu.org](https://maracatu.org) · [Hugging Face](https://huggingface.co/maracatu-ai) · [Contribuindo](CONTRIBUTING.md) · [Código de Conduta](CODE_OF_CONDUCT.md) · [Segurança](SECURITY.md)

</div>

---

## Modelos publicados

| Modelo | Parâmetros | Val Perplexidade | Corpus | Hugging Face | Ollama |
|--------|:----------:|:----------------:|--------|--------------|--------|
| **Maracatu-20M** | 17M | 23.81 | Wikipedia PT (~550M tok) | [maracatu-ai/maracatu-20m](https://huggingface.co/maracatu-ai/maracatu-20m) | [whereisanzi/maracatu-20m](https://ollama.com/whereisanzi/maracatu-20m) |
| **Maracatu-80M** | 87.8M | 21.34 | Wiki + Gutenberg + CulturaX-PT (~1.6B tok) | [maracatu-ai/maracatu-80m](https://huggingface.co/maracatu-ai/maracatu-80m) | em breve |

Ver [MODEL_CARD.md](MODEL_CARD.md) para detalhes técnicos.

## Roadmap

Escada de releases, cada um ~10× maior que o anterior:

- [x] **Maracatu-20M** — validação de pipeline (Wikipedia PT, T4 Kaggle)
- [x] **Maracatu-80M** — corpus expandido 1.6B tokens, supera Tucano-160M em PP
- [ ] **Maracatu-800M** — primeiro modelo conversacional, instruction tuning
- [ ] **Maracatu-8B** — competitivo com Llama-3-8B em benchmarks PT-BR
- [ ] **Maracatu-80B** — *North Star* — paridade com Llama-3.1-70B em benchmarks brasileiros

## Arquitetura

Decoder-only transformer, estilo Llama, com componentes modernos:

- RMSNorm · RoPE · SwiGLU · sem bias em `nn.Linear` · weight tying
- State dict alinhado com `LlamaForCausalLM` do Hugging Face — carrega via `transformers` sem script de conversão
- Tokenizer SentencePiece BPE 16k treinado em PT-BR
- Framework: PyTorch

## Corpus

Apenas fontes com licença compatível com Apache 2.0:

- **Wikipedia PT** — CC BY-SA (979k artigos, ~550M tokens BPE)
- **Projeto Gutenberg** — domínio público (Machado de Assis, José de Alencar, etc.)
- **CulturaX-PT** — subset filtrado para PT-BR

Detalhes em [`data/README.md`](data/README.md). Pipelines de preparação em [`scripts/`](scripts/).

## Quickstart

Requer Python 3.11+ e PyTorch 2.2+. Para treino em GPU, consulte [`docs/kaggle.md`](docs/kaggle.md) ou [`docs/runpod.md`](docs/runpod.md).

```bash
git clone git@github.com:maracatu-labs/maracatu.git
cd maracatu

python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### Preparar corpus

```bash
python scripts/clean_corpus.py
```

Baixa Wikipedia PT (via `datasets`) para `~/.cache/huggingface/`, limpa e escreve em `data/processed/corpus.txt`.

### Treinar tokenizer

```bash
python tokenizer/train_tokenizer.py
```

### Treinar modelo

```bash
python -m maracatu.train --config configs/maracatu_20m.yaml
python -m maracatu.train --config configs/maracatu_80m.yaml --device cuda
```

### Gerar texto

```bash
python -m maracatu.sample --checkpoint checkpoints/latest.pt --prompt "O Brasil é"
```

## Experimentos

Registro cronológico dos runs de treino, métricas e análises em [`docs/experiments/`](docs/experiments/).

## Avaliação

Benchmarks em exames brasileiros (ENEM, ASSIN), via `lm-evaluation-harness`:

```bash
bash scripts/eval/run_benchmarks.sh
```

Tasks customizadas em `scripts/eval/tasks/`.

## Estrutura

```
maracatu/
├── src/maracatu/    # Modelo, treino, geração
├── tokenizer/       # Treino do tokenizer SentencePiece
├── scripts/         # Preparação de corpus, eval, deploy
├── configs/         # Hiperparâmetros (YAML)
├── data/            # Corpus (gitignored, ver data/README.md)
├── checkpoints/     # Pesos (gitignored)
├── docs/            # Documentação técnica, experimentos, deploy
├── notebooks/       # Exploração
└── MODEL_CARD.md
```

## Publicação

Pipelines de publicação para Hugging Face, Ollama e Kaggle em `scripts/publish_all.sh` e `scripts/export_*.{py,sh}`. Detalhes operacionais em [`docs/publishing.md`](docs/publishing.md).

## Contribuindo

Toda contribuição é bem-vinda — código, melhorias de corpus, novos benchmarks, relatos de bug. Leia [CONTRIBUTING.md](CONTRIBUTING.md) para o fluxo de PRs e [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) para o que esperamos do ambiente da comunidade.

Encontrou uma vulnerabilidade? Veja [SECURITY.md](SECURITY.md) antes de abrir uma issue pública.

## Licença

Código e pesos sob [Apache License 2.0](LICENSE).

## Agradecimentos

- Andrej Karpathy pelo [nanoGPT](https://github.com/karpathy/nanoGPT) — base pedagógica indispensável
- Comunidade brasileira de IA (Maritaca, WideLabs, LNCC, USP, Unicamp)
- [Tucano](https://huggingface.co/TucanoBR) pela referência pública de baselines em PT-BR
- Plano Brasileiro de Inteligência Artificial (PBIA)

# Contribuindo com o Maracatu

Obrigado pelo interesse em contribuir! Maracatu é um esforço open source de criação de LLMs brasileiros, com pesos abertos sob Apache 2.0. O escopo é amplo — código, corpus, eval, infraestrutura de treino, documentação, relatos de bugs. Toda forma de ajuda conta.

## Antes de começar

- Leia o [Código de Conduta](CODE_OF_CONDUCT.md).
- Encontrou uma vulnerabilidade ou problema sério de modelo (ex: vazamento de dados de treino, geração de conteúdo perigoso)? Não abra issue pública. Siga o [SECURITY.md](SECURITY.md).
- Para mudanças grandes (refatoração de modelo, troca de arquitetura, novo corpus), abra uma issue antes para discutir a abordagem.

## Áreas onde contribuições fazem diferença

| Área | Exemplos |
|------|----------|
| **Modelo** | Otimizações de eficiência, novos componentes (atenção, normalização), reduções de footprint |
| **Corpus** | Fontes em PT-BR com licença compatível, filtros de qualidade, deduplicação |
| **Tokenizer** | Experimentos de vocabulário, análise de cobertura, BPE vs Unigram |
| **Treino** | Estabilidade, schedules de learning rate, mixed precision, gradient accumulation |
| **Eval** | Novos benchmarks PT-BR (ENEM, OAB, BLUEX, POSCOMP, Revalida), tasks customizadas |
| **Deploy** | Quantização, exportação (GGUF, ONNX), embeddings, inferência otimizada |
| **Docs** | Documentação técnica, guias, exemplos de uso |
| **Hardware** | Relatos de treino em diferentes GPUs (T4, A100, H100, MPS), benchmarks de throughput |

## Como rodar localmente

Requer Python 3.11+ e PyTorch 2.2+.

```bash
git clone git@github.com:maracatu-labs/maracatu.git
cd maracatu

python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Para treinar em GPU na nuvem (Kaggle T4, Modal, RunPod): ver [`docs/kaggle.md`](docs/kaggle.md) e [`docs/runpod.md`](docs/runpod.md).

## Fluxo de PR

1. Faça fork do repositório.
2. Crie um branch a partir de `main` com nome descritivo (`feat/...`, `fix/...`, `docs/...`, `data/...`, `eval/...`).
3. Faça seus commits seguindo a [convenção abaixo](#commit-messages).
4. Se introduziu código novo, adicione testes em `tests/` quando aplicável.
5. Se mexeu em modelo/treino, documente o experimento em `docs/experiments/`.
6. Abra o PR descrevendo o problema, a solução e como testar.
7. Aguarde revisão. Toda contribuição passa por code review.

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

Identifiers in code (variables, functions, classes) are in English. User-facing content (README, this guide, error messages aimed at Portuguese-speaking contributors) stays in Brazilian Portuguese.

## Convenções de código

### Python

- Use `ruff` para lint (configurado em `pyproject.toml`).
- Type hints quando o tipo não é óbvio.
- Sem comentários em código além de docstrings — nomes claros valem mais.
- Configs de hiperparâmetros em YAML (`configs/`), não hardcoded.
- Experimentos novos: documentar em `docs/experiments/AAAA-MM-DD-nome.md` (template em `docs/experiments/_TEMPLATE.md`).

### Modelo

- State dict alinhado com `LlamaForCausalLM` do Hugging Face — não quebre essa compatibilidade sem discutir antes.
- Componentes modernos: RMSNorm, RoPE, SwiGLU, sem bias em `nn.Linear`, weight tying.

### Corpus

- Apenas fontes com licença compatível com Apache 2.0 (CC BY-SA, CC0, domínio público).
- Script de preparação reprodutível em `scripts/`.
- Documentar fonte, licença e processamento em `data/README.md`.

## Adicionando um novo benchmark

1. Adicione a task em `scripts/eval/tasks/<nome>/` no formato `lm-evaluation-harness`.
2. Documente a task: o que avalia, formato dos prompts, métricas.
3. Inclua no script `scripts/eval/run_benchmarks.sh`.
4. Reporte os resultados num PR ou em `docs/experiments/`.

## Reportando bugs

Use o template de issue. Inclua:
- Versão do Python, PyTorch, CUDA (se aplicável)
- Hardware (GPU, RAM, etc.)
- Passos para reproduzir
- O que esperava vs. o que aconteceu
- Logs relevantes

Para bugs de treino, anexe o YAML de config + logs do run.

## Dúvidas

Abra uma issue marcando como `question` ou inicie uma discussion no GitHub.

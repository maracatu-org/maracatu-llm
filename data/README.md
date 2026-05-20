# Dados de treino do Maracatu

Este diretório contém o corpus usado para treinar o Maracatu. Por serem arquivos grandes, os dados não são versionados no Git. Use os scripts em `scripts/` para baixá-los e processá-los.

## Estrutura

```
data/
├── raw/         # Dumps originais (ignorado pelo Git)
└── processed/   # Corpus limpo e pronto para tokenização (ignorado pelo Git)
```

## Fontes e licenças

Todas as fontes abaixo têm licenças compatíveis com uso em treinamento de modelos e redistribuição dos pesos resultantes.

### 1. Wikipedia em Português

- **Licença**: Creative Commons Attribution-ShareAlike 4.0 (CC BY-SA 4.0)
- **Fonte**: dataset `wikimedia/wikipedia` no HuggingFace (https://huggingface.co/datasets/wikimedia/wikipedia), config `20231101.pt`
- **Tamanho aproximado**: ~1-2 GB de texto já extraído
- **Cache local**: `~/.cache/huggingface/datasets/` (não versionado)
- **Atribuição**: Os textos são de autoria da comunidade Wikipedia e seus colaboradores

### 2. Projeto Gutenberg: obras em domínio público brasileiro

- **Licença**: Domínio público
- **Fonte**: https://www.gutenberg.org/
- **Autores incluídos** (exemplos):
  - Machado de Assis (1839-1908)
  - José de Alencar (1829-1877)
  - Aluísio Azevedo (1857-1913)
  - Castro Alves (1847-1871)
  - Euclides da Cunha (1866-1909)
  - Gonçalves Dias (1823-1864)

## Fontes e licenças (corpus v2)

O corpus v2 expande para três fontes, todas com licenças compatíveis com Apache 2.0 e com redistribuição dos pesos do modelo. É produzido por `scripts/build_corpus_v2.py`. Alvo: ~1.7B tokens (Chinchilla-ótimo para o Maracatu-80M).

### 1. Wikipedia PT (mantida do v1)

- **Licença**: CC BY-SA 3.0
- **Fonte HF**: `wikimedia/wikipedia`, config `20231101.pt`
- **Tokens estimados**: ~550M

### 2. Project Gutenberg PT

- **Licença**: Domínio Público
- **Fonte**: https://www.gutenberg.org/
- **Critério de inclusão**: autores falecidos há mais de 70 anos (critério BR e internacional)
- **Autores incluídos**: Machado de Assis, José de Alencar, Aluísio Azevedo, Euclides da Cunha, Castro Alves, Álvares de Azevedo, Casimiro de Abreu, Eça de Queirós, Graciliano Ramos, Lima Barreto, Monteiro Lobato, Olavo Bilac, Gonçalves Dias, Raul Pompeia, Visconde de Taunay
- **Tokens estimados**: ~150M
- **Rate limit**: 1 req/s para não sobrecarregar o servidor

### 3. CulturaX PT

- **Licença**: ODC-BY 1.0 (Open Data Commons Attribution License)
- **Fonte HF**: `uonlp/CulturaX`, subset `pt`, https://huggingface.co/datasets/uonlp/CulturaX
- **Ingestão**: streaming (dataset completo tem dezenas de GB)
- **Tokens estimados**: ~1B após filtragem

**Atribuição obrigatória (ODC-BY)**: este corpus inclui dados do CulturaX (Nguyen et al., 2023), disponível em https://huggingface.co/datasets/uonlp/CulturaX, sob licença ODC-BY 1.0.

### Pipeline de filtros (v2)

Aplicados em cadeia por documento:

1. `min_doc_chars=200`: descarta documentos muito curtos
2. Heurística de idioma: proporção mínima de stopwords PT (ratio >= 0.05)
3. MinHash LSH (Jaccard >= 0.85, 128 permutações): deduplicação fuzzy entre documentos
4. Limpeza linha a linha: `min_line_chars=30` + remoção de linhas só-símbolos
5. PII regex: descarta linhas com CPF, email, telefone BR, CEP, padrão de endereço
6. SHA-1 exata por linha: deduplicação exata compartilhada entre todas as fontes

### Reprodutibilidade (v2)

```bash
# Corpus completo v2 (demora horas, requer ~50GB em disco)
caffeinate -is python -u scripts/build_corpus_v2.py --source all

# Smoke test (1.000 docs por fonte, poucos minutos, valida pipeline)
python -u scripts/build_corpus_v2.py --smoke-test

# Apenas Wikipedia (equivalente ao corpus v1 com filtros v2)
python -u scripts/build_corpus_v2.py --source wikipedia
```

Registra em `data/processed/MANIFEST_v2.txt` o SHA-256 do corpus, todos os parâmetros de filtro e estatísticas por fonte. Duas execuções com mesmos parâmetros e mesmas versões dos datasets HF produzem SHA-256 idêntico.

## O que NÃO está incluído (e por quê)

As fontes abaixo são comumente usadas em pesquisa de LLMs, mas foram deliberadamente excluídas por cautela:

- **Corpus Carolina (USP)**: CC BY-NC-4.0. A cláusula "não comercial" é incompatível com Apache 2.0 e com planos futuros de comercialização. Reativar somente se o escopo mudar para pesquisa sem fins comerciais, mediante confirmação legal.
- **OSCAR / Common Crawl (pt) bruto**: dumps da web aberta que podem conter conteúdo com copyright ativo. Uso indireto via CulturaX-PT (já filtrado academicamente).
- **BrWaC**: corpus acadêmico brasileiro, uso permitido para pesquisa mas redistribuição nos pesos requer formalização.
- **Livros com copyright ativo**: nunca. Independente de disponibilidade.
- **Conteúdo de redes sociais**: questões de privacidade (PII), copyright de usuários e ToS das plataformas.

## Estatísticas do corpus (serão preenchidas após processamento)

- Tokens totais: TBD
- Documentos: TBD
- Vocabulário: 16.000 (BPE via SentencePiece)

## Reprodutibilidade

Para reproduzir exatamente o corpus usado no treino:

```bash
python scripts/clean_corpus.py
```

O script baixa o dataset via HuggingFace (versão fixa `20231101.pt` por padrão) e registra em `data/processed/MANIFEST.txt` a fonte exata, os parâmetros de filtro usados e as estatísticas do corpus resultante.

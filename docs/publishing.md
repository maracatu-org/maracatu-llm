# Maracatu AI: Guia de publicação multi-canal

**Canais**: HuggingFace Hub (primário) + Ollama Hub (secundário) + Kaggle Models (terciário)  
**Naming convention**: `maracatu-ai/maracatu-<scale>`, minúsculo, hífen, sem underscores  
**Última revisão**: 2026-04-19

---

## 1. Hierarquia dos canais

### HuggingFace Hub (primário)

O HF Hub é a plataforma padrão da comunidade de pesquisa e desenvolvimento de LLMs. Publica safetensors (formato canônico, mais seguro que pickle), config.json (LlamaConfig legível por máquina), tokenizer e metadados. É o único canal que recebe **todos** os artefatos: pesos fp32/bf16 originais via safetensors, GGUFs de todas as quantizações, e a documentação (README/model card).

Razão para ser primário: qualquer pessoa com `pip install transformers` pode carregar o modelo com dois comandos. É o ponto de entrada da maioria dos pesquisadores e devs. Também serve como storage centralizado; os GGUFs do Ollama referenciamos a partir daqui.

### Ollama Hub (secundário)

Ollama é a forma mais simples de rodar um LLM localmente: um comando, sem Python, sem CUDA configurado. Atinge um público mais amplo que o HF: devs que querem integrar LLM em apps locais, entusiastas sem experiência em ML. Para o Maracatu, ter um modelo no Ollama Hub aumenta a superfície de adoção, especialmente no Brasil onde poucos LLMs estão disponíveis em PT-BR.

Razão para ser secundário (e não primário): Ollama só serve GGUF quantizado (não os pesos originais). O HF Hub precisa existir antes, porque é de lá que vem o GGUF que o Modelfile aponta ou que subimos.

### Kaggle Models (terciário)

Kaggle Models permite que qualquer pessoa use o modelo diretamente dentro de um Kaggle Notebook, sem download externo. Isso é valioso para o público de pesquisa e estudantes no Brasil: Kaggle free tier tem GPU T4, e poder carregar o Maracatu de dentro do notebook com uma linha facilita experimentos e fine-tuning. É terciário porque a tooling de publicação é a mais trabalhosa das três (ver seção dedicada), e o alcance imediato é menor.

---

## 2. Pré-requisitos

### Ferramentas necessárias

**huggingface-cli**: já incluído no `transformers` que está no `.venv`:

```bash
.venv/bin/huggingface-cli --version
```

**ollama**: instalar via Homebrew (recomendado no macOS):

```bash
brew install ollama
# Verificar:
ollama --version   # deve retornar algo como "ollama version 0.x.y"
```

Alternativa sem Homebrew (curl):

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

**llama.cpp**: clonar e buildar localmente. O build é necessário para dois binários: `llama-quantize` e (opcionalmente) `llama-cli` para smoke test. Requer `cmake >= 3.21` e compilador C++17.

```bash
# Verificar pré-requisitos do build
cmake --version          # precisa ser >= 3.21
clang++ --version        # macOS: clang via Xcode Command Line Tools

# Instalar cmake se ausente
brew install cmake

# Instalar Xcode Command Line Tools se ausente
xcode-select --install
```

Clonar e buildar (fazer uma vez, path `.cache/llama.cpp/` está em `.gitignore`):

```bash
mkdir -p .cache
git clone https://github.com/ggerganov/llama.cpp .cache/llama.cpp
cd .cache/llama.cpp
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j$(sysctl -n hw.logicalcpu)
# Verificar que os binários existem:
ls build/bin/llama-quantize build/bin/llama-cli
```

O `convert_hf_to_gguf.py` do llama.cpp tem suas próprias dependências Python. Instalar em venv separado para não contaminar o `.venv` do projeto:

```bash
python3 -m venv .cache/llama-venv
.cache/llama-venv/bin/pip install -r .cache/llama.cpp/requirements.txt
```

### Autenticação

**HuggingFace**: fazer uma vez por máquina. Requer token com escopo `write`:

```bash
source .env    # carrega HF_TOKEN definido em .env (ver .env.example)
.venv/bin/huggingface-cli login --token "$HF_TOKEN"
```

Para verificar se está autenticado:

```bash
.venv/bin/huggingface-cli whoami
# deve retornar seu username e a org maracatu-ai nas orgs listadas
```

**Ollama Hub**: fazer uma vez por máquina:

```bash
ollama login
# Abre o browser para autenticação em ollama.com
# Requer conta criada em https://ollama.com; o username dessa conta vira o namespace dos modelos
# Ollama não tem conceito de org: o namespace é sempre pessoal (ex: maracatu-ai, maracatuai, whereisanzi)
```

**Kaggle**: já configurado em `~/.kaggle/kaggle.json` (ver [`docs/kaggle.md`](kaggle.md)).

### Namespaces: o que precisa ser criado manualmente antes do primeiro upload

| Canal | Namespace | Criação |
|---|---|---|
| HF Hub | `maracatu-ai/maracatu-<scale>` | automática via `huggingface-cli upload` |
| Ollama Hub (M-20M → M-800M) | `whereisanzi/maracatu-<scale>` | conta pessoal já existente; criado automaticamente pelo `ollama push` |
| Ollama Hub (M-8B+) | `maracatuai/maracatu-<scale>` | username dedicado a reservar agora em ollama.com (Ollama não permite `.` no username; `maracatuai` é a forma válida) |
| Kaggle Models | `whereisanzi/maracatu-<scale>` | **obrigatório criar via UI antes do primeiro upload CLI**: ver Passo 6 |

> **Política de namespace Ollama por escala**:
> - **M-20M → M-1B** (smoke test + primeiros modelos úteis): publicar sob `whereisanzi/maracatu-<scale>` (conta pessoal já existente do Anderson). Aceitável porque ainda não temos tração pública que justifique reservar branding premium.
> - **M-8B em diante** (primeiro modelo sério, pitch de edital): migrar pra username dedicado `maracatuai` (Ollama não aceita `.` nem `-` em todos os casos; `maracatuai` é a forma segura). **Reservar esse username agora em https://ollama.com**, mesmo antes de usar, pra garantir disponibilidade.
> - Modelos antigos ficam no namespace original; novos lançamentos usam o novo. Não tentar "mover" modelo entre namespaces no Ollama Hub: apenas republicar com novo namespace e marcar o antigo como deprecated no model card.
>
> **Assimetria de namespace por canal** é esperada: HF usa a org `maracatu-ai`; Ollama e Kaggle usam username pessoal (nenhum dos dois tem conceito de org). Documentar explicitamente em MODEL_CARD.md pra evitar confusão pros usuários.

---

## 3. Pipeline sequencial completo

```
checkpoint.pt
    └─► [Passo 1] export_hf.py
            └─► exports/maracatu-<scale>-hf/
                    ├── model.safetensors
                    ├── config.json
                    ├── tokenizer.model
                    ├── tokenizer_config.json
                    └── special_tokens_map.json
                        │
                        ├─► [Passo 2] huggingface-cli upload ──────────► HF Hub (safetensors)
                        │
                        └─► [Passo 3] convert_hf_to_gguf.py
                                └─► exports/maracatu-<scale>-gguf/
                                        ├── maracatu-<scale>-fp16.gguf
                                        ├── maracatu-<scale>-Q4_K_M.gguf
                                        ├── maracatu-<scale>-Q5_K_M.gguf
                                        └── maracatu-<scale>-Q8_0.gguf
                                                │
                                                ├─► [Passo 4] huggingface-cli upload ─► HF Hub (gguf/)
                                                │
                                                ├─► [Passo 5] Modelfile + ollama create + ollama push ─► Ollama Hub
                                                │
                                                └─► [Passo 6] Kaggle Models upload (UI ou CLI isolado)
```

---

### Passo 1: Export HF (checkpoint.pt → safetensors)

Referência: [`scripts/export_hf.py`](../scripts/export_hf.py)

```bash
.venv/bin/python scripts/export_hf.py \
    --checkpoint checkpoints/kaggle/best.pt \
    --tokenizer tokenizer/maracatu.model \
    --output-dir exports/maracatu-20m-hf
```

O script valida equivalência numérica dos logits entre nossa implementação PyTorch e a `LlamaForCausalLM` do HF. A saída esperada inclui `max_abs_diff: 0.00e+00` (ou < 1e-3 no pior caso). Não prosseguir se a validação falhar.

Artefatos gerados:

```
exports/maracatu-20m-hf/
├── model.safetensors          # pesos completos (bf16/fp32 conforme o checkpoint)
├── config.json                # LlamaConfig: parâmetros de arquitetura
├── generation_config.json     # gerado pelo save_pretrained
├── tokenizer.model            # SentencePiece BPE 16k
├── tokenizer_config.json      # aponta para LlamaTokenizer slow
└── special_tokens_map.json    # bos/eos/unk/pad tokens
```

**Nota sobre `tokenizer.json` (tokenizer Fast vs Slow):** o `export_hf.py` escreve `"tokenizer_class": "LlamaTokenizer"`, que carrega a versão slow (Python puro via SentencePiece). A versão Fast requer um `tokenizer.json` no formato HF Tokenizers, que exigiria conversão adicional via `convert_slow_tokenizer.py` ou exportação manual. Para o M-20M e M-80M a versão slow é suficiente: `AutoTokenizer.from_pretrained()` funciona corretamente e a diferença de velocidade só importa em serving com alto throughput (M-8B+). Quando chegar essa escala, adicionar o `tokenizer.json` Fast.

**Troubleshooting**

| Erro | Causa | Solução |
|---|---|---|
| `max_abs_diff > 1e-3` | convenção de RoPE ou ordem de heads divergiu | não publicar; comparar implementações |
| `RuntimeError: real_missing keys` | state_dict com prefixo `_orig_mod.` não removido | checkpoint de `torch.compile`; o script já faz `removeprefix`, verificar se `load_state_dict` foi chamado com o dict processado |
| `AutoTokenizer` falha com `None` | `tokenizer_config.json` ausente | verificar se `write_tokenizer_files()` rodou; checar `output-dir` |
| `torch.load` com warning de segurança | `weights_only=False` em PyTorch >= 2.6 | esperado; o script já usa essa flag, não é erro |

---

### Passo 2: Upload HF Hub (safetensors)

```bash
.venv/bin/huggingface-cli upload maracatu-ai/maracatu-20m exports/maracatu-20m-hf .
```

Esse comando cria o repo se não existir e sobe todos os arquivos do diretório. Para modelos maiores (>2GB por arquivo), adicionar `--max-shard-size 2GB` no `save_pretrained` do `export_hf.py` antes de rodar o Passo 1: isso sharda o safetensors automaticamente. O `huggingface-cli` já usa Git LFS para arquivos grandes quando necessário; não requer configuração manual.

Após o upload, configurar via UI do HF Hub (em `huggingface.co/maracatu-ai/maracatu-20m/settings`):
- `pipeline_tag: text-generation`
- `library_name: transformers`
- Adicionar `README.md` baseado em `MODEL_CARD.md` do repo

**Troubleshooting**

| Erro | Causa | Solução |
|---|---|---|
| `401 Unauthorized` | token sem escopo write ou expirado | `huggingface-cli login` com token correto |
| `Repository not found` | org `maracatu-ai` sem permissão no token | verificar que o token pertence à conta dona da org |
| Upload trava em arquivo grande | timeout de rede | `--chunk-size 50000000` no upload |
| `OSError: git-lfs not found` | git-lfs não instalado (necessário para >5GB) | `brew install git-lfs && git lfs install` |

---

### Passo 3: Conversão GGUF e quantização

**Por que GGUF?** GGUF é o formato binário do `llama.cpp`: compacto, auto-descritivo (carrega metadados de arquitetura embutidos) e suportado por praticamente todos os runtimes de inferência locais além do próprio llama.cpp: Ollama, LM Studio, Jan, GPT4All. A conversão parte do HF safetensors (não do checkpoint.pt) para garantir que os pesos que chegam ao GGUF são exatamente os validados no Passo 1.

**Passo 3a: Converter HF para GGUF fp16 (baseline sem perda):**

```bash
.cache/llama-venv/bin/python .cache/llama.cpp/convert_hf_to_gguf.py \
    exports/maracatu-20m-hf \
    --outtype f16 \
    --outfile exports/maracatu-20m-gguf/maracatu-20m-fp16.gguf
```

**Passo 3b: Quantizar (a partir do fp16):**

```bash
mkdir -p exports/maracatu-20m-gguf

QUANTIZE=.cache/llama.cpp/build/bin/llama-quantize
FP16=exports/maracatu-20m-gguf/maracatu-20m-fp16.gguf
OUT=exports/maracatu-20m-gguf

$QUANTIZE "$FP16" "$OUT/maracatu-20m-Q4_K_M.gguf" Q4_K_M
$QUANTIZE "$FP16" "$OUT/maracatu-20m-Q5_K_M.gguf" Q5_K_M
$QUANTIZE "$FP16" "$OUT/maracatu-20m-Q8_0.gguf"   Q8_0
```

**Passo 3c: Gerar checksums e verificar tamanhos:**

```bash
for f in exports/maracatu-20m-gguf/*.gguf; do
    echo "$(shasum -a 256 "$f") | $(du -sh "$f" | cut -f1)"
done
```

Registrar os SHA-256 nas release notes.

**Passo 3d: Smoke test de cada quantização:**

```bash
for quant in Q4_K_M Q5_K_M Q8_0; do
    echo "=== $quant ==="
    .cache/llama.cpp/build/bin/llama-cli \
        -m "exports/maracatu-20m-gguf/maracatu-20m-${quant}.gguf" \
        -p "O Brasil é" \
        -n 30 \
        --log-disable 2>/dev/null
    echo
done
```

**Tamanhos esperados e impacto em qualidade:**

| Quantização | ~20M | ~800M | ~8B | Qualidade vs fp16 |
|---|---|---|---|---|
| fp16 (referência) | ~34MB | ~1GB | ~14GB | baseline |
| Q8_0 | ~18MB | ~540MB | ~7.5GB | perda < 0.1% perplexidade |
| Q5_K_M | ~13MB | ~380MB | ~4.5GB | perda ~0.5%, boa qualidade |
| Q4_K_M | ~10MB | ~290MB | ~3GB | perda ~1-2%, **default recomendado** |

Nota: para modelos pequenos como o 20M, a diferença absoluta de tamanho entre quantizações é de poucos MB. O impacto relativo na qualidade tende a ser maior que em modelos 7B+: considerar subir Q8_0 também como opção "sem compromisso".

**Troubleshooting**

| Erro | Causa | Solução |
|---|---|---|
| `cmake: command not found` | cmake não instalado | `brew install cmake` |
| Build falha em C++17 | compilador antigo | `xcode-select --install` |
| `unsupported model architecture` no convert_hf_to_gguf | llama.cpp desatualizado | `git pull` em `.cache/llama.cpp/` e rebuildar |
| `GGUF load error: magic mismatch` | arquivo corrompido | verificar SHA-256 do fp16 antes de quantizar |
| `convert_hf_to_gguf.py: model type not supported` | version do script anterior a suporte Llama | confirmar que o llama.cpp está na versão >= b3000 (2024-10+) |

---

### Passo 4: Upload GGUF ao HF Hub (mesmo repo)

Os GGUFs vão no mesmo repo dos safetensors, na subpasta `gguf/`. Isso mantém tudo centralizado e facilita a descoberta: quem encontra o repo pelo HF Hub vê os dois formatos num só lugar.

```bash
for quant in Q4_K_M Q5_K_M Q8_0; do
    .venv/bin/huggingface-cli upload \
        maracatu-ai/maracatu-20m \
        "exports/maracatu-20m-gguf/maracatu-20m-${quant}.gguf" \
        "gguf/maracatu-20m-${quant}.gguf"
done
```

Não subir o `fp16.gguf`: é arquivo intermediário do pipeline, já coberto pelos safetensors.

---

### Passo 5: Ollama

#### 5a. Modelfile para modelo base (completion)

O Maracatu é um modelo **base/completion**, não instruct. O Ollama foi desenhado para modelos de chat, mas suporta modelos base via um `TEMPLATE` minimalista que passa o prompt diretamente sem formatação de turno (sem `[INST]`, sem `<|user|>`, sem role markers).

Criar o arquivo `exports/maracatu-20m-gguf/Modelfile`:

```dockerfile
FROM ./maracatu-20m-Q4_K_M.gguf

# Modelo base/completion: sem template de chat
# O input é passado direto ao modelo sem formatação de turno.
TEMPLATE """{{ .Prompt }}"""

# Stop tokens do nosso tokenizer SentencePiece 16k
PARAMETER stop "</s>"
PARAMETER stop "<unk>"

# Parâmetros de geração padrão
PARAMETER temperature 0.8
PARAMETER top_k 50
PARAMETER top_p 0.95
PARAMETER num_ctx 512
```

Notas sobre os campos:
- `num_ctx 512`: contexto máximo definido pelo `max_position_embeddings` do M-20M. Para escalas maiores (800M, 8B), ajustar conforme o config.
- `temperature 0.8` + `top_k 50` + `top_p 0.95`: defaults razoáveis para completion criativo. O usuário pode sobrescrever em runtime com `ollama run ... --temperature`.
- `stop "</s>"`: token EOS do nosso tokenizer (`eos_token_id=3` no LlamaConfig). Sem isso o modelo pode continuar gerando indefinidamente.
- O Ollama pode exibir aviso "no chat template found": é esperado para modelos base, não é erro.

#### 5b. Criar modelo local, testar e publicar

```bash
# Substituir <username> pelo username real da conta Ollama (ex: maracatu-ai, maracatuai ou whereisanzi)
USERNAME=<username>

# 1. Criar modelo local (registra no daemon do Ollama)
cd exports/maracatu-20m-gguf
ollama create "$USERNAME/maracatu-20m" -f Modelfile

# 2. Smoke test: resposta esperada é continuação em PT-BR
ollama run "$USERNAME/maracatu-20m" "O Brasil é"

# 3. Push pro Ollama Hub
ollama push "$USERNAME/maracatu-20m"
```

O comando `ollama create` precisa ser executado de dentro do diretório onde está o GGUF (ou usar path absoluto no `FROM` do Modelfile).

O `ollama push` vai fazer upload do GGUF pra infra do Ollama, pode demorar alguns minutos na primeira vez. Progresso visível no terminal.

Após o push, o modelo fica disponível em `https://ollama.com/<username>/maracatu-20m` e pode ser instalado por qualquer pessoa com:

```bash
ollama run <username>/maracatu-20m
```

**Troubleshooting**

| Erro | Causa | Solução |
|---|---|---|
| `ollama push` retorna 403 | não autenticado ou username errado | `ollama login`; confirmar que o username do push bate com a conta autenticada |
| `model not found` no `ollama create` | path do GGUF errado ou Ollama daemon parado | `ollama serve` em outro terminal; verificar path absoluto no FROM |
| Geração produz tokens repetidos | quantização agressiva + modelo pequeno | testar com Q8_0; se persistir, problema no modelo em si |
| `ollama create` falha com "invalid model file" | Modelfile mal formatado | verificar aspas triplas no TEMPLATE e ausência de tabs no início das linhas |
| Daemon do Ollama não iniciou no macOS | primeiro uso após install | `ollama serve` ou abrir o app Ollama pela primeira vez |

---

### Passo 6: Kaggle Models

#### Contexto importante: o CLI 1.5.16 não tem o subcomando `models`

O `kaggle models` existe apenas na versão 1.6+. Nossa CLI está fixada em `1.5.16` no `.venv` do projeto por causa do bug "hashlink null" que ainda afeta `kernels push` na 1.6+. **Não fazer upgrade global**: isso pode quebrar o pipeline de treino.

Ver discussão completa em [`docs/kaggle.md`](kaggle.md#pin-da-versão-do-cli).

#### Opção A: UI (sempre confiável)

1. Acessar https://www.kaggle.com/models/create
2. Preencher:
   - Owner: `whereisanzi` (sua conta pessoal; Kaggle Models não suporta orgs na criação, só em transferência posterior)
   - Title: `Maracatu 20M`
   - Slug: `maracatu-20m`
   - Task: Text Generation
   - Framework: Other
3. Criar instância: variante `v1`, framework `Other`
4. Upload dos artefatos via drag-and-drop:
   - Tudo de `exports/maracatu-20m-hf/` (safetensors + tokenizer)
   - `exports/maracatu-20m-gguf/maracatu-20m-Q4_K_M.gguf`

O modelo fica em `https://www.kaggle.com/models/whereisanzi/maracatu-20m` e pode ser referenciado em notebooks Kaggle.

#### Opção B: CLI em venv isolado (evita contaminar o .venv do projeto)

```bash
python3 -m venv /tmp/kaggle-new-venv
/tmp/kaggle-new-venv/bin/pip install "kaggle>=1.6" --quiet

# Criar o model (namespace no Kaggle)
/tmp/kaggle-new-venv/bin/kaggle models create \
    --owner whereisanzi \
    --name maracatu-20m \
    --title "Maracatu 20M" \
    --license Apache-2.0

# Criar instância com os artefatos
/tmp/kaggle-new-venv/bin/kaggle models instances create \
    --owner whereisanzi \
    --name maracatu-20m \
    --framework other \
    --version maracatu-20m-v1 \
    --source-files exports/maracatu-20m-hf
```

Se aparecer "slugs and hashlink are all null", cair para a Opção A. O bug é intermitente e depende do estado da conta.

**Troubleshooting**

| Erro | Causa | Solução |
|---|---|---|
| `kaggle: command 'models' not found` | CLI 1.5.16 | usar Opção B com venv isolado ou Opção A |
| `slugs and hashlink are all null` | bug do CLI 1.6+ em contas recém-criadas | usar Opção A |
| `403 Forbidden` no upload via UI | conta sem phone verification | verificar em https://www.kaggle.com/settings |
| Modelo aparece mas sem GPU no notebook | framework `Other` sem preset de device | adicionar instrução de uso no description: "carregar via transformers ou llama.cpp" |

---

## 4. Naming convention

Aplicar consistentemente em todos os canais, arquivos e comandos:

| Campo | Formato | Exemplos |
|---|---|---|
| Repo HF Hub | `maracatu-ai/maracatu-<scale>` | `maracatu-ai/maracatu-20m`, `maracatu-ai/maracatu-500m` |
| Modelo Ollama Hub (M-20M → M-800M) | `whereisanzi/maracatu-<scale>` | `whereisanzi/maracatu-20m` |
| Modelo Ollama Hub (M-8B+) | `maracatuai/maracatu-<scale>` | `maracatuai/maracatu-7b` (migração no M-8B) |
| Modelo Kaggle Models | `whereisanzi/maracatu-<scale>` | `whereisanzi/maracatu-20m` |
| Diretório de export HF | `exports/maracatu-<scale>-hf` | `exports/maracatu-20m-hf` |
| Diretório de export GGUF | `exports/maracatu-<scale>-gguf` | `exports/maracatu-20m-gguf` |
| Arquivos GGUF | `maracatu-<scale>-<quant>.gguf` | `maracatu-20m-Q4_K_M.gguf` |
| Subpasta GGUF no HF | `gguf/` | `gguf/maracatu-20m-Q4_K_M.gguf` |
| Tag semver git | `v<N>.<M>.<P>` | `v0.1.0` (M-20M), `v0.2.0` (M-80M), `v0.3.0` (M-800M) |
| Config YAML | `configs/maracatu_<scale>.yaml` | `configs/maracatu_20m.yaml` |

Regras:
- Sempre minúsculo. Nunca `Maracatu-20M` como slug/nome de repo.
- Sempre hífen como separador. Nunca underscore em slugs de plataforma.
- A escala usa a unidade implícita mais legível: `20m` (não `20M` nem `20000000`).
- O `<scale>` no nome do arquivo GGUF bate com o nome do repo para evitar ambiguidade ao distribuir arquivos isolados.

---

## 5. Troubleshooting consolidado por passo

Esta seção é um índice rápido. Cada passo acima tem sua tabela de troubleshooting detalhada.

| Sintoma | Passo | Diagnóstico rápido |
|---|---|---|
| `max_abs_diff > 1e-3` no export | 1 | divergência de implementação: não publicar |
| `huggingface-cli: 401` | 2 | token expirado ou escopo errado |
| `cmake: not found` | 3 | `brew install cmake` |
| GGUF corrompido / `magic mismatch` | 3 | verificar SHA-256 do fp16 antes de quantizar |
| `llama-cli` trava sem output | 3 | tentar sem `--log-disable`; verificar se o modelo carregou |
| `ollama push: 403` | 5 | `ollama login`; confirmar que o username do push bate com a conta autenticada (Ollama não tem org) |
| `ollama create: no such file` | 5 | executar de dentro do dir do GGUF ou usar path absoluto no `FROM` |
| `kaggle models: unknown command` | 6 | CLI 1.5.16: usar venv isolado ou UI |
| Modelo Kaggle sem GPU | 6 | documentar requisito de device no description do model |

---

## 6. Release tag e release notes

### Tag semver no GitHub

```bash
git tag -a v0.1.0 -m "Maracatu-20M: primeiro release público (Apache 2.0)"
git push origin v0.1.0
```

Convenção de versioning:
- `v0.1.0`: M-20M, primeiro release público
- `v0.2.0`: M-80M
- `v0.3.0`: M-800M
- `v0.x.y`: iterações no mesmo scale (corpus expandido, fine-tuning, etc.)
- `v1.0.0`: M-8B ou primeiro modelo que bate benchmark relevante
- `v2.0.0`: M-80B (North Star)

### Release notes (postar no HF Hub como commit description ou Discussion)

Conteúdo mínimo obrigatório:
- Escala e arquitetura (número de parâmetros, camadas, heads, vocabulário)
- Configuração de treino: `max_iters`, `learning_rate`, `batch_size`, `device`
- Tokens consumidos (extrair do log do Kaggle: `step * batch_size * seq_len`)
- Benchmarks: ao menos perplexidade no holdout de validação; se disponível, perplexidade num corpus externo PT-BR
- Limitações conhecidas: contexto curto (512 tokens no M-20M), tendência a repetição em temperaturas baixas (comum em modelos pequenos), sem RLHF
- Hash do commit git (`git_revision` já armazenado no checkpoint: `ckpt["git_revision"]`)

---

## 7. Checklist de release

- [ ] `export_hf.py` rodou com `max_abs_diff=0.0` (ou < 1e-3 no mínimo)
- [ ] `model.generate()` produziu texto PT-BR sintaticamente válido no sanity check
- [ ] `tokenizer.model` presente em `exports/maracatu-<scale>-hf/`
- [ ] `tokenizer_config.json` e `special_tokens_map.json` presentes
- [ ] Upload HF Hub concluído; URL do repo anotada
- [ ] `pipeline_tag` e `library_name` configurados no HF Hub
- [ ] README.md no HF repo atualizado (baseado em `MODEL_CARD.md`)
- [ ] GGUF fp16 gerado e SHA-256 registrado
- [ ] Q4_K_M, Q5_K_M e Q8_0 gerados e smoke testados via `llama-cli`
- [ ] SHA-256 de cada GGUF registrado nas release notes
- [ ] GGUFs uploaded para `gguf/` no HF Hub
- [ ] Modelfile criado com stop tokens, temperature, top_k, num_ctx corretos
- [ ] `ollama create` concluiu sem erro
- [ ] `ollama run <username>/maracatu-<scale> "O Brasil é"` produziu output (substituir `<username>` pelo username real da conta Ollama)
- [ ] `ollama push` concluiu; URL do modelo no Ollama Hub anotada (`https://ollama.com/<username>/maracatu-<scale>`)
- [ ] Kaggle Models criado (UI ou CLI isolado); safetensors + Q4_K_M uploaded
- [ ] Git tag `v<N>.<M>.<P>` criada e pushed
- [ ] Release notes no HF: training config, tokens, benchmarks, limitações, `git_revision`

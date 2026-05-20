# Treinar o Maracatu no Kaggle

Guia passo-a-passo para reproduzir o treino do Maracatu em um notebook Kaggle usando GPU T4 free tier. Adequado para modelos de 20M-125M parâmetros (~2-6h de treino).

## Por que Kaggle

- **30h/semana de GPU T4 grátis** (T4 x1, ou P100, ou T4 x2 dependendo do disponível)
- **Sessão de até 9h por run**: suficiente para nossos modelos pequenos
- **Pago gratuito** para datasets e notebooks privados
- **Persistência de outputs**: checkpoints ficam acessíveis por download via CLI

Para modelos maiores (300M+), considerar RunPod/Lambda em GPUs A100 pagas.

## Pré-requisitos

- Conta Kaggle ativa
- Python 3.11+ local
- `.venv` do projeto ativado (via `uv venv && uv pip install -e ".[dev]"`)
- Phone verification: **NÃO é necessária** para datasets/kernels (só para modelos públicos)

## ⚠️ Pin da versão do CLI

**IMPORTANTE:** a partir do `kaggle 1.6.0`, o CLI migrou para endpoints gRPC-style em `api.kaggle.com/v1/*.Service/*` que apresentam bugs conhecidos com contas recém-criadas: datasets sobem mas falha no passo final com "Dataset creation error: slugs and hashlink are all null".

**Solução:** fixar em `kaggle<1.6`. Já está no `pyproject.toml` em `[dev]`, mas se precisar manualmente:

```bash
uv pip install "kaggle==1.5.16"
```

## ⚠️ Auth: três tipos de token diferentes (lição do M-80M)

Kaggle hoje (2026) tem **dois sistemas de credencial coexistindo**:

1. **API Token clássico** (`{"key":"<32 chars>"}`): formato hex/base32, cerca de 32 caracteres. Funciona com CLI (todas as versões) e com chamadas REST diretas (`https://www.kaggle.com/api/v1/...`). É o que o `kaggle.json` baixado em `kaggle.com/settings → "Create New Token"` contém.
2. **Personal Access Token** (PAT, `KGAT_*`): prefixo `KGAT_`, cerca de 37 caracteres. É um token estilo OAuth gerado em outro fluxo da UI. **NÃO funciona com o CLI** (legacy ou nova). Usado por integrações novas (GitHub Actions oficiais, alguns SDKs).
3. **OAuth/JWT** (cookies de sessão): usado pela UI web e por features novas como Kaggle Notebooks integradas. Não exposto como string copiável.

**Sintoma de token errado**: o CLI lê (`kaggle datasets list`) com sucesso, mas qualquer write (`datasets create`, `models create`) retorna `401 Unauthorized` independentemente da versão do CLI ou do endpoint.

**Como diferenciar visualmente**:
- 32 chars, alfanumérico hex-like → API Token clássico → **OK pro CLI**
- 37 chars, começa com `KGAT_` → Personal Access Token → **NÃO funciona com CLI**

### Fluxo correto para autenticar o CLI

1. Acessar `https://www.kaggle.com/settings` logado.
2. Na seção **API**, clicar **"Create New Token"** (NÃO usar o botão de Personal Access Token na seção OAuth).
3. Browser baixa `kaggle.json` para `~/Downloads/kaggle.json`. **Esse arquivo é a fonte da verdade**, não copie o token pra outros lugares.
4. Mover/copiar para `~/.kaggle/kaggle.json` (CLI lê desse path por default) com permissões 600:
   ```bash
   mkdir -p ~/.kaggle
   mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json
   chmod 600 ~/.kaggle/kaggle.json
   ```
5. Verificar:
   ```bash
   kaggle datasets list --user whereisanzi   # deve listar datasets sem erro
   ```

### O que NÃO fazer

- **Não derive** `kaggle.json` a partir de `KAGGLE_API_TOKEN` no `.env`. Esse campo no `.env.example` do projeto induz a salvar um Personal Access Token (formato `KGAT_*`) que **não autentica o CLI**. Mantemos a entrada `KAGGLE_API_TOKEN` no `.env.example` apenas como referência histórica; a fonte real é `~/.kaggle/kaggle.json`.
- **Não confie em readonly tests**: o CLI consegue listar datasets mesmo com token sem permissão de write. O único teste real é tentar `kaggle datasets create` ou `kaggle models create` em um diretório de teste.
- **Não regenere o mesmo tipo de token**: se você gerou um Personal Access Token e o CLI continua falhando, o problema é o tipo do token (não a renovação). Use o botão **"Create New Token"** da seção API.

### Diagnóstico: comparar tokens em uso

Quando algo dá errado, comparar os tokens disponíveis no Mac (sem vazar valores):

```bash
python3 <<'PYEOF'
import json, hashlib, os
from datetime import datetime
paths = [
    os.path.expanduser("~/.kaggle/kaggle.json"),
    ".kaggle/kaggle.json",
    os.path.expanduser("~/Downloads/kaggle.json"),
]
for p in paths:
    if os.path.exists(p):
        d = json.load(open(p))
        k = d.get("key", "")
        ts = datetime.fromtimestamp(os.path.getmtime(p)).strftime("%Y-%m-%d %H:%M:%S")
        print(f"{p}: length={len(k)}, prefix={k[:6]}..., sha256={hashlib.sha256(k.encode()).hexdigest()[:12]}, modified={ts}")
PYEOF
```

Procurar:
- Length 32, prefix alfanum (ex: `bd0282`, `c9828d`) → API Token clássico (OK)
- Length 37, prefix `KGAT_*` → Personal Access Token (NÃO OK pra CLI)

A versão 1.5.x usa endpoints legados em `www.kaggle.com/api/v1/*` que funcionam normalmente.

## Passo 1: Autenticação

### Gerar API key

1. https://www.kaggle.com/settings → seção **"API"**
2. Clique em **"Create Legacy API Key"** (mais confiável que os novos `KGAT_` tokens no CLI 1.5.x)
3. Download do `kaggle.json`
4. Mova para `~/.kaggle/kaggle.json`:
   ```bash
   mkdir -p ~/.kaggle
   mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json
   chmod 600 ~/.kaggle/kaggle.json
   ```

### Teste

```bash
.venv/bin/kaggle competitions list | head -3
```

Se retornar competições, auth está ok. Se der 401, refaça o passo de key generation.

### Dica: cópia local

Opcionalmente, manter uma cópia em `.kaggle/kaggle.json` dentro do projeto (está gitignored). Útil para sincronizar entre máquinas sem baixar de novo.

## Passo 2: Upload do corpus como dataset

O corpus (~2.3 GB) já está no repo em `data/processed/corpus.txt` + tokenizer em `tokenizer/maracatu.model`. O Kaggle não precisa dessa estrutura do repo: precisa de uma pasta com os arquivos + `dataset-metadata.json`.

Usamos **hardlinks** para evitar duplicar 2.2 GB no disco:

```bash
mkdir -p .kaggle/corpus-dataset
cd .kaggle/corpus-dataset

ln ../../data/processed/corpus.txt corpus.txt
ln ../../data/processed/MANIFEST.txt MANIFEST.txt
ln ../../tokenizer/maracatu.model maracatu.model
ln ../../tokenizer/maracatu.vocab maracatu.vocab
```

Metadata (`dataset-metadata.json`):

```json
{
  "title": "Maracatu Corpus v1",
  "id": "<SEU_USERNAME>/maracatu-corpus-v1",
  "licenses": [{"name": "CC-BY-SA-4.0"}]
}
```

Upload:

```bash
.venv/bin/kaggle datasets create -p .kaggle/corpus-dataset --dir-mode zip
```

Primeiro upload leva ~5-10 min no CLI 1.5.x com ~25 MB/s (dependendo da sua conexão).

### Atualizar uma versão existente

```bash
.venv/bin/kaggle datasets version -p .kaggle/corpus-dataset -m "descrição da versão"
```

## Passo 3: Upload do código como dataset separado

O `kaggle_run.py` importa `model.py`, `data.py` e o YAML de config. Esses vão como segundo dataset (pequeno):

```bash
mkdir -p .kaggle/code-dataset
cd .kaggle/code-dataset

cp ../../src/maracatu/model.py .
cp ../../src/maracatu/data.py .
cp ../../src/maracatu/train.py .
cp ../../src/maracatu/sample.py .
cp ../../configs/maracatu_20m.yaml .
```

Metadata:

```json
{
  "title": "Maracatu Code",
  "id": "<SEU_USERNAME>/maracatu-code",
  "licenses": [{"name": "apache-2.0"}]
}
```

Upload:

```bash
.venv/bin/kaggle datasets create -p .kaggle/code-dataset
```

## Passo 4: Criar e disparar o kernel de treino

O script runner `scripts/kaggle_run.py` (versionado no repo) é o entry point do kernel. Ele importa do dataset de código e lê do dataset de corpus.

Staging do kernel:

```bash
mkdir -p .kaggle/kernel
cp scripts/kaggle_run.py .kaggle/kernel/
```

Metadata (`.kaggle/kernel/kernel-metadata.json`):

```json
{
  "id": "<SEU_USERNAME>/maracatu-20m-training",
  "title": "Maracatu 20M Training",
  "code_file": "kaggle_run.py",
  "language": "python",
  "kernel_type": "script",
  "is_private": true,
  "enable_gpu": true,
  "enable_tpu": false,
  "enable_internet": true,
  "dataset_sources": [
    "<SEU_USERNAME>/maracatu-corpus-v1",
    "<SEU_USERNAME>/maracatu-code"
  ],
  "competition_sources": [],
  "kernel_sources": [],
  "model_sources": []
}
```

Push (cria e dispara o run automaticamente):

```bash
.venv/bin/kaggle kernels push -p .kaggle/kernel
```

A saída imprime o link do kernel: `https://www.kaggle.com/code/<user>/maracatu-20m-training`.

## Passo 5: Monitorar o treino

### Status

```bash
.venv/bin/kaggle kernels status <user>/maracatu-20m-training
```

Estados possíveis: `queued`, `running`, `complete`, `error`, `cancelAcknowledged`.

### Logs em tempo real

O script usa `print(..., flush=True)` para garantir log unbuffered. No Kaggle, ver logs em tempo real só é possível abrindo o notebook no browser na aba "Log".

### Saída

Quando completar, download dos artefatos:

```bash
.venv/bin/kaggle kernels output <user>/maracatu-20m-training -p checkpoints/kaggle/
```

Isso baixa `tokens.npy`, `best.pt`, `latest.pt`, `final.pt` do `/kaggle/working/` pro seu disco local.

## Passo 6: Export para HuggingFace

Depois que tiver o checkpoint local, rode nosso script de export (já versionado):

```bash
.venv/bin/python scripts/export_hf.py \
    --checkpoint checkpoints/kaggle/best.pt \
    --tokenizer tokenizer/maracatu.model \
    --output-dir exports/maracatu-20m-hf
```

O script:
- Converte nosso `state_dict` para `LlamaForCausalLM` do HF
- Valida equivalência numérica (bit-a-bit)
- Salva em `safetensors`
- Prepara tokenizer SentencePiece como `LlamaTokenizer`
- Sanity check com `AutoModel.from_pretrained`

Publicação:

```bash
.venv/bin/huggingface-cli upload maracatu-ai/maracatu-20m exports/maracatu-20m-hf .
```

## Estrutura final do `.kaggle/`

```
.kaggle/                        (gitignored: contém credentials + staging)
├── kaggle.json                 (~/.kaggle/kaggle.json é o que o CLI lê, este é só backup)
├── corpus-dataset/
│   ├── corpus.txt              (hardlink)
│   ├── maracatu.model          (hardlink)
│   ├── maracatu.vocab          (hardlink)
│   ├── MANIFEST.txt            (hardlink)
│   └── dataset-metadata.json
├── code-dataset/
│   ├── model.py, data.py, train.py, sample.py
│   ├── maracatu_20m.yaml
│   └── dataset-metadata.json
└── kernel/
    ├── kaggle_run.py           (cópia de scripts/kaggle_run.py)
    └── kernel-metadata.json
```

## Troubleshooting

### 401/403 em endpoints que deveriam funcionar

Provavelmente CLI 1.6+ ou combinação de env vars stale. Solução:

```bash
# Remover env vars stale da sessão shell
unset KAGGLE_USERNAME KAGGLE_API_TOKEN KAGGLE_KEY

# Verificar config
.venv/bin/kaggle config view
# Deve mostrar: auth_method=LEGACY_API_KEY, username=<seu>

# Se auth_method estiver ACCESS_TOKEN, forçar legacy via kaggle.json válido
```

### "Dataset creation error: slugs and hashlink are all null"

CLI 1.6+. Downgrade:

```bash
uv pip install "kaggle==1.5.16"
```

### "Invalid Owner Id"

Metadata com `id: del=<hash>/...`: o `del=` é prefixo de conta deletada. Use `<username>/<slug>` direto.

### Kernel trava em "queued" por muito tempo

Cota de GPU da semana esgotada. Veja em https://www.kaggle.com/me/account quota restante.

### Kernel roda mas `Device: cpu` (não GPU)

Duas causas possíveis, em ordem:

1. **Phone + Identity verification não feitas** na conta Kaggle. Obrigatórias pra quota de GPU, mesmo em free tier. Faça em https://www.kaggle.com/settings.
2. **Metadata com `"enable_gpu": "true"` (string entre aspas)** é silenciosamente ignorado. Precisa ser boolean JSON `true` sem aspas. Mesmo erro aplica a `is_private`, `enable_tpu`, `enable_internet`.

### Kernel rodou com P100 mas PyTorch não suporta

Kaggle aloca P100 por padrão (capability sm_60) quando apenas `enable_gpu: true` é especificado. O PyTorch atual (>= 2.5) não suporta sm_60. Erros típicos:

```
Tesla P100-PCIE-16GB with CUDA capability sm_60 is not compatible
with the current PyTorch installation.
```

**A solução é mudar o accelerator pra T4 (sm_75, suportado).**

O campo `accelerator: NvidiaTeslaT4` no kernel-metadata.json é aceito pelo CLI mas **ignorado pelo backend**: P100 continua sendo alocado. Mesmo com CLI 1.6+ que suporta a flag.

**O único jeito confiável de pedir T4:** pelo UI:

1. Abra o kernel em `kaggle.com/code/<user>/<slug>`
2. Clica **"Edit"**
3. Painel direito → **"Accelerator"** → muda de "GPU P100" para **"GPU T4 x2"** (ou T4 single)
4. **"Save Version"** → **"Save & Run All (Commit)"**

A preferência **persiste**: pushes subsequentes via CLI continuarão usando T4.

### Sobrescrever um run em andamento

`kaggle kernels push` com o mesmo `id` cria uma **nova versão** (version 2, 3…) que automaticamente dispara. O run anterior não é interrompido, fica terminando em paralelo, competindo pela cota. Se quiser matar o anterior, vá no UI: notebook page → aba "Settings" → "Cancel run".

### Script falha com `ModuleNotFoundError`

Verifique se os datasets `maracatu-code` e `maracatu-corpus-v1` estão listados em `dataset_sources` no `kernel-metadata.json`, e se o `kaggle_run.py` adiciona `/kaggle/input/maracatu-code` ao `sys.path`.

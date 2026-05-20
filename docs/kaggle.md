# Train Maracatu on Kaggle

Step-by-step guide to reproduce Maracatu training in a Kaggle notebook using a free-tier T4 GPU. Suitable for models from 20M-125M parameters (~2-6h of training).

## Why Kaggle

- **30h/week of free T4 GPU** (T4 x1, or P100, or T4 x2 depending on availability)
- **Up to 9h per run**: enough for our small models
- **Free** for datasets and private notebooks
- **Output persistence**: checkpoints stay accessible via CLI download

For larger models (300M+), consider RunPod/Lambda on paid A100 GPUs.

## Prerequisites

- Active Kaggle account
- Python 3.11+ locally
- Project `.venv` activated (via `uv venv && uv pip install -e ".[dev]"`)
- Phone verification: **not required** for datasets/kernels (only for public models)

## ⚠️ Pin the CLI version

**IMPORTANT:** starting with `kaggle 1.6.0`, the CLI migrated to gRPC-style endpoints at `api.kaggle.com/v1/*.Service/*` that have known bugs with newly created accounts: datasets upload but fail at the final step with "Dataset creation error: slugs and hashlink are all null".

**Solution:** pin to `kaggle<1.6`. Already set in `pyproject.toml` under `[dev]`, but if you need to do it manually:

```bash
uv pip install "kaggle==1.5.16"
```

## ⚠️ Auth: three different token types (lesson from M-80M)

Kaggle today (2026) has **two credential systems coexisting**:

1. **Classic API Token** (`{"key":"<32 chars>"}`): hex/base32 format, around 32 characters. Works with the CLI (all versions) and with direct REST calls (`https://www.kaggle.com/api/v1/...`). This is what the `kaggle.json` downloaded from `kaggle.com/settings → "Create New Token"` contains.
2. **Personal Access Token** (PAT, `KGAT_*`): prefix `KGAT_`, around 37 characters. An OAuth-style token generated through a different UI flow. **Does NOT work with the CLI** (legacy or new). Used by newer integrations (official GitHub Actions, some SDKs).
3. **OAuth/JWT** (session cookies): used by the web UI and by newer features like integrated Kaggle Notebooks. Not exposed as a copyable string.

**Symptom of the wrong token**: the CLI reads (`kaggle datasets list`) successfully, but any write (`datasets create`, `models create`) returns `401 Unauthorized` regardless of CLI version or endpoint.

**How to tell them apart visually**:
- 32 chars, hex-like alphanumeric → Classic API Token → **OK for the CLI**
- 37 chars, starts with `KGAT_` → Personal Access Token → **Does NOT work with the CLI**

### Correct flow to authenticate the CLI

1. Go to `https://www.kaggle.com/settings` while logged in.
2. In the **API** section, click **"Create New Token"** (do NOT use the Personal Access Token button in the OAuth section).
3. The browser downloads `kaggle.json` to `~/Downloads/kaggle.json`. **This file is the source of truth**, don't copy the token elsewhere.
4. Move/copy it to `~/.kaggle/kaggle.json` (the CLI reads from this path by default) with permissions 600:
   ```bash
   mkdir -p ~/.kaggle
   mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json
   chmod 600 ~/.kaggle/kaggle.json
   ```
5. Verify:
   ```bash
   kaggle datasets list --user whereisanzi   # should list datasets without error
   ```

### What NOT to do

- **Do not derive** `kaggle.json` from `KAGGLE_API_TOKEN` in `.env`. That field in the project's `.env.example` leads to saving a Personal Access Token (`KGAT_*` format) that **does not authenticate the CLI**. We keep the `KAGGLE_API_TOKEN` entry in `.env.example` only as a historical reference; the real source is `~/.kaggle/kaggle.json`.
- **Do not trust readonly tests**: the CLI can list datasets even with a token that has no write permission. The only real test is to try `kaggle datasets create` or `kaggle models create` in a test directory.
- **Do not regenerate the same token type**: if you generated a Personal Access Token and the CLI still fails, the problem is the token type (not the renewal). Use the **"Create New Token"** button in the API section.

### Diagnostics: compare tokens in use

When something goes wrong, compare the tokens available on the Mac (without leaking values):

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

Look for:
- Length 32, alphanumeric prefix (e.g., `bd0282`, `c9828d`) → Classic API Token (OK)
- Length 37, prefix `KGAT_*` → Personal Access Token (NOT OK for the CLI)

Version 1.5.x uses legacy endpoints at `www.kaggle.com/api/v1/*` that work normally.

## Step 1: Authentication

### Generate API key

1. https://www.kaggle.com/settings → **"API"** section
2. Click **"Create Legacy API Key"** (more reliable than the new `KGAT_` tokens with CLI 1.5.x)
3. Download `kaggle.json`
4. Move to `~/.kaggle/kaggle.json`:
   ```bash
   mkdir -p ~/.kaggle
   mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json
   chmod 600 ~/.kaggle/kaggle.json
   ```

### Test

```bash
.venv/bin/kaggle competitions list | head -3
```

If it returns competitions, auth is fine. If you get a 401, redo the key generation step.

### Tip: local copy

Optionally, keep a copy in `.kaggle/kaggle.json` inside the project (it's gitignored). Useful to sync between machines without downloading again.

## Step 2: Upload the corpus as a dataset

The corpus (~2.3 GB) is already in the repo at `data/processed/corpus.txt` + tokenizer at `tokenizer/maracatu.model`. Kaggle does not need that repo structure: it needs a folder with the files + `dataset-metadata.json`.

We use **hardlinks** to avoid duplicating 2.2 GB on disk:

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
  "id": "<YOUR_USERNAME>/maracatu-corpus-v1",
  "licenses": [{"name": "CC-BY-SA-4.0"}]
}
```

Upload:

```bash
.venv/bin/kaggle datasets create -p .kaggle/corpus-dataset --dir-mode zip
```

The first upload takes ~5-10 min on CLI 1.5.x at ~25 MB/s (depending on your connection).

### Update an existing version

```bash
.venv/bin/kaggle datasets version -p .kaggle/corpus-dataset -m "version description"
```

## Step 3: Upload the code as a separate dataset

The `kaggle_run.py` imports `model.py`, `data.py` and the YAML config. These go as a second (small) dataset:

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
  "id": "<YOUR_USERNAME>/maracatu-code",
  "licenses": [{"name": "apache-2.0"}]
}
```

Upload:

```bash
.venv/bin/kaggle datasets create -p .kaggle/code-dataset
```

## Step 4: Create and trigger the training kernel

The runner script `scripts/kaggle_run.py` (versioned in the repo) is the kernel's entry point. It imports from the code dataset and reads from the corpus dataset.

Kernel staging:

```bash
mkdir -p .kaggle/kernel
cp scripts/kaggle_run.py .kaggle/kernel/
```

Metadata (`.kaggle/kernel/kernel-metadata.json`):

```json
{
  "id": "<YOUR_USERNAME>/maracatu-20m-training",
  "title": "Maracatu 20M Training",
  "code_file": "kaggle_run.py",
  "language": "python",
  "kernel_type": "script",
  "is_private": true,
  "enable_gpu": true,
  "enable_tpu": false,
  "enable_internet": true,
  "dataset_sources": [
    "<YOUR_USERNAME>/maracatu-corpus-v1",
    "<YOUR_USERNAME>/maracatu-code"
  ],
  "competition_sources": [],
  "kernel_sources": [],
  "model_sources": []
}
```

Push (creates and triggers the run automatically):

```bash
.venv/bin/kaggle kernels push -p .kaggle/kernel
```

The output prints the kernel link: `https://www.kaggle.com/code/<user>/maracatu-20m-training`.

## Step 5: Monitor training

### Status

```bash
.venv/bin/kaggle kernels status <user>/maracatu-20m-training
```

Possible states: `queued`, `running`, `complete`, `error`, `cancelAcknowledged`.

### Real-time logs

The script uses `print(..., flush=True)` to guarantee unbuffered logs. On Kaggle, viewing logs in real time is only possible by opening the notebook in the browser on the "Log" tab.

### Output

When it completes, download the artifacts:

```bash
.venv/bin/kaggle kernels output <user>/maracatu-20m-training -p checkpoints/kaggle/
```

This downloads `tokens.npy`, `best.pt`, `latest.pt`, `final.pt` from `/kaggle/working/` to your local disk.

## Step 6: Export to Hugging Face

Once you have the checkpoint locally, run our export script (already versioned):

```bash
.venv/bin/python scripts/export_hf.py \
    --checkpoint checkpoints/kaggle/best.pt \
    --tokenizer tokenizer/maracatu.model \
    --output-dir exports/maracatu-20m-hf
```

The script:
- Converts our `state_dict` to HF `LlamaForCausalLM`
- Validates numerical equivalence (bit-for-bit)
- Saves as `safetensors`
- Prepares the SentencePiece tokenizer as `LlamaTokenizer`
- Sanity check with `AutoModel.from_pretrained`

Publishing:

```bash
.venv/bin/huggingface-cli upload maracatu-ai/maracatu-20m exports/maracatu-20m-hf .
```

## Final `.kaggle/` structure

```
.kaggle/                        (gitignored: contains credentials + staging)
├── kaggle.json                 (~/.kaggle/kaggle.json is what the CLI reads, this is just backup)
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
    ├── kaggle_run.py           (copy of scripts/kaggle_run.py)
    └── kernel-metadata.json
```

## Troubleshooting

### 401/403 on endpoints that should work

Most likely CLI 1.6+ or a combination of stale env vars. Solution:

```bash
# Remove stale env vars from the shell session
unset KAGGLE_USERNAME KAGGLE_API_TOKEN KAGGLE_KEY

# Check the config
.venv/bin/kaggle config view
# Should show: auth_method=LEGACY_API_KEY, username=<yours>

# If auth_method is ACCESS_TOKEN, force legacy via a valid kaggle.json
```

### "Dataset creation error: slugs and hashlink are all null"

CLI 1.6+. Downgrade:

```bash
uv pip install "kaggle==1.5.16"
```

### "Invalid Owner Id"

Metadata with `id: del=<hash>/...`: the `del=` is a deleted-account prefix. Use `<username>/<slug>` directly.

### Kernel stays in "queued" for too long

Weekly GPU quota exhausted. Check remaining quota at https://www.kaggle.com/me/account.

### Kernel runs but `Device: cpu` (no GPU)

Two possible causes, in order:

1. **Phone + Identity verification not done** on the Kaggle account. Required for GPU quota, even on the free tier. Set it at https://www.kaggle.com/settings.
2. **Metadata with `"enable_gpu": "true"` (string in quotes)** is silently ignored. It must be JSON boolean `true` without quotes. The same applies to `is_private`, `enable_tpu`, `enable_internet`.

### Kernel ran with P100 but PyTorch does not support it

Kaggle allocates P100 by default (capability sm_60) when only `enable_gpu: true` is specified. Current PyTorch (>= 2.5) does not support sm_60. Typical errors:

```
Tesla P100-PCIE-16GB with CUDA capability sm_60 is not compatible
with the current PyTorch installation.
```

**The fix is to change the accelerator to T4 (sm_75, supported).**

The `accelerator: NvidiaTeslaT4` field in kernel-metadata.json is accepted by the CLI but **ignored by the backend**: P100 keeps getting allocated. Same even with CLI 1.6+ that supports the flag.

**The only reliable way to request T4:** via the UI:

1. Open the kernel at `kaggle.com/code/<user>/<slug>`
2. Click **"Edit"**
3. Right panel → **"Accelerator"** → change "GPU P100" to **"GPU T4 x2"** (or T4 single)
4. **"Save Version"** → **"Save & Run All (Commit)"**

The preference **persists**: subsequent pushes via CLI will keep using T4.

### Overwrite a run in progress

`kaggle kernels push` with the same `id` creates a **new version** (version 2, 3…) that automatically triggers. The previous run is not interrupted; it keeps finishing in parallel, competing for the quota. To kill the previous one, go to the UI: notebook page → "Settings" tab → "Cancel run".

### Script fails with `ModuleNotFoundError`

Check that the `maracatu-code` and `maracatu-corpus-v1` datasets are listed in `dataset_sources` in `kernel-metadata.json`, and that `kaggle_run.py` adds `/kaggle/input/maracatu-code` to `sys.path`.

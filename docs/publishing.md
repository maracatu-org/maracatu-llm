# Maracatu AI: multi-channel publishing guide

**Channels**: Hugging Face Hub (primary) + Ollama Hub (secondary) + Kaggle Models (tertiary)
**Naming convention**: `maracatu-labs/maracatu-<scale>`, lowercase, hyphen, no underscores
**Last revised**: 2026-04-19

---

## 1. Channel hierarchy

### Hugging Face Hub (primary)

The HF Hub is the standard platform of the LLM research and development community. It publishes safetensors (canonical format, safer than pickle), config.json (machine-readable LlamaConfig), tokenizer and metadata. It is the only channel that receives **all** artifacts: original fp32/bf16 weights via safetensors, GGUFs for all quantizations, and the documentation (README/model card).

Reason for being primary: anyone with `pip install transformers` can load the model in two commands. It's the entry point for most researchers and devs. It also acts as centralized storage; we reference the Ollama GGUFs from here.

### Ollama Hub (secondary)

Ollama is the simplest way to run an LLM locally: one command, no Python, no CUDA setup. It reaches a broader audience than HF: devs who want to integrate an LLM into local apps, enthusiasts with no ML experience. For Maracatu, having a model on Ollama Hub increases the adoption surface, especially in Brazil, where few LLMs are available in Brazilian Portuguese.

Reason for being secondary (and not primary): Ollama only serves quantized GGUF (not the original weights). The HF Hub must exist first, because that is where the GGUF that the Modelfile points to or that we upload comes from.

### Kaggle Models (tertiary)

Kaggle Models lets anyone use the model directly inside a Kaggle Notebook, with no external download. This is valuable for the Brazilian research and student audience: the Kaggle free tier has T4 GPUs, and being able to load Maracatu from inside the notebook with one line makes experiments and fine-tuning easier. It is tertiary because the publishing tooling is the most painful of the three (see dedicated section), and immediate reach is smaller.

---

## 2. Prerequisites

### Required tools

**huggingface-cli**: already included in the `transformers` shipped in the `.venv`:

```bash
.venv/bin/huggingface-cli --version
```

**ollama**: install via Homebrew (recommended on macOS):

```bash
brew install ollama
# Check:
ollama --version   # should return something like "ollama version 0.x.y"
```

Alternative without Homebrew (curl):

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

**llama.cpp**: clone and build locally. The build is needed for two binaries: `llama-quantize` and (optionally) `llama-cli` for smoke tests. Requires `cmake >= 3.21` and a C++17 compiler.

```bash
# Check build prerequisites
cmake --version          # must be >= 3.21
clang++ --version        # macOS: clang via Xcode Command Line Tools

# Install cmake if missing
brew install cmake

# Install Xcode Command Line Tools if missing
xcode-select --install
```

Clone and build (do once, path `.cache/llama.cpp/` is in `.gitignore`):

```bash
mkdir -p .cache
git clone https://github.com/ggerganov/llama.cpp .cache/llama.cpp
cd .cache/llama.cpp
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j$(sysctl -n hw.logicalcpu)
# Check that the binaries exist:
ls build/bin/llama-quantize build/bin/llama-cli
```

The `convert_hf_to_gguf.py` from llama.cpp has its own Python dependencies. Install in a separate venv to not contaminate the project `.venv`:

```bash
python3 -m venv .cache/llama-venv
.cache/llama-venv/bin/pip install -r .cache/llama.cpp/requirements.txt
```

### Authentication

**Hugging Face**: do once per machine. Requires a token with `write` scope:

```bash
source .env    # loads HF_TOKEN defined in .env (see .env.example)
.venv/bin/huggingface-cli login --token "$HF_TOKEN"
```

To check that you're authenticated:

```bash
.venv/bin/huggingface-cli whoami
# should return your username and the maracatu-labs org in the listed orgs
```

**Ollama Hub**: do once per machine:

```bash
ollama login
# Opens the browser to authenticate at ollama.com
# Requires an account at https://ollama.com; that account's username becomes the model namespace
# Ollama has no org concept: the namespace is always personal (e.g., maracatuai, whereisanzi)
```

**Kaggle**: already configured in `~/.kaggle/kaggle.json` (see [`docs/kaggle.md`](kaggle.md)).

### Namespaces: what must be created manually before the first upload

| Channel | Namespace | Creation |
|---|---|---|
| HF Hub | `maracatu-labs/maracatu-<scale>` | automatic via `huggingface-cli upload` |
| Ollama Hub (M-20M → M-800M) | `whereisanzi/maracatu-<scale>` | personal account already exists; created automatically by `ollama push` |
| Ollama Hub (M-8B+) | `maracatuai/maracatu-<scale>` | dedicated username to be reserved now at ollama.com (Ollama does not allow `.` in the username; `maracatuai` is the valid form) |
| Kaggle Models | `whereisanzi/maracatu-<scale>` | **mandatory to create via UI before the first CLI upload**: see Step 6 |

> **Ollama namespace policy by scale**:
> - **M-20M → M-1B** (smoke test + first useful models): publish under `whereisanzi/maracatu-<scale>` (Anderson's existing personal account). Acceptable because we don't yet have public traction that justifies reserving premium branding.
> - **M-8B onward** (first serious model, pitch for grants): migrate to dedicated username `maracatuai` (Ollama doesn't accept `.` nor `-` in all cases; `maracatuai` is the safe form). **Reserve that username now at https://ollama.com**, even before using it, to guarantee availability.
> - Older models stay in the original namespace; new releases use the new one. Don't try to "move" a model between namespaces on Ollama Hub: just republish with the new namespace and mark the old one as deprecated in the model card.
>
> **Namespace asymmetry across channels** is expected: HF uses the `maracatu-labs` org; Ollama and Kaggle use a personal username (neither has an org concept). Document this explicitly in MODEL_CARD.md to avoid confusion for users.

---

## 3. Complete sequential pipeline

```
checkpoint.pt
    └─► [Step 1] export_hf.py
            └─► exports/maracatu-<scale>-hf/
                    ├── model.safetensors
                    ├── config.json
                    ├── tokenizer.model
                    ├── tokenizer_config.json
                    └── special_tokens_map.json
                        │
                        ├─► [Step 2] huggingface-cli upload ──────────► HF Hub (safetensors)
                        │
                        └─► [Step 3] convert_hf_to_gguf.py
                                └─► exports/maracatu-<scale>-gguf/
                                        ├── maracatu-<scale>-fp16.gguf
                                        ├── maracatu-<scale>-Q4_K_M.gguf
                                        ├── maracatu-<scale>-Q5_K_M.gguf
                                        └── maracatu-<scale>-Q8_0.gguf
                                                │
                                                ├─► [Step 4] huggingface-cli upload ─► HF Hub (gguf/)
                                                │
                                                ├─► [Step 5] Modelfile + ollama create + ollama push ─► Ollama Hub
                                                │
                                                └─► [Step 6] Kaggle Models upload (UI or isolated CLI)
```

---

### Step 1: HF export (checkpoint.pt → safetensors)

Reference: [`scripts/export_hf.py`](../scripts/export_hf.py)

```bash
.venv/bin/python scripts/export_hf.py \
    --checkpoint checkpoints/kaggle/best.pt \
    --tokenizer tokenizer/maracatu.model \
    --output-dir exports/maracatu-20m-hf
```

The script validates numerical equivalence of logits between our PyTorch implementation and HF's `LlamaForCausalLM`. The expected output includes `max_abs_diff: 0.00e+00` (or < 1e-3 in the worst case). Do not proceed if validation fails.

Generated artifacts:

```
exports/maracatu-20m-hf/
├── model.safetensors          # full weights (bf16/fp32 depending on the checkpoint)
├── config.json                # LlamaConfig: architecture parameters
├── generation_config.json     # generated by save_pretrained
├── tokenizer.model            # SentencePiece BPE 16k
├── tokenizer_config.json      # points to LlamaTokenizer slow
└── special_tokens_map.json    # bos/eos/unk/pad tokens
```

**Note on `tokenizer.json` (Fast vs Slow tokenizer):** `export_hf.py` writes `"tokenizer_class": "LlamaTokenizer"`, which loads the slow version (pure Python via SentencePiece). The Fast version requires a `tokenizer.json` in the HF Tokenizers format, which would need additional conversion via `convert_slow_tokenizer.py` or manual export. For M-20M and M-80M the slow version is enough: `AutoTokenizer.from_pretrained()` works correctly and the speed difference only matters in high-throughput serving (M-8B+). When we reach that scale, add the Fast `tokenizer.json`.

**Troubleshooting**

| Error | Cause | Solution |
|---|---|---|
| `max_abs_diff > 1e-3` | RoPE convention or head order diverged | do not publish; compare implementations |
| `RuntimeError: real_missing keys` | state_dict with `_orig_mod.` prefix not removed | checkpoint from `torch.compile`; the script already does `removeprefix`, check that `load_state_dict` was called with the processed dict |
| `AutoTokenizer` fails with `None` | `tokenizer_config.json` missing | check whether `write_tokenizer_files()` ran; check `output-dir` |
| `torch.load` with a security warning | `weights_only=False` on PyTorch >= 2.6 | expected; the script already uses that flag, not an error |

---

### Step 2: HF Hub upload (safetensors)

```bash
.venv/bin/huggingface-cli upload maracatu-labs/maracatu-20m exports/maracatu-20m-hf .
```

This command creates the repo if it doesn't exist and uploads all files from the directory. For larger models (>2GB per file), add `--max-shard-size 2GB` in the `save_pretrained` of `export_hf.py` before running Step 1: this shards the safetensors automatically. The `huggingface-cli` already uses Git LFS for large files when needed; no manual configuration required.

After the upload, configure via the HF Hub UI (at `huggingface.co/maracatu-labs/maracatu-20m/settings`):
- `pipeline_tag: text-generation`
- `library_name: transformers`
- Add `README.md` based on `MODEL_CARD.md` from the repo

**Troubleshooting**

| Error | Cause | Solution |
|---|---|---|
| `401 Unauthorized` | token without write scope or expired | `huggingface-cli login` with the correct token |
| `Repository not found` | `maracatu-labs` org without permission in the token | check that the token belongs to the account that owns the org |
| Upload hangs on a large file | network timeout | `--chunk-size 50000000` on the upload |
| `OSError: git-lfs not found` | git-lfs not installed (required for >5GB) | `brew install git-lfs && git lfs install` |

---

### Step 3: GGUF conversion and quantization

**Why GGUF?** GGUF is the `llama.cpp` binary format: compact, self-describing (carries embedded architecture metadata) and supported by virtually every local inference runtime beyond llama.cpp itself: Ollama, LM Studio, Jan, GPT4All. The conversion starts from the HF safetensors (not the checkpoint.pt) to guarantee that the weights reaching the GGUF are exactly the ones validated in Step 1.

**Step 3a: Convert HF to GGUF fp16 (lossless baseline):**

```bash
.cache/llama-venv/bin/python .cache/llama.cpp/convert_hf_to_gguf.py \
    exports/maracatu-20m-hf \
    --outtype f16 \
    --outfile exports/maracatu-20m-gguf/maracatu-20m-fp16.gguf
```

**Step 3b: Quantize (from the fp16):**

```bash
mkdir -p exports/maracatu-20m-gguf

QUANTIZE=.cache/llama.cpp/build/bin/llama-quantize
FP16=exports/maracatu-20m-gguf/maracatu-20m-fp16.gguf
OUT=exports/maracatu-20m-gguf

$QUANTIZE "$FP16" "$OUT/maracatu-20m-Q4_K_M.gguf" Q4_K_M
$QUANTIZE "$FP16" "$OUT/maracatu-20m-Q5_K_M.gguf" Q5_K_M
$QUANTIZE "$FP16" "$OUT/maracatu-20m-Q8_0.gguf"   Q8_0
```

**Step 3c: Generate checksums and verify sizes:**

```bash
for f in exports/maracatu-20m-gguf/*.gguf; do
    echo "$(shasum -a 256 "$f") | $(du -sh "$f" | cut -f1)"
done
```

Record the SHA-256s in the release notes.

**Step 3d: Smoke test each quantization:**

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

**Expected sizes and quality impact:**

| Quantization | ~20M | ~800M | ~8B | Quality vs fp16 |
|---|---|---|---|---|
| fp16 (reference) | ~34MB | ~1GB | ~14GB | baseline |
| Q8_0 | ~18MB | ~540MB | ~7.5GB | < 0.1% perplexity loss |
| Q5_K_M | ~13MB | ~380MB | ~4.5GB | ~0.5% loss, good quality |
| Q4_K_M | ~10MB | ~290MB | ~3GB | ~1-2% loss, **recommended default** |

Note: for small models like the 20M, the absolute size difference between quantizations is just a few MB. Relative impact on quality tends to be larger than on 7B+ models: consider also publishing Q8_0 as a "no-compromise" option.

**Troubleshooting**

| Error | Cause | Solution |
|---|---|---|
| `cmake: command not found` | cmake not installed | `brew install cmake` |
| Build fails on C++17 | old compiler | `xcode-select --install` |
| `unsupported model architecture` in convert_hf_to_gguf | llama.cpp out of date | `git pull` in `.cache/llama.cpp/` and rebuild |
| `GGUF load error: magic mismatch` | corrupted file | check SHA-256 of fp16 before quantizing |
| `convert_hf_to_gguf.py: model type not supported` | script version older than Llama support | confirm llama.cpp is at version >= b3000 (2024-10+) |

---

### Step 4: Upload GGUF to HF Hub (same repo)

The GGUFs go in the same repo as the safetensors, in the `gguf/` subfolder. This keeps everything centralized and helps discovery: anyone who finds the repo on the HF Hub sees both formats in one place.

```bash
for quant in Q4_K_M Q5_K_M Q8_0; do
    .venv/bin/huggingface-cli upload \
        maracatu-labs/maracatu-20m \
        "exports/maracatu-20m-gguf/maracatu-20m-${quant}.gguf" \
        "gguf/maracatu-20m-${quant}.gguf"
done
```

Don't upload the `fp16.gguf`: it's an intermediate pipeline artifact, already covered by the safetensors.

---

### Step 5: Ollama

#### 5a. Modelfile for a base (completion) model

Maracatu is a **base/completion** model, not instruct. Ollama was designed for chat models, but supports base models via a minimal `TEMPLATE` that passes the prompt straight through with no turn formatting (no `[INST]`, no `<|user|>`, no role markers).

Create the file `exports/maracatu-20m-gguf/Modelfile`:

```dockerfile
FROM ./maracatu-20m-Q4_K_M.gguf

# Base/completion model: no chat template
# Input is passed straight to the model without turn formatting.
TEMPLATE """{{ .Prompt }}"""

# Stop tokens from our 16k SentencePiece tokenizer
PARAMETER stop "</s>"
PARAMETER stop "<unk>"

# Default generation parameters
PARAMETER temperature 0.8
PARAMETER top_k 50
PARAMETER top_p 0.95
PARAMETER num_ctx 512
```

Notes on the fields:
- `num_ctx 512`: max context defined by M-20M's `max_position_embeddings`. For larger scales (800M, 8B), adjust per the config.
- `temperature 0.8` + `top_k 50` + `top_p 0.95`: reasonable defaults for creative completion. The user can override at runtime with `ollama run ... --temperature`.
- `stop "</s>"`: EOS token from our tokenizer (`eos_token_id=3` in LlamaConfig). Without this the model may keep generating indefinitely.
- Ollama may show a "no chat template found" warning: expected for base models, not an error.

#### 5b. Create local model, test and publish

```bash
# Replace <username> with the real Ollama account username (e.g., maracatuai or whereisanzi)
USERNAME=<username>

# 1. Create the local model (registers it in the Ollama daemon)
cd exports/maracatu-20m-gguf
ollama create "$USERNAME/maracatu-20m" -f Modelfile

# 2. Smoke test: expected response is a continuation in Portuguese
ollama run "$USERNAME/maracatu-20m" "O Brasil é"

# 3. Push to Ollama Hub
ollama push "$USERNAME/maracatu-20m"
```

The `ollama create` command must be run from inside the directory where the GGUF is (or use an absolute path in the Modelfile's `FROM`).

`ollama push` will upload the GGUF to Ollama's infrastructure; it can take a few minutes the first time. Progress visible in the terminal.

After the push, the model is available at `https://ollama.com/<username>/maracatu-20m` and can be installed by anyone with:

```bash
ollama run <username>/maracatu-20m
```

**Troubleshooting**

| Error | Cause | Solution |
|---|---|---|
| `ollama push` returns 403 | not authenticated or wrong username | `ollama login`; confirm the push username matches the authenticated account |
| `model not found` on `ollama create` | wrong GGUF path or Ollama daemon stopped | `ollama serve` in another terminal; check the absolute path in FROM |
| Generation produces repeated tokens | aggressive quantization + small model | test with Q8_0; if it persists, problem in the model itself |
| `ollama create` fails with "invalid model file" | malformed Modelfile | check triple quotes in TEMPLATE and absence of tabs at the start of lines |
| Ollama daemon didn't start on macOS | first use after install | `ollama serve` or open the Ollama app for the first time |

---

### Step 6: Kaggle Models

#### Important context: CLI 1.5.16 doesn't have the `models` subcommand

`kaggle models` exists only on version 1.6+. Our CLI is pinned to `1.5.16` in the project `.venv` because of the "hashlink null" bug that still affects `kernels push` on 1.6+. **Do not upgrade globally**: that may break the training pipeline.

See full discussion in [`docs/kaggle.md`](kaggle.md#pin-the-cli-version).

#### Option A: UI (always reliable)

1. Go to https://www.kaggle.com/models/create
2. Fill in:
   - Owner: `whereisanzi` (your personal account; Kaggle Models doesn't support orgs at creation time, only via later transfer)
   - Title: `Maracatu 20M`
   - Slug: `maracatu-20m`
   - Task: Text Generation
   - Framework: Other
3. Create instance: variant `v1`, framework `Other`
4. Upload the artifacts via drag-and-drop:
   - Everything in `exports/maracatu-20m-hf/` (safetensors + tokenizer)
   - `exports/maracatu-20m-gguf/maracatu-20m-Q4_K_M.gguf`

The model lives at `https://www.kaggle.com/models/whereisanzi/maracatu-20m` and can be referenced in Kaggle notebooks.

#### Option B: CLI in an isolated venv (avoids contaminating the project .venv)

```bash
python3 -m venv /tmp/kaggle-new-venv
/tmp/kaggle-new-venv/bin/pip install "kaggle>=1.6" --quiet

# Create the model (namespace on Kaggle)
/tmp/kaggle-new-venv/bin/kaggle models create \
    --owner whereisanzi \
    --name maracatu-20m \
    --title "Maracatu 20M" \
    --license Apache-2.0

# Create instance with the artifacts
/tmp/kaggle-new-venv/bin/kaggle models instances create \
    --owner whereisanzi \
    --name maracatu-20m \
    --framework other \
    --version maracatu-20m-v1 \
    --source-files exports/maracatu-20m-hf
```

If you see "slugs and hashlink are all null", fall back to Option A. The bug is intermittent and depends on the account state.

**Troubleshooting**

| Error | Cause | Solution |
|---|---|---|
| `kaggle: command 'models' not found` | CLI 1.5.16 | use Option B with an isolated venv or Option A |
| `slugs and hashlink are all null` | CLI 1.6+ bug on newly created accounts | use Option A |
| `403 Forbidden` on UI upload | account without phone verification | check at https://www.kaggle.com/settings |
| Model shows up but no GPU in the notebook | framework `Other` with no device preset | add usage instructions in the description: "load via transformers or llama.cpp" |

---

## 4. Naming convention

Apply consistently across all channels, files and commands:

| Field | Format | Examples |
|---|---|---|
| HF Hub repo | `maracatu-labs/maracatu-<scale>` | `maracatu-labs/maracatu-20m`, `maracatu-labs/maracatu-500m` |
| Ollama Hub model (M-20M → M-800M) | `whereisanzi/maracatu-<scale>` | `whereisanzi/maracatu-20m` |
| Ollama Hub model (M-8B+) | `maracatuai/maracatu-<scale>` | `maracatuai/maracatu-7b` (migration at M-8B) |
| Kaggle Models model | `whereisanzi/maracatu-<scale>` | `whereisanzi/maracatu-20m` |
| HF export directory | `exports/maracatu-<scale>-hf` | `exports/maracatu-20m-hf` |
| GGUF export directory | `exports/maracatu-<scale>-gguf` | `exports/maracatu-20m-gguf` |
| GGUF files | `maracatu-<scale>-<quant>.gguf` | `maracatu-20m-Q4_K_M.gguf` |
| GGUF subfolder on HF | `gguf/` | `gguf/maracatu-20m-Q4_K_M.gguf` |
| Git semver tag | `v<N>.<M>.<P>` | `v0.1.0` (M-20M), `v0.2.0` (M-80M), `v0.3.0` (M-800M) |
| YAML config | `configs/maracatu_<scale>.yaml` | `configs/maracatu_20m.yaml` |

Rules:
- Always lowercase. Never `Maracatu-20M` as a slug/repo name.
- Always hyphen as separator. Never underscore in platform slugs.
- Scale uses the most readable implicit unit: `20m` (not `20M` nor `20000000`).
- The `<scale>` in the GGUF filename matches the repo name to avoid ambiguity when distributing isolated files.

---

## 5. Consolidated troubleshooting by step

This section is a quick index. Each step above has its own detailed troubleshooting table.

| Symptom | Step | Quick diagnosis |
|---|---|---|
| `max_abs_diff > 1e-3` on export | 1 | implementation mismatch: do not publish |
| `huggingface-cli: 401` | 2 | expired token or wrong scope |
| `cmake: not found` | 3 | `brew install cmake` |
| Corrupted GGUF / `magic mismatch` | 3 | check SHA-256 of fp16 before quantizing |
| `llama-cli` hangs with no output | 3 | try without `--log-disable`; check that the model loaded |
| `ollama push: 403` | 5 | `ollama login`; confirm the push username matches the authenticated account (Ollama has no org concept) |
| `ollama create: no such file` | 5 | run from inside the GGUF dir or use an absolute path in `FROM` |
| `kaggle models: unknown command` | 6 | CLI 1.5.16: use isolated venv or UI |
| Kaggle model without GPU | 6 | document device requirement in the model description |

---

## 6. Release tag and release notes

### Semver tag on GitHub

```bash
git tag -a v0.1.0 -m "Maracatu-20M: first public release (Apache 2.0)"
git push origin v0.1.0
```

Versioning convention:
- `v0.1.0`: M-20M, first public release
- `v0.2.0`: M-80M
- `v0.3.0`: M-800M
- `v0.x.y`: iterations on the same scale (expanded corpus, fine-tuning, etc.)
- `v1.0.0`: M-8B or first model that hits a relevant benchmark
- `v2.0.0`: M-80B (North Star)

### Release notes (post on the HF Hub as commit description or Discussion)

Required minimum content:
- Scale and architecture (parameter count, layers, heads, vocabulary)
- Training config: `max_iters`, `learning_rate`, `batch_size`, `device`
- Tokens consumed (extract from the Kaggle log: `step * batch_size * seq_len`)
- Benchmarks: at least perplexity on the validation holdout; if available, perplexity on an external PT-BR corpus
- Known limitations: short context (512 tokens on M-20M), tendency to repeat at low temperatures (common in small models), no RLHF
- Git commit hash (`git_revision` already stored in the checkpoint: `ckpt["git_revision"]`)

---

## 7. Release checklist

- [ ] `export_hf.py` ran with `max_abs_diff=0.0` (or < 1e-3 at minimum)
- [ ] `model.generate()` produced syntactically valid PT-BR text in the sanity check
- [ ] `tokenizer.model` present in `exports/maracatu-<scale>-hf/`
- [ ] `tokenizer_config.json` and `special_tokens_map.json` present
- [ ] HF Hub upload completed; repo URL recorded
- [ ] `pipeline_tag` and `library_name` configured on the HF Hub
- [ ] README.md in the HF repo updated (based on `MODEL_CARD.md`)
- [ ] GGUF fp16 generated and SHA-256 recorded
- [ ] Q4_K_M, Q5_K_M and Q8_0 generated and smoke tested via `llama-cli`
- [ ] SHA-256 of each GGUF recorded in the release notes
- [ ] GGUFs uploaded to `gguf/` on the HF Hub
- [ ] Modelfile created with correct stop tokens, temperature, top_k, num_ctx
- [ ] `ollama create` completed without error
- [ ] `ollama run <username>/maracatu-<scale> "O Brasil é"` produced output (replace `<username>` with the real Ollama account username)
- [ ] `ollama push` completed; Ollama Hub model URL recorded (`https://ollama.com/<username>/maracatu-<scale>`)
- [ ] Kaggle Models created (UI or isolated CLI); safetensors + Q4_K_M uploaded
- [ ] Git tag `v<N>.<M>.<P>` created and pushed
- [ ] HF release notes: training config, tokens, benchmarks, limitations, `git_revision`

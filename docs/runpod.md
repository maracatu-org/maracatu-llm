# Train Maracatu on RunPod (A100 Spot)

Step-by-step guide to train Maracatu-80M on a RunPod A100 80GB instance with spot pricing. Covers pod creation, bootstrap, monitoring, preemption recovery and artifact download.

Estimated cost: **$10-13 per complete run** of 100k iters (~11h on A100 80GB at $0.90-1.20/h).

## Prerequisites (do once, before the first run)

### 1. RunPod account

- Sign up at [runpod.io](https://runpod.io) with a personal credit card
- Initial credit loaded (suggestion: $20 to cover 1 run + margin)
- SSH key registered under **Settings > SSH Public Keys** (use your Mac key: `cat ~/.ssh/id_ed25519.pub`)

### 2. Training assets accessible

The three files below are **not in git** (gitignored) and need to be available for the pod to download:

| File | Size | Where it lives locally |
|---|---|---|
| `data/processed/corpus_v2.txt` | 6.2 GB | `data/processed/corpus_v2.txt` |
| `data/processed/tokens.npy` | 3.0 GB | `data/processed/tokens.npy` |
| `tokenizer/maracatu.model` | 519 KB | `tokenizer/maracatu.model` |

**Recommended canonical method: private HF Dataset**

Upload the three files to a private dataset under the `maracatu-labs` org on the HF Hub:

```bash
# On your Mac, inside the repo
huggingface-cli login   # use your HF_TOKEN

# Create the private dataset (once)
huggingface-cli repo create maracatu-corpus-v2 --type dataset --organization maracatu-labs --private

# Upload the assets (keeping directory structure)
huggingface-cli upload maracatu-labs/maracatu-corpus-v2 \
    data/processed/corpus_v2.txt data/processed/corpus_v2.txt \
    --repo-type dataset

huggingface-cli upload maracatu-labs/maracatu-corpus-v2 \
    data/processed/tokens.npy data/processed/tokens.npy \
    --repo-type dataset

huggingface-cli upload maracatu-labs/maracatu-corpus-v2 \
    tokenizer/maracatu.model tokenizer/maracatu.model \
    --repo-type dataset
```

Why a private HF dataset instead of the other options:

- **HF > S3/R2 presigned**: presigned URLs expire (typically 1-7 days). If the URL expires during a preemption and the pod tries to re-download post-restart, the download fails silently. HF Hub with a token doesn't expire.
- **HF > direct SCP from the Mac**: SCP requires the pod to reach your Mac's IP, which doesn't work if the Mac is behind CGNAT (most residential Brazilian ISPs). It also requires the Mac to be on and not suspended, which is impractical for long runs.
- A private HF dataset is accessible by token, from any pod, 24/7, with no URL that expires.

### 3. HF_TOKEN available

The token must have read permission on the private dataset `maracatu-labs/maracatu-corpus-v2` and write permission on the repo `maracatu-labs/maracatu-80m-checkpoints`.

Create it at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) with `write` scope (write includes read).

## Pod creation

1. Go to [runpod.io/console/pods](https://runpod.io/console/pods)
2. Click **Deploy**
3. Required settings:

| Field | Value |
|---|---|
| GPU | **NVIDIA A100 80GB** |
| Cloud type | **Community Cloud** (cheaper) or Secure Cloud (more stable, more expensive) |
| Pricing | **Spot** - click "Spot" near the price |
| Container image | `runpod/pytorch:2.4.0-py3.11-cuda12.1.1-devel-ubuntu22.04` |
| Container disk | 20 GB |
| Volume disk | **80 GB** (mounted at `/workspace`, persists across preemptions on Secure Cloud) |
| Volume mount path | `/workspace` |

4. Under **Environment Variables**, add:

| Key | Value |
|---|---|
| `HF_TOKEN` | your HF token with write permission |

5. Expand **SSH** and confirm your key appears listed
6. Click **Deploy**
7. Wait for the status to change to **Running** (usually 1-3 minutes)

**Important note on preemption and data persistence:**

- **Community Cloud**: the `/workspace` volume may or may not persist across preemptions depending on the physical provider. Consider Community Cloud acceptable for runs where the corpus is already on HF (re-download is fast compared to training).
- **Secure Cloud**: `/workspace` is guaranteed to persist across preemptions of the same template. More expensive (~20-30%), but eliminates corpus re-download.
- The pod can be preempted at any time on spot pricing. train.py saves `latest.pt` every 1000 iters AND every 30 minutes by wallclock, so the maximum loss is 30 minutes of training.

## First run

After the pod is Running, copy the SSH command from the RunPod console ("Connect" button) and run it on your Mac:

```bash
ssh root@<pod-ip> -p <port>
```

Inside the pod, run the bootstrap (replace HF_TOKEN if it isn't in the env):

```bash
export HF_TOKEN=hf_xxxxxxxxxxxx   # if you didn't configure it on the pod as an env var
export CORPUS_METHOD=hf
export HF_DATASET_REPO=maracatu-labs/maracatu-corpus-v2

curl -fsSL https://raw.githubusercontent.com/maracatu-labs/maracatu/main/scripts/runpod_bootstrap.sh \
    | bash
```

Or, if you prefer to clone before running:

```bash
git clone https://github.com/maracatu-labs/maracatu.git /workspace/maracatu
bash /workspace/maracatu/scripts/runpod_bootstrap.sh
```

The bootstrap will:

1. Check GPU and disk space
2. Install system dependencies (git, tmux, wget...)
3. Clone the repo at `/workspace/maracatu`
4. Install uv and create a Python 3.12 venv
5. Install PyTorch cu121 + project deps
6. Log in to HF Hub
7. Download the 3 training assets (~9.2 GB total)
8. Start training in a detached tmux session called `train`

**Estimated time from bootstrap to the first training step: 12-20 minutes** (dominated by asset download: corpus 6.2 GB + tokens 3.0 GB depend on the pod's bandwidth, typically 500 MB/s-1 GB/s on RunPod pods, so ~10-20s per GB = 1-3 min; deps install ~5-8 min).

Once the bootstrap prints the reconnection instructions, you can close the SSH.

## Monitoring

To follow training:

```bash
# Reconnect to the pod
ssh root@<pod-ip> -p <port>

# Follow the log in real time
tmux attach -t train
# (Ctrl+B D to exit without killing the process)

# Or via tail, without needing tmux
tail -f /workspace/maracatu/checkpoints/training.log

# View GPU usage
watch -n 5 nvidia-smi
```

Signs of healthy training in the log:

- `Device: cuda` line and GPU name present at the start
- `loss` dropping from ~9-10 initial to ~3-4 in a few thousand steps
- `tok/s` between 60,000-120,000 on A100 (depends on batch_size and compile)
- `[resume-safe] wallclock checkpoint saved` appearing every ~30 minutes

## Recovery after preemption

The spot pod can be preempted at any time. When that happens:

1. RunPod terminates the pod and, if configured, may try to reallocate it
2. You get an email notification (configure under Settings > Notifications)

To resume:

1. In the RunPod console, check whether the pod was reallocated automatically or create a new pod with the same settings
2. SSH into the new pod
3. Run exactly the same bootstrap command:

```bash
export HF_TOKEN=hf_xxxxxxxxxxxx
export CORPUS_METHOD=hf
export HF_DATASET_REPO=maracatu-labs/maracatu-corpus-v2
bash /workspace/maracatu/scripts/runpod_bootstrap.sh
```

The bootstrap is idempotent:

- If `/workspace/maracatu` already exists, it does `git pull` instead of cloning
- If the venv already exists, it doesn't recreate it
- If the assets already exist on disk, it doesn't re-download
- If `checkpoints/latest.pt` exists, `runpod_train.sh` detects it and uses auto-resume

**Community Cloud post-preemption scenario** (volume didn't persist):

The bootstrap will re-clone the repo and re-download the assets from HF. `latest.pt` won't exist, so training restarts from scratch. To mitigate:

- Use Secure Cloud for long runs
- Or manually upload `latest.pt` to the HF Hub after every long session (see download section below)

## Artifact download

### At the end of training

`runpod_train.sh` automatically uploads `best.pt` and `final.pt` to the HF Hub when it finishes. Verify at:

```
https://huggingface.co/maracatu-labs/maracatu-80m-checkpoints
```

### Manual download from the pod to the Mac

```bash
# On your Mac
scp -P <port> root@<pod-ip>:/workspace/maracatu/checkpoints/best.pt \
    ./checkpoints/runpod/maracatu-80m-best.pt

scp -P <port> root@<pod-ip>:/workspace/maracatu/checkpoints/final.pt \
    ./checkpoints/runpod/maracatu-80m-final.pt
```

### Validate the downloaded checkpoint

```bash
python -c "
import torch
ckpt = torch.load('checkpoints/runpod/maracatu-80m-best.pt', weights_only=False)
print('step:', ckpt['step'])
print('loss:', ckpt['loss'])
print('model_config:', ckpt['model_config'])
print('git_revision:', ckpt['git_revision'])
"
```

### Preventive backup of latest.pt during training

To protect against loss on Community Cloud, you can upload manually at any time (by reconnecting to the pod):

```bash
ssh root@<pod-ip> -p <port>
source /workspace/maracatu/.venv/bin/activate
huggingface-cli upload maracatu-labs/maracatu-80m-checkpoints \
    /workspace/maracatu/checkpoints/latest.pt checkpoints/latest.pt \
    --repo-type model
```

## Cost and alarms

**Expected cost:**

- A100 80GB spot: $0.90-1.20/h (varies with RunPod supply/demand)
- Full 100k iter run: ~10-13h estimated, so $9-16
- Bootstrap (download + install, ~20 min): $0.30-0.40 extra
- **Realistic total: $10-17 per run**

**Alarms:**

| Signal | Likely cause | Action |
|---|---|---|
| `loss: nan` in the first few hundred steps | lr too high or corrupted data | Check tokens.npy; the current config with lr=2.5e-4 was calibrated, do not change it |
| `CUDA out of memory` | batch_size too large | A100 80GB easily handles batch=64-128 with the current config (batch=16); unlikely, but check if something else is using VRAM |
| `GPU: Tesla P100` instead of A100 in the log | Pod allocated with the wrong GPU | Terminate the pod and create it again, explicitly picking A100 |
| Training stops but `latest.pt` hasn't been updated for >30min | Process died before the wallclock save | Resume from the last available latest.pt; theoretical max loss is the first iters before the first save at 1000 iters |
| Disk full | tokens.npy + corpus + checkpoints > volume | Create pod with 80GB+ volume; checkpoints grow ~1.2 GB per save |

## Troubleshooting

**"CUDA not available" in the bootstrap log:**

The pod may have come up with CUDA not yet initialized. Wait 30s and rerun the bootstrap (it's idempotent). If it persists, the pod may have allocated a faulty machine; terminate and create another.

**"No space left on device" during download:**

An 80GB volume should be enough (corpus 6.2 + tokens 3.0 + repo ~0.5 + checkpoints ~5-10 = ~20 GB). If it overflows, something is wrong with the configured volume. Check the volume size in the RunPod console.

**HF download hangs or is very slow:**

Use the `wget` variant directly if you have a presigned URL. Alternatively, `CORPUS_METHOD=presigned` with URLs generated in the HF console (Settings > Access Tokens does not generate URLs; use the Python API with `hf_hub_url()` + `requests`).

**`tmux: command not found`:**

The script installs tmux via apt at the start. If apt failed for some reason, run manually: `apt-get install -y tmux` and re-run the bootstrap.

**train.py imports but fails with `ModuleNotFoundError: maracatu`:**

The project is installed as an editable package (`-e`). If the repo clone failed or is incomplete, `src/maracatu/` doesn't exist. Check: `ls /workspace/maracatu/src/maracatu/`. If empty, delete the repo and let the bootstrap re-clone it.

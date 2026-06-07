#!/usr/bin/env bash

set -euo pipefail

REPO_DIR="/workspace/maracatu"
VENV_DIR="${REPO_DIR}/.venv"
CONFIG="${REPO_DIR}/configs/maracatu_80m.yaml"
CKPT_DIR="${REPO_DIR}/checkpoints"
LOG_FILE="${CKPT_DIR}/training.log"
HF_REPO="maracatu-labs/maracatu-80m-checkpoints"
HF_REPO_TYPE="model"

export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1

[[ -f /root/.bashrc ]] && source /root/.bashrc || true

log() { echo "[train] $(date '+%H:%M:%S') $*"; }

if [[ ! -f "${VENV_DIR}/bin/python" ]]; then
    echo "[ERRO] Venv nao encontrado em ${VENV_DIR}. Rode runpod_bootstrap.sh primeiro." >&2
    exit 1
fi

if [[ ! -f "${CONFIG}" ]]; then
    echo "[ERRO] Config nao encontrada: ${CONFIG}" >&2
    exit 1
fi

mkdir -p "${CKPT_DIR}"

"${VENV_DIR}/bin/python" -c "
import torch
if not torch.cuda.is_available():
    raise RuntimeError('CUDA nao disponivel')
print(f'GPU: {torch.cuda.get_device_name(0)} | VRAM: {torch.cuda.get_device_properties(0).total_memory // 1024**3}GB')
"

RESUME_FLAG="--no-resume"
if [[ -f "${CKPT_DIR}/latest.pt" ]]; then
    log "latest.pt encontrado, treinamento ira auto-resumir."
    RESUME_FLAG=""
else
    log "Nenhum checkpoint previo. Iniciando do zero."
fi

TRAIN_CMD=(
    "${VENV_DIR}/bin/python"
    -u
    -m maracatu.train
    --config "${CONFIG}"
    --device cuda
    --compile
)

if [[ -n "${RESUME_FLAG}" ]]; then
    TRAIN_CMD+=("${RESUME_FLAG}")
fi

log "Comando: ${TRAIN_CMD[*]}"
log "Log: ${LOG_FILE}"
log "Inicio: $(date)"
log "----------------------------------------------"

cd "${REPO_DIR}"

set -o pipefail
"${TRAIN_CMD[@]}" 2>&1 | tee -a "${LOG_FILE}"
TRAIN_EXIT=${PIPESTATUS[0]}

log "----------------------------------------------"
log "Treino finalizado com exit code: ${TRAIN_EXIT}"
log "Fim: $(date)"

if [[ "${TRAIN_EXIT}" -ne 0 ]]; then
    echo "[ERRO] Processo de treino terminou com erro (exit ${TRAIN_EXIT})." >&2
    echo "       Verifique: tail -100 ${LOG_FILE}" >&2
    exit "${TRAIN_EXIT}"
fi

upload_to_hf() {
    if [[ -z "${HF_TOKEN:-}" ]]; then
        log "HF_TOKEN nao definido, pulando upload automatico."
        log "Para upload manual:"
        log "  huggingface-cli upload ${HF_REPO} ${CKPT_DIR}/best.pt checkpoints/best.pt --repo-type ${HF_REPO_TYPE}"
        return
    fi

    log "Iniciando upload pro HF Hub (${HF_REPO})..."

    local files_to_upload=()
    for f in best.pt final.pt; do
        if [[ -f "${CKPT_DIR}/${f}" ]]; then
            files_to_upload+=("${f}")
        fi
    done

    if [[ ${#files_to_upload[@]} -eq 0 ]]; then
        log "Nenhum checkpoint encontrado em ${CKPT_DIR} para upload."
        return
    fi

    "${VENV_DIR}/bin/python" -c "
from huggingface_hub import HfApi
api = HfApi(token='${HF_TOKEN}')
try:
    api.repo_info(repo_id='${HF_REPO}', repo_type='${HF_REPO_TYPE}')
    print('  Repo ja existe: ${HF_REPO}')
except Exception:
    api.create_repo(repo_id='${HF_REPO}', repo_type='${HF_REPO_TYPE}', private=True)
    print('  Repo criado (privado): ${HF_REPO}')
"

    for f in "${files_to_upload[@]}"; do
        log "  Uploading ${f}..."
        "${VENV_DIR}/bin/python" -c "
from huggingface_hub import HfApi
api = HfApi(token='${HF_TOKEN}')
url = api.upload_file(
    path_or_fileobj='${CKPT_DIR}/${f}',
    path_in_repo='checkpoints/${f}',
    repo_id='${HF_REPO}',
    repo_type='${HF_REPO_TYPE}',
)
print(f'  Upload OK: {url}')
"
    done

    log "Upload concluido. Repo: https://huggingface.co/${HF_REPO}"
}

upload_to_hf

log "runpod_train.sh encerrado com sucesso."

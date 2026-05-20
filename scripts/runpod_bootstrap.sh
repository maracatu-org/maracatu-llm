
set -euo pipefail

REPO_URL="https://github.com/maracatu-ai/maracatu.git"
REPO_BRANCH="main"
REPO_DIR="/workspace/maracatu"
PYTHON_VERSION="3.12"
VENV_DIR="${REPO_DIR}/.venv"
CORPUS_METHOD="${CORPUS_METHOD:-hf}"
HF_DATASET_REPO="${HF_DATASET_REPO:-maracatu-ai/maracatu-corpus-v2}"
MIN_DISK_GB=40

log() { echo "[bootstrap] $(date '+%H:%M:%S') $*"; }

check_preconditions() {
    log "Verificando preconditions..."

    if ! command -v nvidia-smi &>/dev/null; then
        echo "[ERRO] nvidia-smi nao encontrado. Confirme que o pod tem GPU alocada." >&2
        exit 1
    fi
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    echo ""

    local available_gb
    available_gb=$(df /workspace --output=avail -BG | tail -1 | tr -d 'G ')
    if [[ "${available_gb}" -lt "${MIN_DISK_GB}" ]]; then
        echo "[ERRO] Apenas ${available_gb}GB livres em /workspace; minimo ${MIN_DISK_GB}GB." >&2
        echo "       Crie o pod com Storage >= 80GB." >&2
        exit 1
    fi
    log "Disco OK: ${available_gb}GB livres."

    if ! command -v python3 &>/dev/null; then
        echo "[ERRO] python3 nao encontrado na imagem." >&2
        exit 1
    fi
    log "Python OK: $(python3 --version)"
}

install_system_deps() {
    log "Atualizando apt e instalando dependencias de sistema..."
    apt-get update -qq
    apt-get install -y -qq git curl tmux htop wget unzip build-essential \
        python3-dev python3-pip python3-venv
}

clone_or_update_repo() {
    if [[ -d "${REPO_DIR}/.git" ]]; then
        log "Repo ja existe em ${REPO_DIR}, fazendo git pull..."
        git -C "${REPO_DIR}" fetch origin "${REPO_BRANCH}"
        git -C "${REPO_DIR}" reset --hard "origin/${REPO_BRANCH}"
    else
        log "Clonando ${REPO_URL} (branch ${REPO_BRANCH})..."
        git clone --branch "${REPO_BRANCH}" --depth 1 \
            "${REPO_URL}" "${REPO_DIR}"
    fi
    log "Repo em: $(git -C ${REPO_DIR} rev-parse --short HEAD)"
}

install_uv() {
    if command -v uv &>/dev/null; then
        log "uv ja instalado: $(uv --version)"
        return
    fi
    log "Instalando uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="${HOME}/.local/bin:${PATH}"
    log "uv instalado: $(uv --version)"
}

create_venv_and_install() {
    if [[ -d "${VENV_DIR}" && -f "${VENV_DIR}/bin/python" ]]; then
        log "Venv ja existe em ${VENV_DIR}, pulando criacao."
    else
        log "Criando venv Python ${PYTHON_VERSION}..."
        uv venv --python "${PYTHON_VERSION}" "${VENV_DIR}"
    fi

    log "Instalando PyTorch cu121..."
    "${VENV_DIR}/bin/pip" install --quiet \
        torch torchvision \
        --index-url https://download.pytorch.org/whl/cu121

    log "Instalando dependencias do projeto..."
    uv pip install --python "${VENV_DIR}/bin/python" \
        --quiet \
        -e "${REPO_DIR}[dev]"

    log "Instalando huggingface-hub CLI..."
    "${VENV_DIR}/bin/pip" install --quiet "huggingface-hub[cli]"

    log "Validando torch+CUDA..."
    "${VENV_DIR}/bin/python" -c "
import torch
assert torch.cuda.is_available(), 'CUDA nao disponivel pelo torch'
print(f'  torch {torch.__version__} | CUDA {torch.version.cuda} | GPU: {torch.cuda.get_device_name(0)}')
"
}

configure_hf_token() {
    if [[ -z "${HF_TOKEN:-}" ]]; then
        log "HF_TOKEN nao definido. Pulando login HF (necessario para download de assets privados)."
        log "Para definir: export HF_TOKEN=hf_xxx e rode o bootstrap novamente."
        return
    fi
    log "Configurando HF token..."
    "${VENV_DIR}/bin/python" -c "
from huggingface_hub import login
login(token='${HF_TOKEN}', add_to_git_credential=False)
print('  HF login OK')
"
    echo "export HF_TOKEN=${HF_TOKEN}" >> /root/.bashrc
}

download_assets_hf() {
    log "Baixando assets via HF dataset (${HF_DATASET_REPO})..."
    mkdir -p "${REPO_DIR}/data/processed" "${REPO_DIR}/tokenizer"

    local targets=(
        "data/processed/corpus_v2.txt"
        "data/processed/tokens.npy"
        "tokenizer/maracatu.model"
    )

    for target in "${targets[@]}"; do
        local dest="${REPO_DIR}/${target}"
        if [[ -f "${dest}" ]]; then
            log "  ja existe: ${target}, pulando."
            continue
        fi
        log "  baixando ${target}..."
        "${VENV_DIR}/bin/python" -c "
from huggingface_hub import hf_hub_download
import shutil, os
src = hf_hub_download(
    repo_id='${HF_DATASET_REPO}',
    filename='${target}',
    repo_type='dataset',
)
os.makedirs(os.path.dirname('${dest}'), exist_ok=True)
shutil.copy(src, '${dest}')
print(f'  salvo em ${dest}')
"
    done
}

download_assets_presigned() {
    log "Baixando assets via URLs presignadas..."
    mkdir -p "${REPO_DIR}/data/processed" "${REPO_DIR}/tokenizer"

    declare -A urls=(
        ["data/processed/corpus_v2.txt"]="${PRESIGNED_CORPUS_URL}"
        ["data/processed/tokens.npy"]="${PRESIGNED_TOKENS_URL}"
        ["tokenizer/maracatu.model"]="${PRESIGNED_TOKENIZER_URL}"
    )

    for target in "${!urls[@]}"; do
        local dest="${REPO_DIR}/${target}"
        local url="${urls[$target]}"
        if [[ -f "${dest}" ]]; then
            log "  ja existe: ${target}, pulando."
            continue
        fi
        if [[ -z "${url}" ]]; then
            echo "[ERRO] URL presignada nao definida para ${target}" >&2
            exit 1
        fi
        log "  baixando ${target}..."
        wget --quiet --show-progress -O "${dest}" "${url}"
    done
}

download_assets_scp() {
    log "Baixando assets via SCP de ${SCP_USER}@${SCP_HOST}:${SCP_BASE_PATH}..."
    mkdir -p "${REPO_DIR}/data/processed" "${REPO_DIR}/tokenizer"

    local files=(
        "data/processed/corpus_v2.txt"
        "data/processed/tokens.npy"
        "tokenizer/maracatu.model"
    )

    for f in "${files[@]}"; do
        local dest="${REPO_DIR}/${f}"
        if [[ -f "${dest}" ]]; then
            log "  ja existe: ${f}, pulando."
            continue
        fi
        log "  scp ${f}..."
        scp -o StrictHostKeyChecking=no \
            "${SCP_USER}@${SCP_HOST}:${SCP_BASE_PATH}/${f}" \
            "${dest}"
    done
}

download_assets() {
    case "${CORPUS_METHOD}" in
        hf)
            if [[ -z "${HF_TOKEN:-}" ]]; then
                echo "[ERRO] CORPUS_METHOD=hf requer HF_TOKEN definido." >&2
                exit 1
            fi
            download_assets_hf
            ;;
        presigned)
            download_assets_presigned
            ;;
        scp)
            if [[ -z "${SCP_HOST:-}" || -z "${SCP_USER:-}" || -z "${SCP_BASE_PATH:-}" ]]; then
                echo "[ERRO] CORPUS_METHOD=scp requer SCP_HOST, SCP_USER e SCP_BASE_PATH." >&2
                exit 1
            fi
            download_assets_scp
            ;;
        *)
            echo "[ERRO] CORPUS_METHOD invalido: '${CORPUS_METHOD}'. Use: hf | presigned | scp" >&2
            exit 1
            ;;
    esac

    local required_files=(
        "${REPO_DIR}/data/processed/corpus_v2.txt"
        "${REPO_DIR}/data/processed/tokens.npy"
        "${REPO_DIR}/tokenizer/maracatu.model"
    )
    for f in "${required_files[@]}"; do
        if [[ ! -f "${f}" ]]; then
            echo "[ERRO] Asset nao encontrado apos download: ${f}" >&2
            exit 1
        fi
        local size
        size=$(du -sh "${f}" | cut -f1)
        log "  ${f} (${size}) OK"
    done
}

start_training_in_tmux() {
    local session="train"

    if tmux has-session -t "${session}" 2>/dev/null; then
        log "Sessao tmux '${session}' ja existe."
        log "Para ver o log: tmux attach -t ${session}"
        log "Para sair sem matar: Ctrl+B D"
        return
    fi

    log "Iniciando sessao tmux '${session}' com treino..."
    tmux new-session -d -s "${session}" \
        "bash ${REPO_DIR}/scripts/runpod_train.sh 2>&1; echo 'PROCESSO FINALIZADO'; bash"

    log ""
    log "============================================================"
    log "Treino iniciado em background (tmux: ${session})"
    log ""
    log "  Reconectar:     tmux attach -t ${session}"
    log "  Sair sem matar: Ctrl+B D"
    log "  Ver log:        tail -f ${REPO_DIR}/checkpoints/training.log"
    log "  GPU usage:      watch -n5 nvidia-smi"
    log "============================================================"
    log ""
}

main() {
    log "=== Maracatu RunPod Bootstrap ==="
    log "Repo dir:      ${REPO_DIR}"
    log "Corpus method: ${CORPUS_METHOD}"
    log ""

    check_preconditions
    install_system_deps
    clone_or_update_repo
    install_uv
    create_venv_and_install
    configure_hf_token
    download_assets
    start_training_in_tmux

    log "Bootstrap concluido."
}

main "$@"

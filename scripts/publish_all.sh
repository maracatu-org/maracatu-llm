#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

CHECKPOINT=""
TOKENIZER=""
MODEL_NAME=""
OLLAMA_USER="whereisanzi"
SKIP_HF=false
SKIP_GGUF=false
SKIP_OLLAMA=false
SKIP_KAGGLE=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --checkpoint)   CHECKPOINT="$(realpath "$2")"; shift 2 ;;
        --tokenizer)    TOKENIZER="$(realpath "$2")";  shift 2 ;;
        --name)         MODEL_NAME="$2";               shift 2 ;;
        --ollama-user)  OLLAMA_USER="$2";              shift 2 ;;
        --skip-hf)      SKIP_HF=true;                  shift   ;;
        --skip-gguf)    SKIP_GGUF=true;                shift   ;;
        --skip-ollama)  SKIP_OLLAMA=true;              shift   ;;
        --skip-kaggle)  SKIP_KAGGLE=true;              shift   ;;
        *)
            echo "Argumento desconhecido: $1" >&2
            echo "Uso: $0 --checkpoint <path> --tokenizer <path> --name <scale> [--ollama-user <user>] [--skip-hf] [--skip-gguf] [--skip-ollama] [--skip-kaggle]" >&2
            exit 1
            ;;
    esac
done

if [[ -z "$CHECKPOINT" || -z "$TOKENIZER" || -z "$MODEL_NAME" ]]; then
    echo "ERRO: --checkpoint, --tokenizer e --name são obrigatórios." >&2
    exit 1
fi

if [[ ! -f "$CHECKPOINT" ]]; then
    echo "ERRO: checkpoint não encontrado: ${CHECKPOINT}" >&2
    exit 1
fi

if [[ ! -f "$TOKENIZER" ]]; then
    echo "ERRO: tokenizer não encontrado: ${TOKENIZER}" >&2
    exit 1
fi

HF_REPO="maracatu-labs/${MODEL_NAME}"
OLLAMA_REPO="${OLLAMA_USER}/${MODEL_NAME}"
EXPORT_HF_DIR="${REPO_ROOT}/exports/${MODEL_NAME}-hf"
EXPORT_GGUF_DIR="${REPO_ROOT}/exports/${MODEL_NAME}-gguf"
MODELFILE_TEMPLATE="${REPO_ROOT}/.kaggle/modelfile/Modelfile.template"
MODELFILE_RENDERED="/tmp/Modelfile.${MODEL_NAME}"

if [[ -f "${REPO_ROOT}/.env" ]]; then
    set -a; source "${REPO_ROOT}/.env"; set +a
    echo "[publish] .env carregado"
fi

log_step() { echo ""; echo "==> $*"; }
log_ok()   { echo "    OK: $*"; }
log_skip() { echo "    (pulado)"; }

log_step "[1/6] Export HF: checkpoint → safetensors"

if $SKIP_HF; then
    log_skip
elif [[ -f "${EXPORT_HF_DIR}/model.safetensors" || -f "${EXPORT_HF_DIR}/model-00001-of-*.safetensors" ]]; then
    echo "    Export HF já existe em ${EXPORT_HF_DIR} — pulando conversão."
    echo "    Para forçar re-export, remova o diretório e rode novamente."
else
    python "${REPO_ROOT}/scripts/export_hf.py" \
        --checkpoint "${CHECKPOINT}" \
        --tokenizer "${TOKENIZER}" \
        --output-dir "${EXPORT_HF_DIR}"
    log_ok "Artefatos HF em ${EXPORT_HF_DIR}"
fi

log_step "[2/6] Upload HF Hub: ${HF_REPO}"

if $SKIP_HF; then
    log_skip
else
    hf upload "${HF_REPO}" "${EXPORT_HF_DIR}" .
    log_ok "Publicado em https://huggingface.co/${HF_REPO}"
fi

log_step "[3/6] GGUF: conversão fp16 + quantizações Q4_K_M / Q5_K_M / Q8_0"

if $SKIP_GGUF; then
    log_skip
else
    bash "${REPO_ROOT}/scripts/export_gguf.sh" "${EXPORT_HF_DIR}" "${EXPORT_GGUF_DIR}"
    log_ok "GGUFs em ${EXPORT_GGUF_DIR}"
fi

log_step "[4/6] Upload GGUF ao HF Hub (subpasta gguf/)"

if $SKIP_GGUF; then
    log_skip
else
    for quant in Q4_K_M Q5_K_M Q8_0; do
        local_file="${EXPORT_GGUF_DIR}/${MODEL_NAME}-${quant}.gguf"
        remote_path="gguf/${MODEL_NAME}-${quant}.gguf"
        if [[ -f "${local_file}" ]]; then
            hf upload "${HF_REPO}" "${local_file}" "${remote_path}"
            log_ok "${remote_path}"
        else
            echo "    AVISO: ${local_file} não encontrado, pulando upload deste quant."
        fi
    done
fi

log_step "[5/6] Ollama: create + push → ollama.com/${OLLAMA_REPO}"

if $SKIP_OLLAMA; then
    log_skip
else
    if [[ ! -f "${MODELFILE_TEMPLATE}" ]]; then
        echo "    ERRO: ${MODELFILE_TEMPLATE} não encontrado." >&2
        echo "    Crie o arquivo antes de rodar este script." >&2
        exit 1
    fi

    GGUF_Q4="${EXPORT_GGUF_DIR}/${MODEL_NAME}-Q4_K_M.gguf"
    if [[ ! -f "${GGUF_Q4}" ]]; then
        echo "    ERRO: ${GGUF_Q4} não encontrado. Execute sem --skip-gguf primeiro." >&2
        exit 1
    fi

    sed \
        "s|<name>|${MODEL_NAME}|g; s|<gguf_path>|${GGUF_Q4}|g" \
        "${MODELFILE_TEMPLATE}" > "${MODELFILE_RENDERED}"

    ollama create "${OLLAMA_REPO}" -f "${MODELFILE_RENDERED}"
    log_ok "Modelo Ollama local criado: ${OLLAMA_REPO}"

    echo "    Smoke test Ollama..."
    OLLAMA_OUT="$(ollama run "${OLLAMA_REPO}" "O Brasil é" --nowordwrap 2>&1 | head -5)"
    echo "    Output: ${OLLAMA_OUT}"

    ollama push "${OLLAMA_REPO}"
    log_ok "Publicado em https://ollama.com/${OLLAMA_REPO}"
fi

log_step "[6/6] Kaggle Models (requer ação manual)"

if $SKIP_KAGGLE; then
    log_skip
else
    echo ""
    echo "  O Kaggle CLI 1.5.16 não suporta o subcomando 'models'."
    echo "  Siga os passos manuais:"
    echo ""
    echo "  1. Acesse: https://www.kaggle.com/models/create"
    echo "  2. Owner: whereisanzi | Name: ${MODEL_NAME}"
    echo "  3. Framework: Other | Task: Text Generation"
    echo "  4. Faça upload dos artefatos:"
    echo "       - ${EXPORT_HF_DIR}/model.safetensors"
    echo "       - ${EXPORT_HF_DIR}/config.json"
    echo "       - ${EXPORT_HF_DIR}/tokenizer.model"
    echo "       - ${EXPORT_HF_DIR}/tokenizer_config.json"
    echo "       - ${EXPORT_HF_DIR}/special_tokens_map.json"
    echo "       - ${EXPORT_GGUF_DIR}/${MODEL_NAME}-Q4_K_M.gguf"
    echo ""
fi

echo ""
echo "============================================================"
echo "  Publicacao concluida — ${MODEL_NAME}"
echo "============================================================"
echo ""
if ! $SKIP_HF; then
    echo "  HF Hub:   https://huggingface.co/${HF_REPO}"
fi
if ! $SKIP_OLLAMA; then
    echo "  Ollama:   https://ollama.com/${OLLAMA_REPO}"
fi
if ! $SKIP_KAGGLE; then
    echo "  Kaggle:   https://www.kaggle.com/models/whereisanzi/${MODEL_NAME}"
fi
echo ""
echo "  Proximos passos:"
echo "    git tag -a v0.1.0 -m '${MODEL_NAME}: primeiro release publico'"
echo "    git push origin v0.1.0"
echo "    Atualizar release notes no HF Hub com training config e benchmarks."
echo "============================================================"

#!/usr/bin/env bash

set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "Uso: $0 <hf_input_dir> <output_dir>" >&2
    exit 1
fi

HF_INPUT_DIR="$(realpath "$1")"
mkdir -p "$2"
OUTPUT_DIR="$(realpath "$2")"

MODEL_NAME="$(basename "$HF_INPUT_DIR")"
MODEL_NAME="${MODEL_NAME%-hf}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CACHE_DIR="${REPO_ROOT}/.cache"
LLAMA_CPP_DIR="${CACHE_DIR}/llama.cpp"
LLAMA_BUILD_DIR="${LLAMA_CPP_DIR}/build"
LLAMA_QUANTIZE_BIN="${LLAMA_BUILD_DIR}/bin/llama-quantize"
CONVERT_SCRIPT="${LLAMA_CPP_DIR}/convert_hf_to_gguf.py"

GGUF_FP16="${OUTPUT_DIR}/${MODEL_NAME}-fp16.gguf"
GGUF_Q4="${OUTPUT_DIR}/${MODEL_NAME}-Q4_K_M.gguf"
GGUF_Q5="${OUTPUT_DIR}/${MODEL_NAME}-Q5_K_M.gguf"
GGUF_Q8="${OUTPUT_DIR}/${MODEL_NAME}-Q8_0.gguf"

log() { echo "[export_gguf] $*"; }

check_cmake() {
    if ! command -v cmake &>/dev/null; then
        echo "[export_gguf] ERRO: cmake não encontrado. Instale com: brew install cmake" >&2
        exit 1
    fi
    local cmake_version
    cmake_version="$(cmake --version | head -1 | awk '{print $3}')"
    log "cmake ${cmake_version} encontrado"
}

sha256_file() {
    if command -v shasum &>/dev/null; then
        shasum -a 256 "$1" | awk '{print $1}'
    else
        sha256sum "$1" | awk '{print $1}'
    fi
}

human_size() {
    local size
    size="$(wc -c < "$1" | tr -d ' ')"
    if [[ $size -ge $((1024 * 1024 * 1024)) ]]; then
        echo "$(echo "scale=1; $size / 1073741824" | bc)GB"
    elif [[ $size -ge $((1024 * 1024)) ]]; then
        echo "$(echo "scale=1; $size / 1048576" | bc)MB"
    else
        echo "$(echo "scale=1; $size / 1024" | bc)KB"
    fi
}

log "Verificando entrada: ${HF_INPUT_DIR}"
if [[ ! -f "${HF_INPUT_DIR}/config.json" ]]; then
    echo "[export_gguf] ERRO: ${HF_INPUT_DIR}/config.json não encontrado. Execute export_hf.py primeiro." >&2
    exit 1
fi

check_cmake
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${CACHE_DIR}"

if [[ ! -d "${LLAMA_CPP_DIR}/.git" ]]; then
    log "Clonando llama.cpp em ${LLAMA_CPP_DIR}..."
    git clone --depth 1 https://github.com/ggerganov/llama.cpp.git "${LLAMA_CPP_DIR}"
else
    log "llama.cpp já presente em ${LLAMA_CPP_DIR}"
fi

if [[ ! -f "${LLAMA_QUANTIZE_BIN}" ]]; then
    log "Building llama-quantize..."
    cmake -S "${LLAMA_CPP_DIR}" -B "${LLAMA_BUILD_DIR}" \
        -DCMAKE_BUILD_TYPE=Release \
        -DLLAMA_BUILD_SERVER=OFF \
        -DLLAMA_BUILD_TESTS=OFF \
        -DLLAMA_BUILD_EXAMPLES=OFF
    cmake --build "${LLAMA_BUILD_DIR}" --target llama-quantize -j "$(sysctl -n hw.logicalcpu 2>/dev/null || nproc)"
    log "Build concluído: ${LLAMA_QUANTIZE_BIN}"
else
    log "llama-quantize já buildado: ${LLAMA_QUANTIZE_BIN}"
fi

if [[ ! -f "${CONVERT_SCRIPT}" ]]; then
    echo "[export_gguf] ERRO: ${CONVERT_SCRIPT} não encontrado. Repositório llama.cpp incompleto." >&2
    exit 1
fi

LLAMA_REQS="${LLAMA_CPP_DIR}/requirements/requirements-convert_hf_to_gguf.txt"
if [[ -f "${LLAMA_REQS}" ]]; then
    log "Instalando dependências Python do llama.cpp..."
    pip install -q -r "${LLAMA_REQS}"
fi

if [[ ! -f "${GGUF_FP16}" ]]; then
    log "Convertendo HF → GGUF fp16..."
    python "${CONVERT_SCRIPT}" \
        "${HF_INPUT_DIR}" \
        --outtype f16 \
        --outfile "${GGUF_FP16}"
    log "GGUF fp16 gerado: ${GGUF_FP16}"
else
    log "GGUF fp16 já existe: ${GGUF_FP16}"
fi

quantize() {
    local quant_type="$1"
    local output_file="$2"

    if [[ ! -f "${output_file}" ]]; then
        log "Quantizando ${quant_type}..."
        "${LLAMA_QUANTIZE_BIN}" "${GGUF_FP16}" "${output_file}" "${quant_type}"
        log "Gerado: ${output_file}"
    else
        log "${quant_type} já existe: ${output_file}"
    fi
}

quantize "Q4_K_M" "${GGUF_Q4}"
quantize "Q5_K_M" "${GGUF_Q5}"
quantize "Q8_0"   "${GGUF_Q8}"

echo ""
echo "============================================================"
echo "  Artefatos GGUF — ${MODEL_NAME}"
echo "============================================================"
echo ""

for f in "${GGUF_FP16}" "${GGUF_Q8}" "${GGUF_Q5}" "${GGUF_Q4}"; do
    if [[ -f "$f" ]]; then
        name="$(basename "$f")"
        size="$(human_size "$f")"
        sha="$(sha256_file "$f")"
        printf "  %-40s  %8s  SHA256: %s\n" "${name}" "${size}" "${sha}"
    fi
done

echo ""
echo "  Diretório: ${OUTPUT_DIR}"
echo ""
echo "  Próximos passos:"
echo "    Upload HF:"
echo "      huggingface-cli upload maracatu-ai/${MODEL_NAME} ${GGUF_Q4} gguf/${MODEL_NAME}-Q4_K_M.gguf"
echo "      huggingface-cli upload maracatu-ai/${MODEL_NAME} ${GGUF_Q5} gguf/${MODEL_NAME}-Q5_K_M.gguf"
echo "      huggingface-cli upload maracatu-ai/${MODEL_NAME} ${GGUF_Q8} gguf/${MODEL_NAME}-Q8_0.gguf"
echo ""
echo "    Ollama (ajustar path no Modelfile antes):"
echo "      ollama create maracatu-ai/${MODEL_NAME} -f Modelfile"
echo "      ollama push maracatu-ai/${MODEL_NAME}"
echo "============================================================"

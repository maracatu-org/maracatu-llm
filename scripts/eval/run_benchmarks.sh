
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODEL="${1:-}"

if [[ -z "$MODEL" ]]; then
    echo "Uso: bash scripts/eval/run_benchmarks.sh <hf_repo_ou_path_local>"
    exit 1
fi

MODEL_SLUG="$(echo "$MODEL" | tr '/' '-' | tr '.' '-')"
TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
OUTPUT_BASE="${REPO_ROOT}/eval_results/${MODEL_SLUG}-${TIMESTAMP}"
TASK_PATH="${REPO_ROOT}/scripts/eval/tasks"
LM_EVAL="${REPO_ROOT}/.cache/lm-eval-venv/bin/lm_eval"
SEED=42
BATCH=8

echo "===== Maracatu AI Benchmark Pipeline ====="
echo "Modelo:    $MODEL"
echo "Output:    $OUTPUT_BASE"
echo "Seed:      $SEED"
echo "Batch:     $BATCH"
echo "=========================================="

run_eval() {
    local task="$1"
    local fewshot="${2:-0}"
    local extra_args="${3:-}"
    local out_dir="${OUTPUT_BASE}/${task}-${fewshot}shot"

    echo ""
    echo ">>> Rodando: $task (${fewshot}-shot)..."
    "$LM_EVAL" \
        --model hf \
        --model_args "pretrained=${MODEL},trust_remote_code=True" \
        --tasks "$task" \
        --include_path "$TASK_PATH" \
        --num_fewshot "$fewshot" \
        --batch_size "$BATCH" \
        --seed "$SEED" \
        --output_path "$out_dir" \
        $extra_args \
        2>&1
    echo "<<< $task done. Resultados em: $out_dir"
}

run_eval "belebele_por_Latn" 0

run_eval "assin_entailment,assin_paraphrase" 0

run_eval "enem_challenge" 0
run_eval "enem_challenge" 3

echo ""
echo "===== Pipeline completo ====="
echo "Resultados em: $OUTPUT_BASE"
echo ""

python3 - <<'PYEOF'
import json, glob, os, sys

base = os.environ.get('OUTPUT_BASE', '.')
if not os.path.exists(base):
    print("Diretorio de output nao encontrado.")
    sys.exit(0)

print(f"{'Modelo':30s} | {'Task':25s} | {'acc':>6s} | {'stderr':>6s}")
print("-" * 80)
for jf in sorted(glob.glob(f'{base}/**/*.json', recursive=True)):
    with open(jf) as f:
        data = json.load(f)
    for task, metrics in data.get('results', {}).items():
        acc = metrics.get('acc,none')
        stderr = metrics.get('acc_stderr,none')
        if acc is not None:
            print(f"{'':30s} | {task:25s} | {acc:6.4f} | {stderr:6.4f}")
PYEOF

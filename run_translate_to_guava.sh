#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash run_translate_to_guava.sh "我爸爸在新竹上班，家住臺北，從臺北開車到新竹要二個小時。"
#   bash run_translate_to_guava.sh "我爸爸在新竹上班，家住臺北，從臺北開車到新竹要二個小時。" --no-render
# or:
#   bash run_translate_to_guava.sh
#   # then type input interactively
#
# Arguments after the first sentence are forwarded to gloss_to_guava_pipeline.py.
# For example: --no-render, --dry-run, --output-name demo, -d 0

CONDA_BASE="${HOME}/anaconda3"
SIGN_ENV="sign_translate_env"
GUAVA_ENV="GUAVA"
SIGN_PROJECT="${HOME}/GUAVA/sign_translate_code"
GUAVA_PROJECT="${HOME}/GUAVA"
TRANSLATE_PY="${SIGN_PROJECT}/CODE/GUI/translate_cli.py"
TEMP_OUTPUT="${SIGN_PROJECT}/gloss_output.txt"
PIPELINE_PY="${GUAVA_PROJECT}/gloss_to_guava_pipeline.py"

if [[ $# -ge 1 ]]; then
  INPUT_TEXT="$1"
  shift
else
  read -r -p "請輸入自然語言句子: " INPUT_TEXT
fi
PIPELINE_ARGS=("$@")

if [[ -z "${INPUT_TEXT}" ]]; then
  echo "錯誤：沒有輸入文字" >&2
  exit 1
fi

if [[ ! -f "${TRANSLATE_PY}" ]]; then
  echo "錯誤：找不到翻譯程式：${TRANSLATE_PY}" >&2
  exit 1
fi

if [[ ! -f "${PIPELINE_PY}" ]]; then
  echo "錯誤：找不到 GUAVA pipeline：${PIPELINE_PY}" >&2
  exit 1
fi

# enable conda in bash
source "${CONDA_BASE}/etc/profile.d/conda.sh"

# Step 1: run translation in sign_translate_env
conda activate "${SIGN_ENV}"
cd "${SIGN_PROJECT}/CODE/GUI"
python "${TRANSLATE_PY}" --text "${INPUT_TEXT}" > "${TEMP_OUTPUT}"

echo "=== sign_translate_env output ==="
cat "${TEMP_OUTPUT}"

# Extract only the joined gloss line
GLOSS_JOINED=$(grep '^Gloss joined:' "${TEMP_OUTPUT}" | sed 's/^Gloss joined:[[:space:]]*//')

if [[ -z "${GLOSS_JOINED}" ]]; then
  echo "錯誤：沒有抓到 Gloss joined 結果" >&2
  exit 1
fi

# Step 2: switch to GUAVA env and print result
conda activate "${GUAVA_ENV}"
cd "${GUAVA_PROJECT}"

echo
echo "=== GUAVA env output ==="
echo "CONDA_DEFAULT_ENV=$CONDA_DEFAULT_ENV"
echo "which python: $(which python)"
python -c "import sys; print('sys.executable:', sys.executable)"
echo "收到手語 gloss 結果：${GLOSS_JOINED}"

echo
echo "=== GUAVA pipeline ==="
PYTHONPATH="${GUAVA_PROJECT}:${PYTHONPATH:-}" python "${PIPELINE_PY}" \
  --gloss-output "${TEMP_OUTPUT}" \
  "${PIPELINE_ARGS[@]}"

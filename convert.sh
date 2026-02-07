#!/bin/bash
# 講義轉換快捷指令
# 用法: ./convert.sh <講義.docx> [--pages] [--batch]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="$(dirname "$SCRIPT_DIR")"

# 啟動虛擬環境
source "$WORKSPACE/ocr-env/bin/activate"

# 設定環境變數
export PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True

# 執行轉換
python3 "$SCRIPT_DIR/convert_lecture.py" "$@"

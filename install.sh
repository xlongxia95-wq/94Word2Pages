#!/bin/bash
# 94Word2Pages 安裝腳本

echo "📚 94Word2Pages 安裝程式"
echo "========================"

# 檢查 Python
if ! command -v python3 &> /dev/null; then
    echo "❌ 請先安裝 Python 3.11+"
    exit 1
fi

# 建立虛擬環境
echo "📦 建立虛擬環境..."
python3 -m venv ocr-env
source ocr-env/bin/activate

# 安裝依賴
echo "📦 安裝依賴套件..."
pip install --upgrade pip
pip install paddlepaddle paddleocr playwright

# 安裝 Chromium
echo "🌐 安裝 Chromium..."
playwright install chromium

echo ""
echo "✅ 安裝完成！"
echo ""
echo "使用方式:"
echo "  source ocr-env/bin/activate"
echo "  python convert_lecture.py 講義.docx --pages"

#!/usr/bin/env python3
"""
講義轉換工具 v2.0
陳建豪物理 - 小豪編輯部

完整流程：Word (.docx) → HTML → 截圖 → OCR → Markdown → Pages
"""

import os
import sys
import json
import subprocess
import asyncio
import re
from pathlib import Path
from datetime import datetime

SCRIPT_DIR = Path(__file__).parent.parent
OCR_ENV = SCRIPT_DIR / "ocr-env"

# ============================================================
# Step 1: Word → HTML
# ============================================================
def docx_to_html(docx_path: Path, output_dir: Path) -> Path:
    """Word 轉 HTML"""
    html_path = output_dir / f"{docx_path.stem}.html"
    
    result = subprocess.run([
        "textutil", "-convert", "html",
        "-output", str(html_path),
        str(docx_path)
    ], capture_output=True, text=True)
    
    if html_path.exists():
        print(f"✅ Word → HTML: {html_path.name}")
        return html_path
    
    print(f"❌ 轉換失敗: {result.stderr}")
    return None

# ============================================================
# Step 2: HTML → 截圖（Playwright）
# ============================================================
async def html_to_screenshots(html_path: Path, output_dir: Path, page_height: int = 1200) -> list:
    """HTML 轉多頁截圖"""
    from playwright.async_api import async_playwright
    
    screenshots = []
    
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page(viewport={"width": 800, "height": page_height})
        
        # 載入 HTML（使用絕對路徑）
        abs_path = html_path.resolve()
        await page.goto(f"file://{abs_path}")
        await page.wait_for_load_state("networkidle")
        
        # 取得頁面總高度
        total_height = await page.evaluate("document.body.scrollHeight")
        print(f"📄 頁面總高度: {total_height}px")
        
        # 分頁截圖
        page_num = 0
        y_offset = 0
        
        while y_offset < total_height:
            page_num += 1
            
            # 滾動到指定位置
            await page.evaluate(f"window.scrollTo(0, {y_offset})")
            await asyncio.sleep(0.3)  # 等待渲染
            
            # 截圖
            img_path = output_dir / f"page_{page_num:03d}.png"
            await page.screenshot(path=str(img_path), full_page=False)
            
            screenshots.append(img_path)
            print(f"  📸 第 {page_num} 頁: {img_path.name}")
            
            y_offset += page_height - 100  # 留些重疊避免漏字
        
        await browser.close()
    
    print(f"✅ 共截取 {len(screenshots)} 頁")
    return screenshots

# ============================================================
# Step 3: 截圖 → OCR
# ============================================================
def ocr_image(image_path: Path) -> dict:
    """OCR 單張圖片"""
    env = os.environ.copy()
    env["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
    
    ocr_script = f'''
import sys
sys.path.insert(0, "{OCR_ENV}/lib/python3.11/site-packages")
from paddleocr import PaddleOCR
import json

ocr = PaddleOCR(lang='ch')
result = ocr.predict("{image_path}")

lines = []
for item in result:
    for text, score, box in zip(item['rec_texts'], item['rec_scores'], item.get('rec_boxes', [[]]* len(item['rec_texts']))):
        if score > 0.5:
            lines.append({{"text": text, "score": float(score), "y": int(box[1]) if len(box) > 1 else 0}})

# 按 y 座標排序
lines.sort(key=lambda x: x.get("y", 0))
print(json.dumps(lines, ensure_ascii=False))
'''
    
    result = subprocess.run(
        [f"{OCR_ENV}/bin/python3", "-c", ocr_script],
        capture_output=True, text=True, env=env
    )
    
    if result.returncode == 0:
        for line in reversed(result.stdout.strip().split('\n')):
            if line.startswith('['):
                try:
                    return json.loads(line)
                except:
                    pass
    
    return []

# ============================================================
# Step 4: OCR → Markdown（智慧格式化）
# ============================================================
def convert_to_latex(text: str) -> str:
    """將物理公式轉換為 LaTeX 格式"""
    result = text
    
    # 常見物理符號替換
    replacements = [
        # 希臘字母
        (r'α', r'$\\alpha$'),
        (r'β', r'$\\beta$'),
        (r'γ', r'$\\gamma$'),
        (r'θ', r'$\\theta$'),
        (r'ω', r'$\\omega$'),
        (r'τ', r'$\\tau$'),
        (r'Σ', r'$\\Sigma$'),
        (r'Δ', r'$\\Delta$'),
        
        # 數學運算
        (r'√(\w+)', r'$\\sqrt{\1}$'),
        (r'(\w+)²', r'$\1^2$'),
        (r'(\w+)³', r'$\1^3$'),
        
        # 物理公式模式
        (r'F\s*=\s*ma', r'$F = ma$'),
        (r'E\s*=\s*mc²', r'$E = mc^2$'),
        (r'v\s*=\s*rω', r'$v = r\\omega$'),
        (r'L\s*=\s*rmv', r'$L = rmv$'),
        (r'L\s*=\s*Iω', r'$L = I\\omega$'),
        (r'τ\s*=\s*r×F', r'$\\tau = r \\times F$'),
        (r'τ\s*=\s*dL/dt', r'$\\tau = \\frac{dL}{dt}$'),
        
        # 向量表示
        (r'→(\w)', r'$\\vec{\1}$'),
    ]
    
    for pattern, replacement in replacements:
        result = re.sub(pattern, replacement, result)
    
    return result

def format_to_markdown(all_lines: list, title: str) -> str:
    """將 OCR 結果轉成格式化 Markdown"""
    md = [f"# {title}", ""]
    md.append(f"_轉換時間: {datetime.now().strftime('%Y-%m-%d %H:%M')}_")
    md.append("")
    md.append("---")
    md.append("")
    
    prev_was_option = False
    
    for item in all_lines:
        text = item["text"].strip()
        score = item.get("score", 0)
        
        if not text or score < 0.6:
            continue
        
        # === 標題偵測 ===
        if text.startswith("Example") or text.startswith("例題"):
            md.append("")
            md.append(f"### 📝 {text}")
            md.append("")
            prev_was_option = False
            continue
        
        # 數字標題 (一、二、三...)
        if re.match(r'^[一二三四五六七八九十]+、', text):
            md.append("")
            md.append(f"#### {text}")
            md.append("")
            prev_was_option = False
            continue
        
        # === 選項偵測 ===
        if re.match(r'^\(?[A-Ea-e][)）]', text):
            md.append(f"- {text}")
            prev_was_option = True
            continue
        
        # === 解說區塊 ===
        if text.startswith("【解說】") or text.startswith("【思考"):
            md.append("")
            md.append(f"**{text}**")
            md.append("")
            prev_was_option = False
            continue
        
        # === 題目來源 ===
        if text.startswith("【") and "】" in text:
            md.append(f"\n> {text}\n")
            prev_was_option = False
            continue
        
        # === 一般文字 ===
        if prev_was_option:
            md.append("")
        
        # 嘗試轉換 LaTeX
        text = convert_to_latex(text)
        md.append(text)
        prev_was_option = False
    
    return "\n".join(md)

# ============================================================
# Step 5: Markdown → Pages（AppleScript）
# ============================================================
def markdown_to_pages(md_path: Path, output_dir: Path, auto_pages: bool = False) -> Path:
    """Markdown 轉 Pages"""
    pages_path = output_dir / f"{md_path.stem}.pages"
    rtf_path = output_dir / f"{md_path.stem}.rtf"
    
    # Step 1: Markdown → RTF
    result = subprocess.run([
        "pandoc", str(md_path),
        "-o", str(rtf_path),
        "-f", "markdown",
        "-t", "rtf"
    ], capture_output=True, text=True)
    
    if not rtf_path.exists():
        print(f"❌ RTF 轉換失敗")
        return None
    
    print(f"✅ Markdown → RTF: {rtf_path.name}")
    
    # Step 2: RTF → Pages（使用 AppleScript）
    if auto_pages:
        applescript = f'''
        tell application "Pages"
            activate
            open POSIX file "{rtf_path.resolve()}"
            delay 2
            tell front document
                save in POSIX file "{pages_path.resolve()}"
                close
            end tell
        end tell
        '''
        
        result = subprocess.run(
            ["osascript", "-e", applescript],
            capture_output=True, text=True
        )
        
        if pages_path.exists():
            print(f"✅ RTF → Pages: {pages_path.name}")
            return pages_path
        else:
            print(f"⚠️ Pages 自動轉換失敗，請手動開啟 RTF")
    
    return rtf_path

# ============================================================
# 批次處理
# ============================================================
async def batch_convert(input_dir: str, output_dir: str = None, auto_pages: bool = False):
    """批次轉換多個 Word 檔案"""
    input_dir = Path(input_dir)
    
    if not input_dir.exists():
        print(f"❌ 目錄不存在: {input_dir}")
        return
    
    # 找出所有 .docx 檔案
    docx_files = list(input_dir.glob("*.docx"))
    
    if not docx_files:
        print(f"❌ 找不到 .docx 檔案: {input_dir}")
        return
    
    print(f"\n{'='*60}")
    print(f"📚 批次轉換模式")
    print(f"{'='*60}")
    print(f"📁 來源目錄: {input_dir}")
    print(f"📄 檔案數量: {len(docx_files)}")
    print(f"{'='*60}\n")
    
    results = []
    
    for i, docx_file in enumerate(docx_files, 1):
        print(f"\n[{i}/{len(docx_files)}] 處理: {docx_file.name}")
        print("-" * 40)
        
        if output_dir:
            out = Path(output_dir) / docx_file.stem
        else:
            out = docx_file.parent / f"{docx_file.stem}_output"
        
        result = await convert_lecture(str(docx_file), str(out), auto_pages)
        results.append({
            "file": docx_file.name,
            "success": result.get("success", False) if result else False,
            "output": str(out)
        })
    
    # 總結
    success_count = sum(1 for r in results if r["success"])
    
    print(f"\n{'='*60}")
    print(f"📊 批次轉換完成")
    print(f"{'='*60}")
    print(f"✅ 成功: {success_count}/{len(docx_files)}")
    print(f"❌ 失敗: {len(docx_files) - success_count}/{len(docx_files)}")
    print(f"{'='*60}\n")
    
    return results

# ============================================================
# 主程式
# ============================================================
async def convert_lecture(docx_path: str, output_dir: str = None, auto_pages: bool = False):
    """完整轉換流程"""
    docx_path = Path(docx_path)
    
    if not docx_path.exists():
        print(f"❌ 檔案不存在: {docx_path}")
        return None
    
    # 建立輸出目錄
    if output_dir:
        output_dir = Path(output_dir)
    else:
        output_dir = docx_path.parent / f"{docx_path.stem}_output"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"📚 講義轉換工具 v2.0")
    print(f"{'='*60}")
    print(f"📄 來源: {docx_path.name}")
    print(f"📁 輸出: {output_dir}")
    print(f"{'='*60}\n")
    
    result = {
        "source": str(docx_path),
        "output_dir": str(output_dir),
        "success": False,
        "files": {}
    }
    
    # Step 1: Word → HTML
    print("📄 Step 1/5: Word → HTML")
    html_path = docx_to_html(docx_path, output_dir)
    if not html_path:
        return result
    result["files"]["html"] = str(html_path)
    
    # Step 2: HTML → 截圖
    print("\n📸 Step 2/5: HTML → 截圖")
    try:
        screenshots = await html_to_screenshots(html_path, output_dir)
        result["files"]["screenshots"] = [str(s) for s in screenshots]
    except Exception as e:
        print(f"⚠️ 截圖失敗: {e}")
        print("  使用備用方案：單頁截圖")
        screenshots = []
    
    # Step 3: OCR
    print("\n🔍 Step 3/5: OCR 辨識")
    all_lines = []
    
    if screenshots:
        for i, img in enumerate(screenshots, 1):
            print(f"  處理第 {i}/{len(screenshots)} 頁...")
            lines = ocr_image(img)
            all_lines.extend(lines)
            print(f"    ✅ 辨識 {len(lines)} 行")
    else:
        # 備用：使用現有截圖
        existing = list(output_dir.glob("*.png"))
        if existing:
            for img in existing:
                lines = ocr_image(img)
                all_lines.extend(lines)
    
    print(f"  📊 總計: {len(all_lines)} 行文字")
    
    # Step 4: Markdown
    print("\n📝 Step 4/5: 生成 Markdown")
    title = docx_path.stem.replace("_", " ")
    md_content = format_to_markdown(all_lines, title)
    
    md_path = output_dir / f"{docx_path.stem}.md"
    md_path.write_text(md_content, encoding="utf-8")
    result["files"]["markdown"] = str(md_path)
    print(f"✅ {md_path.name} ({len(md_content)} 字元)")
    
    # Step 5: RTF / Pages
    print("\n📄 Step 5/5: 生成 RTF/Pages")
    output_path = markdown_to_pages(md_path, output_dir, auto_pages)
    if output_path:
        if output_path.suffix == ".pages":
            result["files"]["pages"] = str(output_path)
        else:
            result["files"]["rtf"] = str(output_path)
    
    # 儲存結果
    result["success"] = True
    result["line_count"] = len(all_lines)
    
    result_path = output_dir / "conversion_result.json"
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    print(f"\n{'='*60}")
    print(f"✅ 轉換完成！")
    print(f"{'='*60}")
    print(f"📁 輸出目錄: {output_dir}")
    print(f"📄 Markdown: {md_path.name}")
    if output_path:
        print(f"📄 {output_path.suffix.upper()[1:]}: {output_path.name}")
    print(f"{'='*60}\n")
    
    return result

# ============================================================
# CLI
# ============================================================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="講義轉換工具 v2.0 - 陳建豪物理",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例:
  # 單檔轉換
  python convert_lecture.py 力學講義.docx
  python convert_lecture.py 電磁學.docx -o ./output
  
  # 自動轉 Pages（需要 Pages.app）
  python convert_lecture.py 講義.docx --pages
  
  # 批次轉換整個目錄
  python convert_lecture.py --batch ./講義目錄/
  python convert_lecture.py --batch ./講義目錄/ -o ./輸出目錄/ --pages
"""
    )
    
    parser.add_argument("input", help="Word 檔案 (.docx) 或目錄（配合 --batch）")
    parser.add_argument("-o", "--output", help="輸出目錄")
    parser.add_argument("--batch", action="store_true", help="批次處理目錄內所有 .docx")
    parser.add_argument("--pages", action="store_true", help="自動轉換為 Pages 格式")
    
    args = parser.parse_args()
    
    if args.batch:
        asyncio.run(batch_convert(args.input, args.output, args.pages))
    else:
        asyncio.run(convert_lecture(args.input, args.output, args.pages))

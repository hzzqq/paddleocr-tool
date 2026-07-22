# PaddleOCR 批量图文抽取工具（MVP）

一个轻量的批量 OCR 小工具：对目录下图片 / PDF 进行批量文字识别，输出 Markdown / JSON / CSV。
主后端为 [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)，并内置 **优雅降级** 与 **mock 演示** 能力。

> 适用于：批量发票 / 合同 / 截图 / 扫描件文字抽取；金融文档结构化前的预处理。

---

## 功能特性

- ✅ **批量识别**：自动遍历目录下的图片（`.jpg/.jpeg/.png/.bmp`）与 PDF。
- ✅ **多格式输出**：
  - 逐文件 `<原名>.md`（含识别文本）
  - 合并 `results.json`（文件名、文本、耗时、状态）
  - 合并 `results.csv`（UTF-8-BOM，Excel 友好）
- ✅ **多后端**：`--backend paddle`（默认）或 `--backend tesseract` 兜底。
- ✅ **PDF 支持**：通过 `pdf2image` 转图；poppler 不可用时自动跳过 PDF 并提示。
- ✅ **优雅降级**：PaddleOCR 未安装时程序**不崩溃**，给出清晰安装指引。
- ✅ **Mock 模式**：`--mock` 返回假文本，无需任何 OCR 依赖即可跑通完整流程。
- ✅ **可选 Web 界面**：`ui.py` 提供目录选择 + 一键识别 + 结果下载。

---

## CLI 用法示例

```bash
# 1) 基本用法：识别目录下所有图片，输出 md
python ocr_tool.py --input ./images --output ./out

# 2) 递归 + 指定语言 + 输出 JSON
python ocr_tool.py --input ./docs --output ./out --recursive --lang ch --format json

# 3) 用 tesseract 兜底后端
python ocr_tool.py --input ./docs --output ./out --backend tesseract --lang chi_sim

# 4) 无依赖演示流程（mock 假文本）
python ocr_tool.py --input ./images --output ./out --mock

# 5) 处理单个 PDF
python ocr_tool.py --input ./a.pdf --output ./out
```

参数说明：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--input` | 输入目录或单个文件（图片/PDF） | 必填 |
| `--output` | 输出目录 | 必填 |
| `--lang` | 识别语言（paddle: `ch`；tesseract: `chi_sim`） | `ch` |
| `--format` | 输出格式 `md\|json\|csv` | `md` |
| `--recursive` | 递归遍历子目录 | 关 |
| `--backend` | OCR 后端 `paddle\|tesseract` | `paddle` |
| `--mock` | mock 模式（返回假文本） | 关 |

---

## 依赖安装

### 轻量依赖（推荐先装）

```bash
pip install pytesseract Pillow pdf2image streamlit
```

- **PDF 支持** 还需系统安装 [poppler](https://github.com/oscommunity/opensource-community-files)：
  - Windows：下载并加入 PATH
  - Linux：`sudo apt install poppler-utils`
  - Mac：`brew install poppler`

### 主后端 PaddleOCR（体积较大，按需安装）

> 本工具的体量为数 GB 的 `paddlepaddle` 框架，**默认不随轻量依赖安装**。
> 仅在使用 `--backend paddle`（默认值）时需要。

```bash
pip install paddleocr paddlepaddle
# 如仅需 CPU 版本（推荐）：
pip install paddleocr paddlepaddle-cpu
```

---

## 降级说明

| 场景 | 行为 |
|------|------|
| PaddleOCR 未安装，使用默认后端 | 捕获 `ImportError`，打印安装指引，**程序退出码 1 但不崩溃** |
| 指定 `--backend tesseract` 但 pytesseract 缺失 | 明确报错并打印 tesseract 安装指引 |
| 系统无 poppler | PDF 文件被跳过并打印提示，图片识别不受影响 |
| 完全无 OCR 环境 | 使用 `--mock` 即可跑通完整流程，用于演示与调试 |

---

## Web 界面（可选）

```bash
pip install streamlit
streamlit run ui.py
```

在浏览器中选择输入目录、输出目录、格式，点击「开始识别」，即可查看逐文件结果与下载链接。

---

## 输出示例（results.json）

```json
[
  {
    "file": "./images/demo.png",
    "text": "发票\n金额：100.00 元",
    "elapsed": 1.23,
    "status": "ok"
  }
]
```

---

## 目录结构

```
paddleocr-tool/
├── ocr_tool.py        # CLI 主程序
├── ui.py              # 可选 Streamlit 界面
├── requirements.txt   # 依赖清单
└── README.md          # 本文件
```

## 近期迭代（自驱动开发 10 轮）

- 新增 `--workers` 并行识别（多线程，多图显著提速）
- 新增 `--min-conf` 置信度阈值（仅 paddle 后端生效，低于则丢弃该行）
- 自动跳过隐藏文件（如 `.DS_Store`）
- Streamlit UI 增强：并行/置信度控件、结果指标卡（成功/总数）、逐文件预览与下载
- 附 `run.sh` / `run.bat` 一键启动界面

## 依赖与降级

- **PaddlePaddle 体量警告**：完整 PaddleOCR 后端依赖 paddlepaddle（GB 级），首次安装耗时较长；按需选择。
- **优雅降级**：未安装 PaddleOCR 时程序不崩溃，打印清晰安装指引并退出；可用 `--backend tesseract` 走轻量兜底，或 `--mock` 免任何 OCR 依赖跑通完整流程（含测试）。
- 开发/测试依赖见 `requirements-dev.txt`，运行 `python -m pytest tests -q` 复现测试。


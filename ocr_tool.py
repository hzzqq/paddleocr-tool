#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PaddleOCR 批量图文抽取工具（MVP）
================================

一个轻量的批量 OCR 小工具，模仿百度开源项目 PaddleOCR 的使用体验，
支持对目录下图片 / PDF 进行批量文字识别，并输出 Markdown / JSON / CSV。

设计要点：
- 主后端为 PaddleOCR；若未安装则优雅降级并给出安装指引。
- 支持 `--backend tesseract` 用 pytesseract 兜底（缺失则明确报错）。
- 支持 `--mock` 返回假文本，用于无依赖环境下演示完整流程。
- PDF 通过 pdf2image 转图；若 poppler 不可用则跳过 PDF 并提示。

作者：mirrors-build / build-paddleocr
"""

import argparse
import csv
import fnmatch
import json
import os
import re
import sys
import threading
import time
from pathlib import Path

# 支持的图片扩展名（小写）
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
# 支持的 PDF 扩展名
PDF_EXTS = {".pdf"}

# 并行线程数安全上限（防止 --workers 过大耗尽系统资源）
MAX_WORKERS = 16

# 官方支持的语言代码表（覆盖 PaddleOCR 与 tesseract 两套后端的映射）。
# R1 新能力：--lang-list 列出本表；R2 修复：此前 --lang 为任意字符串，
# 拼写错误（如 chh）会原样透传给后端、报错信息晦涩或静默用错语言，
# 现对未知语言显式告警，并对已知语言按后端解析为正确的引擎代码
# （例如 tesseract 的 fr 应解析为 fra，而非原样传 "fr" 导致识别失败）。
SUPPORTED_LANGS = {
    "ch": {"paddle": "ch", "tesseract": "chi_sim+eng", "name": "简体中文"},
    "en": {"paddle": "en", "tesseract": "eng", "name": "英语"},
    "fr": {"paddle": "french", "tesseract": "fra", "name": "法语"},
    "de": {"paddle": "german", "tesseract": "deu", "name": "德语"},
    "ja": {"paddle": "japan", "tesseract": "jpn", "name": "日语"},
    "ko": {"paddle": "korean", "tesseract": "kor", "name": "韩语"},
    "ru": {"paddle": "russian", "tesseract": "rus", "name": "俄语"},
    "es": {"paddle": "spanish", "tesseract": "spa", "name": "西班牙语"},
    "pt": {"paddle": "portuguese", "tesseract": "por", "name": "葡萄牙语"},
    "it": {"paddle": "italian", "tesseract": "ita", "name": "意大利语"},
}
DEFAULT_LANG = "ch"

# 各后端缺失时的安装提示
INSTALL_HINTS = {
    "paddle": "请安装 PaddleOCR：pip install paddleocr paddlepaddle\n"
              "（注意：paddlepaddle 体积较大，建议按需安装 CPU 版本）",
    "tesseract": "请安装 pytesseract 与 Tesseract 引擎：\n"
                 "  pip install pytesseract\n"
                 "  Windows: 下载安装 https://github.com/tesseract-ocr/tesseract\n"
                 "  Linux:   sudo apt install tesseract-ocr tesseract-ocr-chi-sim\n"
                 "  Mac:     brew install tesseract tesseract-lang",
    "pdf2image": "PDF 转图需要 pdf2image 与 poppler：\n"
                 "  pip install pdf2image\n"
                 "  Windows: 下载 poppler 并加入 PATH（http://blog.alivate.com.au/poppler/）\n"
                 "  Linux:   sudo apt install poppler-utils\n"
                 "  Mac:     brew install poppler",
}


def parse_args(argv=None):
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="PaddleOCR 批量图文抽取工具（MVP）"
    )
    parser.add_argument("--input", required=False, default=None,
                        help="输入目录或单个文件（图片或 PDF）；--lang-list 时可选")
    parser.add_argument("--output", required=False, default=None,
                        help="输出目录；--lang-list 时可选")
    parser.add_argument("--lang", default=DEFAULT_LANG,
                        help=f"识别语言，默认 {DEFAULT_LANG}（中文）。运行 --lang-list 查看全部支持代码")
    parser.add_argument("--lang-list", action="store_true",
                        help="列出所有官方支持的识别语言代码（含两套后端映射）并退出")
    parser.add_argument("--format", choices=["md", "json", "csv", "txt", "jsonl"],
                        default="md", help="输出格式，默认 md（txt 为逐文件纯文本，jsonl 为每行一条 JSON）")
    parser.add_argument("--recursive", action="store_true",
                        help="递归遍历子目录")
    parser.add_argument("--backend", choices=["paddle", "tesseract"],
                        default="paddle", help="OCR 后端，默认 paddle")
    parser.add_argument("--mock", action="store_true",
                        help="使用 mock 模式返回假文本（无需任何 OCR 依赖，用于演示流程）")
    parser.add_argument("--dry-run", action="store_true",
                        help="仅统计待处理文件（按类型），不执行 OCR，便于预检")
    parser.add_argument("--workers", type=int, default=1,
                        help="并行识别的线程数（默认 1，串行）；多图时显著提速")
    parser.add_argument("--min-conf", type=float, default=0.0,
                        help="最低置信度阈值（仅 paddle 后端生效，0~1，低于则丢弃该行）")
    parser.add_argument("--combine", action="store_true",
                        help="额外输出一个合并文件（_combined.md / .txt），把所有成功结果按序拼接，便于下游一次性消费")
    parser.add_argument("--skip-existing", action="store_true",
                        help="跳过已有输出结果的文件（按格式判断），便于断点续跑 / 增量重试，避免重复 OCR")
    parser.add_argument("--low-conf-threshold", type=float, default=-1.0,
                        help="平均置信度低于该值的文件写入 low_confidence.txt 清单（默认 -1 表示不生成；仅 paddle 后端有效）")
    parser.add_argument("--quiet", action="store_true",
                        help="静默模式：不打印逐文件进度，仅输出关键结果与错误（适合脚本/流水线）")
    parser.add_argument("--include", default=None,
                        help="文件名过滤：只处理文件名匹配该子串或 glob 模式（如 '*page*' 或 '封面'）的文件")
    parser.add_argument("--ext", default=None,
                        help="扩展名过滤：只处理指定扩展名（逗号分隔，如 .png,.pdf），覆盖默认图片/PDF 白名单")
    parser.add_argument("--min-chars", type=int, default=0,
                        help="识别字符数低于该值的「成功」结果标记为 filtered（噪声过滤，不计入成功数/合并）")
    parser.add_argument("--max-files", type=int, default=0,
                        help="最多处理的文件数（0 表示不限制），便于对大目录做抽样 / 试跑")
    parser.add_argument("--sort", choices=["name", "size"], default="name",
                        help="处理顺序：name=按文件名字典序（默认，确定性）；size=按体积降序（大文件优先，利于并行吞吐）")
    parser.add_argument("--fail-on-error", action="store_true",
                        help="R1 新能力：任一文件识别失败时返回非零退出码（默认仍返回 0），便于 CI / 流水线把「部分失败」升级为构建失败")
    parser.add_argument("--retries", type=int, default=0,
                        help="R1 新能力：单文件/单页识别失败时的重试次数（默认 0 不重试），提升对瞬时错误的韧性")
    return parser.parse_args(argv)


def _name_matches(name: str, include: str) -> bool:
    """文件名是否匹配 --include（子串或 glob 任一命中即视为匹配）。"""
    if not include:
        return True
    return include in name or fnmatch.fnmatch(name, include)


def collect_files(input_path, recursive, include=None, exts=None):
    """收集需要处理的文件列表（图片 + PDF）。

    返回 (文件列表, 跳过列表)。跳过列表包含不支持类型 / 隐藏文件 / 隐藏目录 /
    未命中 --include / 未命中 --ext，便于调用方透明提示。

    exts：可选扩展名白名单（小写，含点），提供时覆盖默认 IMAGE/PDF 白名单，
    仅处理落在该集合内的文件（R1 新能力，与 --include 的「文件名/子串」维度互补）。
    """
    p = Path(input_path)
    files = []
    skipped = []
    if exts is not None:
        exts = {e.strip().lower() for e in exts.split(",") if e.strip()}
    allowed_exts = exts if exts is not None else (IMAGE_EXTS | PDF_EXTS)
    if p.is_file():
        ext = p.suffix.lower()
        if ext in allowed_exts and _name_matches(p.name, include):
            files.append(p)
        else:
            skipped.append(p)  # 类型不支持 / 未命中过滤
        return files, skipped

    if not p.is_dir():
        # R2 修复（隐性可观测性缺陷）：原实现把「路径不存在」这类错误打印到
        # 标准输出，会被 `ocr ... > files.txt` 这类重定向捕获进产物，污染正常
        # 输出且难以被流水线察觉。错误应走 stderr，与正常进度输出分离。
        print(f"[错误] 输入路径不存在：{input_path}", file=sys.stderr)
        return files, skipped

    # 目录：按扩展名收集
    pattern = "**/*" if recursive else "*"
    for f in sorted(p.glob(pattern)):
        if not f.is_file():
            continue
        # R2 修复（隐性健壮性）：原实现只检查文件自身是否以点开头，漏掉了
        # 「位于隐藏目录内」的文件（如 .git/xxx.png），会把这些无关文件也拉进
        # OCR 批处理——既浪费算力，又可能把 .git 等目录中的敏感文件误识别。
        # 现检查路径上任一层目录是否隐藏，命中则跳过。
        if any(part.startswith(".") for part in f.relative_to(p).parts):
            skipped.append(f)
            continue
        if f.name.startswith("."):
            skipped.append(f)  # 隐藏文件
            continue
        ext = f.suffix.lower()
        if ext in allowed_exts:
            if _name_matches(f.name, include):
                files.append(f)
            else:
                skipped.append(f)  # 未命中 --include
        else:
            skipped.append(f)  # 其他不支持类型
    return files, skipped


def sort_files(files, mode: str = "name"):
    """对收集到的文件列表排序，返回新列表（不修改入参）。

    R1 新能力：大批量处理时控制遍历顺序。
    - "name"（默认）：按路径名字典序，保证每次运行顺序一致、便于复现与调试；
    - "size"：按文件体积降序，优先处理大文件，便于在并行场景下更早启动
      耗时任务、提升整体吞吐（避免把大文件都留到最后）。
    空列表安全返回空列表。
    """
    if not files:
        return []
    if mode == "size":
        return sorted(files, key=lambda p: _safe_size(p), reverse=True)
    # 默认按名称（确定性）
    return sorted(files, key=lambda p: str(p))


def _safe_size(p) -> int:
    try:
        return p.stat().st_size
    except OSError:
        return 0


def collect_all(input_spec, recursive, include=None, exts=None):
    """支持 `--input` 传入多个路径（逗号 / 换行分隔），聚合去重。

    单路径时等价于 collect_files；多路径用于一次性批量处理若干分散文件 / 目录。
    """
    files = []
    skipped = []
    for part in re.split(r"[,\n]", input_spec or ""):
        part = part.strip()
        if not part:
            continue
        f, s = collect_files(part, recursive, include=include, exts=exts)
        files.extend(f)
        skipped.extend(s)
    # 去重并保持顺序
    seen = set()
    uniq = []
    for fpath in files:
        key = str(fpath)
        if key not in seen:
            seen.add(key)
            uniq.append(fpath)
    return uniq, skipped


# 子进程进度行解析（供 ui.py 流式进度复用，抽为纯函数便于单测）
_PROGRESS_RE = re.compile(r"\[进度\].*?(\d+)/(\d+)")


def parse_progress(line: str):
    """从子进程日志行解析 (已完成, 总数)；非进度行返回 None。"""
    if not line:
        return None
    m = _PROGRESS_RE.search(line)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def load_paddle_backend():
    """懒加载 PaddleOCR 后端，缺失时抛出带安装提示的异常。"""
    try:
        from paddleocr import PaddleOCR  # noqa: F401
    except ImportError:
        raise RuntimeError(INSTALL_HINTS["paddle"])
    return PaddleOCR


def load_tesseract_backend():
    """懒加载 pytesseract 后端，缺失时抛出带安装提示的异常。"""
    try:
        import pytesseract  # noqa: F401
        from PIL import Image  # noqa: F401
    except ImportError:
        raise RuntimeError(INSTALL_HINTS["tesseract"])
    return pytesseract, Image


def pdf_to_images(pdf_path):
    """将 PDF 转为临时图片列表；poppler 不可用时返回 None 并提示。"""
    try:
        from pdf2image import convert_from_path
    except ImportError:
        print(f"[跳过] 未安装 pdf2image，跳过 PDF：{pdf_path}\n{INSTALL_HINTS['pdf2image']}")
        return None
    try:
        images = convert_from_path(str(pdf_path), dpi=200)
        return images
    except Exception as e:  # poppler 缺失或转换失败
        print(f"[跳过] PDF 转换失败（可能缺少 poppler）：{pdf_path}\n  原因：{e}\n{INSTALL_HINTS['pdf2image']}")
        return None


# PaddleOCR 识别器懒加载单例：避免每张图都重建（原实现每图 new 一次，
# 多图/并行时是明显性能悬崖），并用锁保证多线程下复用安全。
# 单例按 lang 维度缓存：切换语言时自动重建，避免用错语言的识别器（隐性正确性 bug）。
_paddle_lock = threading.Lock()
_paddle_recognizer = None
_paddle_recognizer_lang = None


def recognize_paddle(recognizer_cls, image_input, lang, min_conf=0.0):
    """用 PaddleOCR 识别单张图片（image_input 可为路径或 PIL.Image）。

    recognizer_cls：PaddleOCR 类（由 build_recognizer 传入）。
    内部以「懒加载单例 + 锁」复用同一个识别器实例，既消除逐图重建的
    性能悬崖，又用锁保证 ThreadPoolExecutor 并行时不会并发踩同一实例。
    当 lang 变化时会自动重建单例，否则复用（正确性 + 性能双重保证）。
    min_conf：最低置信度阈值（0~1），低于该值的识别行将被丢弃（仅 paddle 生效）。
    """
    global _paddle_recognizer, _paddle_recognizer_lang
    with _paddle_lock:
        if _paddle_recognizer is None or _paddle_recognizer_lang != lang:
            _paddle_recognizer = recognizer_cls(use_angle_cls=True, lang=lang)
            _paddle_recognizer_lang = lang
        ocr = _paddle_recognizer

    if hasattr(image_input, "save"):  # PIL.Image（来自 PDF 转图）
        import tempfile
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        image_input.save(tmp.name)
        img_path = tmp.name
    else:
        img_path = str(image_input)
    try:
        result = ocr.ocr(img_path, cls=True)
    finally:
        if hasattr(image_input, "save"):
            try:
                os.unlink(img_path)
            except OSError:
                pass
    lines = []
    confs = []
    if result and result[0]:
        for line in result[0]:
            # line 形如 [bbox, ('文本', 置信度)]
            text = line[1][0]
            conf = line[1][1] if len(line[1]) > 1 else 1.0
            if min_conf and conf < min_conf:
                continue
            lines.append(text)
            confs.append(conf)
    # 返回文本与逐行置信度，供下游审计/质量评估（不再丢弃置信度信息）
    return "\n".join(lines), confs


def recognize_tesseract(pytesseract, Image, image_input, lang):
    """用 pytesseract 识别单张图片。lang 应为已解析的 tesseract 引擎代码。"""
    if hasattr(image_input, "save"):
        img = image_input
    else:
        img = Image.open(str(image_input))
    # R2 修复：lang 已在 build_recognizer 中经 resolve_lang 解析为正确引擎代码
    # （如 fr -> fra），不再在此做局部、不完整（仅 ch/chi_sim）的硬编码映射。
    return pytesseract.image_to_string(img, lang=lang), None


def recognize_mock(image_input, lang):
    """mock 模式：返回假文本，用于无依赖演示完整流程。"""
    name = getattr(image_input, "filename", str(image_input))
    return (f"[MOCK] 这是 {Path(str(name)).name} 的模拟识别文本（语言={lang}）。\n"
            f"欢迎使用 PaddleOCR 批量图文抽取工具。", None)


def resolve_lang(lang: str, backend: str) -> str:
    """将用户传入的 --lang 解析为对应后端的引擎代码。

    R2 修复：tesseract 后端此前只对 ch/chi_sim 做映射，其他语言
    （如 fr）会原样透传成 "fr"，而 tesseract 实际代码是 "fra"，导致识别失败。
    现统一经 SUPPORTED_LANGS 映射；未知语言原样透传（由后端决定，并在 main 中告警）。
    """
    info = SUPPORTED_LANGS.get(lang)
    if info is None:
        return lang
    return info.get(backend, info["paddle"])


def format_lang_list() -> str:
    """返回 --lang-list 的可读表格文本（纯函数，便于单测）。"""
    lines = ["支持的识别语言（--lang 取值）：",
             f"{'代码':<6}{'名称':<10}{'PaddleOCR':<12}{'tesseract'}"]
    for code, info in SUPPORTED_LANGS.items():
        lines.append(f"{code:<6}{info['name']:<10}{info['paddle']:<12}{info['tesseract']}")
    lines.append(f"\n默认语言：{DEFAULT_LANG}（{SUPPORTED_LANGS[DEFAULT_LANG]['name']}）")
    return "\n".join(lines)


def build_recognizer(args):
    """根据参数构建识别器，返回 (识别函数, 描述)。

    识别函数签名：fn(image_input) -> text
    """
    if args.mock:
        return (lambda img: recognize_mock(img, args.lang), "mock")

    if args.backend == "paddle":
        try:
            cls = load_paddle_backend()
        except RuntimeError as e:
            print(f"[降级] 无法加载 PaddleOCR 后端：\n{e}")
            print("[降级] 程序不会崩溃，请按提示安装后重试；或使用 --backend tesseract 或 --mock。")
            raise
        plang = resolve_lang(args.lang, "paddle")
        return (lambda img: recognize_paddle(cls, img, plang, args.min_conf), "paddle")

    if args.backend == "tesseract":
        try:
            pytesseract, Image = load_tesseract_backend()
        except RuntimeError as e:
            print(f"[错误] 无法加载 tesseract 后端：\n{e}")
            raise
        tlang = resolve_lang(args.lang, "tesseract")
        return (lambda img: recognize_tesseract(pytesseract, Image, img, tlang), "tesseract")


def _recognize_with_retry(recognizer, image_input, retries: int):
    """调用识别器，遇异常按 retries 次数重试（仅最后一次异常才向上抛）。

    R1 新能力：单页/单图识别偶发失败（内存抖动、后端瞬时异常）时自动重试，
    提升批处理对瞬时错误的韧性，避免一次抖动就丢掉整份结果。
    """
    last_exc = None
    for attempt in range(retries + 1):
        try:
            return recognizer(image_input)
        except Exception as e:  # 瞬态错误：重试
            last_exc = e
            if attempt < retries:
                continue
    raise last_exc


def process_file(file_path, recognizer, backend_name, retries: int = 0):
    """处理单个文件，返回结果字典。

    对 PDF 会先转图再逐页识别，合并文本。
    retries：单页/单图识别失败时的最大重试次数（R1 新能力，默认 0 不重试）。
    """
    ext = file_path.suffix.lower()
    start = time.time()
    error_msg = ""
    file_confs = []
    try:
        if ext in PDF_EXTS:
            images = pdf_to_images(file_path)
            if images is None:
                return {
                    "file": str(file_path),
                    "text": "",
                    "chars": 0,
                    "elapsed": round(time.time() - start, 3),
                    "status": "skipped_pdf",
                    "error": "",
                    "avg_conf": None,
                    "min_conf": None,
                }
            pages = []
            page_errors = 0
            for idx, img in enumerate(images, 1):
                # R1 韧性：单页识别失败按 --retries 重试，避免偶发异常中断
                try:
                    ptext, pconf = _recognize_with_retry(recognizer, img, retries)
                    pages.append(f"--- 第 {idx} 页 ---\n" + (ptext or ""))
                    if pconf:
                        file_confs.extend(pconf)
                except Exception as e:
                    page_errors += 1
                    pages.append(f"--- 第 {idx} 页 ---\n[第 {idx} 页识别失败：{e}]")
            text = "\n\n".join(pages)
            # R2 修复（隐性健壮性问题）：原实现任一页异常即把整份 PDF 标 error 并
            # 丢弃其余已成功页文本；现仅当「全部页都失败」才标 error，否则保留
            # 已识别页面、把失败页以占位说明呈现，最大化可用产出。
            if page_errors == len(images):
                status = "error"
                error_msg = f"{page_errors}/{len(images)} 页识别失败"
            else:
                status = "ok"
        else:
            text, fconf = _recognize_with_retry(recognizer, file_path, retries)
            if fconf:
                file_confs.extend(fconf)
            status = "ok"
    except Exception as e:
        text = ""
        status = "error"
        error_msg = str(e)
        print(f"[错误] 处理失败 {file_path}：{e}")
    # R2 防护（隐性崩溃风险）：识别器可能返回 None 文本（如退化的自定义识别
    # 函数、或某页返回 (None, None)），后续 `len(text)` 会抛 TypeError 直接
    # 中断整批处理。统一规整为字符串，再交给下方「空文本 -> empty」逻辑处理，
    # 避免单文件异常拖垮整个批处理流程。
    if text is None:
        text = ""
    # R2 修复（隐性可观测性缺陷）：原本空文本（识别不到任何字，但无异常）
    # 也会标记为 ok 并计入「成功」，污染 ok 计数与合并文件。现显式标记为
    # "empty"，既不计入成功也不进入合并，便于发现「识别失败但没报错」的情况。
    if not text and status == "ok":
        status = "empty"
    elapsed = round(time.time() - start, 3)
    avg_conf = round(sum(file_confs) / len(file_confs), 3) if file_confs else None
    min_conf_out = round(min(file_confs), 3) if file_confs else None
    return {
        "file": str(file_path),
        "text": text,
        "chars": len(text),
        "elapsed": elapsed,
        "status": status,
        "error": error_msg,
        "avg_conf": avg_conf,
        "min_conf": min_conf_out,
    }


def _unique_path(output_dir, name: str):
    """返回 output_dir 下不冲突的命名路径（R1 新能力 + R2 修复）。

    原实现直接用 stem + 扩展名，当不同子目录存在同名文件
    （如 a/x.png 与 b/x.png）时，二者都写 x.md，后者静默覆盖前者，
    造成结果丢失且无任何提示。这里在冲突时追加 _2 / _3 … 后缀，
    保证每个输入文件都有独立、不互相覆盖的输出。
    """
    out = Path(output_dir) / name
    if not out.exists():
        return out
    stem, ext = out.stem, out.suffix
    i = 2
    while True:
        cand = Path(output_dir) / f"{stem}_{i}{ext}"
        if not cand.exists():
            return cand
        i += 1


def write_markdown(result, output_dir):
    """写入单个文件的 .md 结果。"""
    src = Path(result["file"])
    out_path = _unique_path(output_dir, src.stem + ".md")
    meta = f"耗时：{result['elapsed']}s　状态：{result['status']}　字符数：{result['chars']}"
    if result.get("avg_conf") is not None:
        meta += f"　平均置信度：{result['avg_conf']}"
    if result.get("error"):
        meta += f"\n> 错误：{result['error']}"
    header = f"# {src.name}\n\n> {meta}\n\n"
    body = result["text"] if result["text"] else "（无识别结果）"
    out_path.write_text(header + body + "\n", encoding="utf-8")
    return out_path


def write_text(result, output_dir):
    """写入单个文件的 .txt 纯文本结果（--format txt）。"""
    src = Path(result["file"])
    out_path = _unique_path(output_dir, src.stem + ".txt")
    out_path.write_text(result["text"] + "\n", encoding="utf-8")
    return out_path


def write_combined(results, output_dir, fmt):
    """把所有成功结果按文件顺序拼接成单个合并文件，格式跟随 fmt。

    合并文件格式：
    - md    -> _combined.md（以文件名作小标题）
    - txt   -> _combined.txt（纯文本段）
    - json  -> _combined.json（结果数组）
    - jsonl -> _combined.jsonl（每行一条 JSON）
    - csv   -> _combined.csv（与 results.csv 同表头）

    R2 修复（一致性缺陷）：原实现无论请求何种 --format，合并文件永远写成
    _combined.md，导致 --format json/jsonl/csv 的流水线里混入一个多余的
    markdown 文件、且与用户指定的产物格式不一致。现严格按 fmt 输出对应格式。
    （R1 新能力：json/jsonl/csv 三种合并产物让下游无需再解析 markdown。）
    error/skipped 的结果不计入合并。无成功结果时跳过并提示，不写空文件。
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ok_results = [r for r in results if r.get("status") == "ok" and r.get("text")]
    if not ok_results:
        # R2 隐性问题：原本会写出一个只含换行的空 _combined 文件，
        # 误导用户「合并产物存在却有内容」。无成功结果时应跳过并提示。
        print("[提示] 没有可合并的成功结果，已跳过合并文件输出。")
        return None
    if fmt == "json":
        path = out_dir / "_combined.json"
        path.write_text(json.dumps(ok_results, ensure_ascii=False, indent=2), encoding="utf-8")
    elif fmt == "jsonl":
        path = out_dir / "_combined.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for r in ok_results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    elif fmt == "csv":
        path = out_dir / "_combined.csv"
        with open(path, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["file", "text", "chars", "elapsed", "status",
                               "error", "avg_conf", "min_conf"]
            )
            writer.writeheader()
            for r in ok_results:
                writer.writerow(r)
    elif fmt == "txt":
        blocks = [f"===== {Path(r['file']).name} =====\n{r['text']}" for r in ok_results]
        path = out_dir / "_combined.txt"
        path.write_text("\n\n".join(blocks) + "\n", encoding="utf-8")
    else:  # md
        blocks = [f"## {Path(r['file']).name}\n\n{r['text']}" for r in ok_results]
        path = out_dir / "_combined.md"
        path.write_text("\n\n".join(blocks) + "\n", encoding="utf-8")
    print(f"[完成] 已写出合并文件：{path}")
    return path


def write_outputs(results, output_dir, fmt, combine=False):
    """根据格式写出结果文件。combine=True 时额外写出合并文件。"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 始终写出合并的 json 与 csv（便于下游消费）
    json_path = output_dir / "results.json"
    csv_path = output_dir / "results.csv"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["file", "text", "chars", "elapsed", "status",
                           "error", "avg_conf", "min_conf"]
        )
        writer.writeheader()
        for r in results:
            writer.writerow(r)

    # 按用户指定格式写出逐文件结果
    if fmt == "md":
        for r in results:
            write_markdown(r, output_dir)
    elif fmt == "txt":
        for r in results:
            write_text(r, output_dir)
    elif fmt == "jsonl":
        # 每行一条 JSON，便于 grep/awk/jq 等行式工具与流式消费
        jsonl_path = output_dir / "results.jsonl"
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[完成] 已写出：{jsonl_path}")

    # 始终写出人类可读的汇总（新产物：一眼看清本次跑批结果）
    summary_path = output_dir / "summary.txt"
    ok = sum(1 for r in results if r["status"] == "ok")
    skipped = sum(1 for r in results if r["status"] == "skipped_pdf")
    errored = sum(1 for r in results if r["status"] == "error")
    total_chars = sum(len(r["text"]) for r in results)
    total_time = round(sum(r["elapsed"] for r in results), 3)
    summary_lines = [
        "PaddleOCR 批量图文抽取 · 运行汇总",
        f"时间：{time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"文件总数：{len(results)}",
        f"  成功(ok)：{ok}",
        f"  跳过 PDF(skipped_pdf)：{skipped}",
        f"  失败(error)：{errored}",
        f"识别字符总数：{total_chars}",
        f"耗时合计：{total_time}s",
        f"输出格式：{fmt}",
    ]
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    # R1 新需求：机器可读的运行统计 stats.json（便于流水线/下游消费，
    # 无需解析 summary.txt 文本）。与 summary.txt 互补：人读 vs 机读。
    conf_vals = [r["avg_conf"] for r in results
                 if isinstance(r.get("avg_conf"), (int, float))]
    stats = {
        "total": len(results),
        "ok": ok,
        "skipped_pdf": skipped,
        "error": errored,
        "total_chars": total_chars,
        "total_time": total_time,
        "avg_conf_overall": round(sum(conf_vals) / len(conf_vals), 3) if conf_vals else None,
        "format": fmt,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    stats_path = output_dir / "stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"[完成] 已写出运行统计：{stats_path}")

    # 可选：合并输出（把全部成功结果按序拼接成单个文件）
    if combine:
        write_combined(results, output_dir, fmt)

    print(f"[完成] 已写出：{json_path}")
    print(f"[完成] 已写出：{csv_path}")
    print(f"[完成] 已写出汇总：{summary_path}")
    if fmt == "md":
        print(f"[完成] 已写出逐文件 .md 到：{output_dir}")


def _output_exists(output_dir, file_path, fmt):
    """判断某文件的识别结果是否已存在（用于 --skip-existing 续跑）。

    md/txt 看对应逐文件产物；json/csv 以整批 results.json 作为完成标记。
    """
    out = Path(output_dir)
    stem = Path(file_path).stem
    if fmt in ("md", "txt"):
        return (out / f"{stem}.{fmt}").exists()
    return (out / "results.json").exists()


def apply_min_chars(results, min_chars: int) -> list:
    """按 --min-chars 过滤噪声结果。

    R1 新能力：识别字符数低于阈值的「成功」结果标记为 filtered，
    既不计入 ok 成功数、也不进入合并文件，避免极短噪声（如空白页、
    水印误识）混入产出。返回被过滤的结果列表，供调用方提示。
    """
    filtered = []
    if not min_chars or min_chars <= 0:
        return filtered
    for r in results:
        if r.get("status") == "ok" and (r.get("chars") or 0) < min_chars:
            r["status"] = "filtered"
            filtered.append(r)
    return filtered


def collect_low_conf(results, threshold: float) -> list:
    """收集平均置信度低于阈值的结果（仅对有置信度信息的文件生效）。

    供「低质量结果清单」导出：用户可据此快速定位需要重识别/人工核对的文件。
    无置信度信息（avg_conf 为 None，如 mock/tesseract 后端）不参与。
    """
    return [
        r for r in results
        if isinstance(r.get("avg_conf"), (int, float)) and r["avg_conf"] < threshold
    ]


def main(argv=None):
    args = parse_args(argv)
    # R1 新能力：--lang-list 仅列出支持的语言即退出，不要求 --input/--output
    if args.lang_list:
        print(format_lang_list())
        return 0
    # 缺少必要参数时给出明确错误（--input/--output 在 --lang-list 外为必填）
    if not args.input or not args.output:
        print("[错误] 缺少 --input 或 --output 参数。运行 --lang-list 可查看支持的语言。")
        return 1
    # R2 修复：未知语言代码显式告警（此前会静默透传、报错晦涩或静默用错语言）
    if args.lang not in SUPPORTED_LANGS:
        print(f"[提示] 语言代码 '{args.lang}' 不在官方支持列表，请运行 --lang-list 查看；"
              f"将按原样传给后端，可能因不支持而失败。")
    # 安全护栏：限制并行线程数，避免 --workers 过大耗尽系统资源
    if args.workers > MAX_WORKERS:
        print(f"[提示] --workers 超过安全上限 {MAX_WORKERS}，已自动限制为 {MAX_WORKERS}")
        args.workers = MAX_WORKERS
    if args.workers < 1:
        args.workers = 1
    print(f"=== PaddleOCR 批量图文抽取工具 ===")
    print(f"输入：{args.input}　输出：{args.output}")
    print(f"后端：{'mock' if args.mock else args.backend}　语言：{args.lang}　格式：{args.format}"
          f"　递归：{args.recursive}　并行：{args.workers}　最低置信度：{args.min_conf}")

    files, skipped = collect_all(args.input, args.recursive, include=args.include, exts=args.ext)
    if not files:
        # 隐性可观测性：原实现把 collect_all 返回的 skipped 直接丢弃，
        # 用户只能看到「未找到」却不知为何被排除；这里显式说明跳过情况。
        if skipped:
            example = skipped[0].name
            print(f"[提示] 未找到可处理的图片 / PDF 文件；另有 {len(skipped)} 个文件因类型不支持或隐藏被跳过（例如：{example}）。")
        else:
            print("[提示] 未找到可处理的图片 / PDF 文件。")
        # 仍创建输出目录，避免下游报错
        Path(args.output).mkdir(parents=True, exist_ok=True)
        write_outputs([], args.output, args.format)
        return 0

    # 断点续跑：跳过已有结果的文件（避免重复 OCR 浪费）
    if args.skip_existing:
        kept, already = [], []
        for f in files:
            if _output_exists(args.output, f, args.format):
                already.append(f)
            else:
                kept.append(f)
        files = kept
        if already:
            print(f"[提示] 跳过 {len(already)} 个已存在结果的文件（--skip-existing）")
        if not files:
            print("✅ 全部文件已有识别结果，无需重复处理。")
            return 0

    # 抽样 / 试跑：限制处理文件数（默认 0 不限制）。超出部分计入 skipped 保持透明。
    if args.max_files and args.max_files > 0 and len(files) > args.max_files:
        extra = files[args.max_files:]
        files = files[:args.max_files]
        skipped.extend(extra)
        print(f"[提示] --max-files {args.max_files}：仅处理前 {args.max_files} 个文件，"
              f"其余 {len(extra)} 个计入跳过。")

    # R1 新能力：按 --sort 指定的顺序处理（默认 name 保证确定性；size 大文件优先）
    files = sort_files(files, args.sort)

    if args.dry_run:
        # 预检模式：只统计待处理文件，不执行 OCR（新能力 + 可观测性）
        from collections import Counter

        ext_counter = Counter(f.suffix.lower() for f in files)
        print(f"[dry-run] 共 {len(files)} 个文件待处理：")
        for ext, c in ext_counter.most_common():
            print(f"  {ext}: {c}")
        if skipped:
            print(f"[dry-run] 跳过 {len(skipped)} 个不支持 / 隐藏的文件")
        return 0

    try:
        recognizer, backend_name = build_recognizer(args)
    except RuntimeError:
        return 1

    # 隐性问题：--min-conf / --low-conf-threshold 仅 paddle 后端生效，
    # 但 tesseract / mock 后端会静默忽略，用户误以为已过滤。这里给出明确提示。
    if (args.min_conf > 0 or args.low_conf_threshold >= 0) and backend_name != "paddle":
        print(f"[提示] 置信度相关选项（--min-conf / --low-conf-threshold）仅 paddle 后端生效，"
              f"当前后端为 {backend_name}，这些选项不会生效。")

    print(f"[信息] 使用后端：{backend_name}，共 {len(files)} 个文件")
    results = []
    if args.workers and args.workers > 1:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = [ex.submit(process_file, f, recognizer, backend_name, args.retries) for f in files]
            for i, fut in enumerate(futures, 1):
                res = fut.result()
                if not args.quiet:
                    print(f"[进度] 完成 {i}/{len(files)}：{res['file']} ({res['status']})")
                results.append(res)
    else:
        for i, f in enumerate(files, 1):
            if not args.quiet:
                print(f"[进度] 处理第 {i}/{len(files)} 个：{f}")
            res = process_file(f, recognizer, backend_name, args.retries)
            results.append(res)

    # R2 修复（隐性一致性 bug）：原实现先 write_outputs 再 apply_min_chars，
    # 导致逐文件 .md / results.json / summary.txt 已按状态 "ok" 落盘，随后 status
    # 被改为 "filtered"，最终 stats.json 与打印的成功数却更低——磁盘产物与统计不一致。
    # 现改为先标记 filtered，再统一写出，保证所有产物状态一致。
    if args.min_chars > 0:
        filtered = apply_min_chars(results, args.min_chars)
        if filtered:
            print(f"[提示] {len(filtered)} 个结果因识别字符数低于 {args.min_chars} 被标记为 filtered（不计入成功）")
    write_outputs(results, args.output, args.format, combine=args.combine)
    ok = sum(1 for r in results if r["status"] == "ok")
    errored = sum(1 for r in results if r["status"] == "error")
    print(f"[汇总] 成功 {ok}/{len(results)}，结果见：{args.output}")
    # R1 新能力：--fail-on-error 把「部分文件识别失败」升级为非零退出码，
    # 便于 CI / 流水线把静默的部分失败暴露为构建失败，而非默认吞掉（exit 0）。
    if errored and args.fail_on_error:
        print(f"[失败] 有 {errored} 个文件识别失败，因 --fail-on-error 退出码置为 1。", file=sys.stderr)
        return 1

    # 低置信度清单：把质量存疑的结果单独列出，便于重识别/人工核对
    if args.low_conf_threshold >= 0:
        low = collect_low_conf(results, args.low_conf_threshold)
        if low:
            low_path = Path(args.output) / "low_confidence.txt"
            lines = [f"低置信度结果（avg_conf < {args.low_conf_threshold}）共 {len(low)} 个：", ""]
            for r in low:
                lines.append(f"  {r['file']}  avg_conf={r['avg_conf']}")
            low_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            print(f"[完成] 已写出低置信度清单：{low_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

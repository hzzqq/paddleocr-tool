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

# 工具版本号（R1 新能力：--version 输出，便于脚本化识别与流水线断言）
TOOL_VERSION = "1.2.0"

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
    parser.add_argument("--list-formats", action="store_true",
                        help="列出所有支持的 --format 输出格式及其说明并退出")
    parser.add_argument("--list-backends", action="store_true",
                        help="列出本机已安装的 OCR 后端依赖（paddle/tesseract/pdf2image）并退出")
    parser.add_argument("--version", action="store_true",
                        help="打印工具版本号并退出（便于脚本化识别）")
    parser.add_argument("--json", action="store_true",
                        help="R1 新能力：信息类命令（--version/--lang-list/--list-formats/"
                             "--list-backends）输出机器可读的 JSON，便于脚本/流水线解析")
    parser.add_argument("--format", choices=["md", "json", "csv", "txt", "jsonl"],
                        default="md", help="输出格式，默认 md（txt 为逐文件纯文本，jsonl 为每行一条 JSON）")
    parser.add_argument("--recursive", action="store_true",
                        help="递归遍历子目录")
    parser.add_argument("--max-depth", type=int, default=0,
                        help="递归深度上限（仅 --recursive 生效，>=1；1=只取顶层，0=不限），避免大目录树下钻过深")
    parser.add_argument("--max-size", type=int, default=0,
                        help="跳过超过该体积（字节）的文件（0 表示不限制），避免超大扫描件/图片拖垮内存或后端")
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
    parser.add_argument("--exclude", default=None,
                        help="R1 新能力：文件名排除模式（逗号/换行分隔的多个子串或 glob），命中任一即跳过；"
                             "与 --include 互补：先 --include 收窄、再 --exclude 剔除指定文件")
    parser.add_argument("--ext", default=None,
                        help="扩展名过滤：只处理指定扩展名（逗号分隔，如 .png,.pdf），覆盖默认图片/PDF 白名单")
    parser.add_argument("--min-chars", type=int, default=0,
                        help="识别字符数低于该值的「成功」结果标记为 filtered（噪声过滤，不计入成功数/合并）")
    parser.add_argument("--normalize", action="store_true",
                        help="R1 新能力：规整识别文本——折叠行内连续空白、删除空行、去首尾空白，输出更利于下游消费")
    parser.add_argument("--max-files", type=int, default=0,
                        help="最多处理的文件数（0 表示不限制），便于对大目录做抽样 / 试跑")
    parser.add_argument("--sort", choices=["name", "size", "mtime"], default="name",
                        help="处理顺序：name=按文件名字典序（默认，确定性）；size=按体积降序（大文件优先，利于并行吞吐）；mtime=按修改时间降序（最新改动优先）")
    parser.add_argument("--fail-on-error", action="store_true",
                        help="R1 新能力：任一文件识别失败时返回非零退出码（默认仍返回 0），便于 CI / 流水线把「部分失败」升级为构建失败")
    parser.add_argument("--retries", type=int, default=0,
                        help="R1 新能力：单文件/单页识别失败时的重试次数（默认 0 不重试），提升对瞬时错误的韧性")
    parser.add_argument("--status-filter", default=None,
                        help="R1 新能力：只把指定状态的结果写出逐文件/合并产物（逗号分隔，如 'ok' 或 'ok,empty'）；"
                             "不影响 results.json/stats.json 的全量审计信息")
    parser.add_argument("--report", default=None,
                        help="R1 新能力：把运行报告（各状态计数 + 各状态文件清单 + 建议重跑清单）"
                             "以 JSON 写入该路径，便于流水线/人工快速定位需重跑的文件")
    return parser.parse_args(argv)


def _split_include(include) -> "list[str]":
    """把 --include 解析为「模式列表」（R1 新能力：支持多个逗号/换行分隔的模式）。

    单字符串按逗号 / 换行切分为多个模式（任一命中即保留）；已是列表则直接清洗。
    空值返回空列表（表示不过滤）。
    """
    if include is None:
        return []
    if isinstance(include, (list, tuple, set)):
        return [p for p in include if p and str(p).strip()]
    return [p.strip() for p in re.split(r"[,\n]", str(include)) if p.strip()]


def _name_matches(name: str, include) -> bool:
    """文件名是否匹配 --include（子串或 glob 任一命中即视为匹配）。

    R1 新能力：include 可为「模式列表」（多模式 OR）；单模式时退化为原行为。
    空列表 / None 视为不过滤。
    """
    if not include:
        return True
    if isinstance(include, (list, tuple, set)):
        return any(_name_matches(name, p) for p in include)
    return include in name or fnmatch.fnmatch(name, include)


def collect_files(input_path, recursive, include=None, exts=None, max_depth=None, max_size=None, exclude=None):
    """收集需要处理的文件列表（图片 + PDF）。

    返回 (文件列表, 跳过列表)。跳过列表包含不支持类型 / 隐藏文件 / 隐藏目录 /
    未命中 --include / 命中 --exclude / 未命中 --ext / 超过 --max-size，便于调用方透明提示。

    exts：可选扩展名白名单（小写，含点），提供时覆盖默认 IMAGE/PDF 白名单，
    仅处理落在该集合内的文件（R1 新能力，与 --include 的「文件名/子串」维度互补）。
    max_depth：可选递归深度上限（仅 recursive 时生效，>=1）；限制相对输入根
    目录的目录层级，1 表示只取顶层。None / <=0 表示不限（R1 新能力）。
    max_size：可选体积上限（字节，>0 生效）；超过该大小的文件被跳过，避免
    超大扫描件 / 图片把内存 / 后端拖垮（R1 新能力 + R2 隐性健壮性护栏）。
    exclude：可选排除模式列表（R1 新能力），文件名命中任一即跳过，与 --include
    互补——先 --include 收窄、再 --exclude 剔除指定文件（如临时件/封面/水印）。
    """
    p = Path(input_path)
    files = []
    skipped = []
    if exts is not None:
        # R2 修复（隐性可用性问题）：--ext 传入 "png"（无前导点）时，
        # 此前会原样与文件后缀 ".png" 比较，永远不相等 -> 全部文件被静默跳过、
        # 结果为空且无任何提示。现统一规整为带前导点的小写扩展名。
        norm = set()
        for e in exts.split(","):
            e = e.strip().lower()
            if not e:
                continue
            if not e.startswith("."):
                e = "." + e
            norm.add(e)
        exts = norm
    allowed_exts = exts if exts is not None else (IMAGE_EXTS | PDF_EXTS)
    include_list = _split_include(include)  # R1：支持多模式 OR
    exclude_list = _split_include(exclude)  # R1：支持多模式 OR
    if p.is_file():
        ext = p.suffix.lower()
        if ext in allowed_exts and _name_matches(p.name, include_list):
            if exclude_list and _name_matches(p.name, exclude_list):  # R1：命中 --exclude 跳过
                skipped.append(p)
                return files, skipped
            if max_size and max_size > 0 and _safe_size(p) > max_size:
                skipped.append(p)  # 超过体积上限
            else:
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
            if _name_matches(f.name, include_list):
                if exclude_list and _name_matches(f.name, exclude_list):  # R1：命中 --exclude 跳过
                    skipped.append(f)
                    continue
                if max_size and max_size > 0 and _safe_size(f) > max_size:
                    skipped.append(f)  # 超过体积上限
                    continue
                files.append(f)
            else:
                skipped.append(f)  # 未命中 --include
        else:
            skipped.append(f)  # 其他不支持类型
    # R1 新能力：递归深度上限（仅 recursive 时生效，>=1）。
    # 限制相对输入根目录的层级，1 表示只取顶层，避免 --recursive 在大目录树里
    # 无差别下钻过深、把无关子目录也卷进批处理。None / <=0 表示不限。
    if recursive and max_depth and max_depth > 0:
        files = [f for f in files if len(f.relative_to(p).parts) <= max_depth]
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
    if mode == "mtime":
        # R1 新能力：按文件修改时间降序（最新改动优先），适合「先处理
        # 刚扫描/刚下载的图片」这类场景（与 size 的「大文件优先」互补）。
        return sorted(files, key=lambda p: _safe_mtime(p), reverse=True)
    # 默认按名称（确定性）
    return sorted(files, key=lambda p: str(p))


def _safe_size(p) -> int:
    try:
        return p.stat().st_size
    except OSError:
        return 0


def _safe_mtime(p) -> float:
    try:
        return p.stat().st_mtime
    except OSError:
        return 0.0


def collect_all(input_spec, recursive, include=None, exts=None, max_depth=None, max_size=None, exclude=None):
    """支持 `--input` 传入多个路径（逗号 / 换行分隔），聚合去重。

    单路径时等价于 collect_files；多路径用于一次性批量处理若干分散文件 / 目录。
    max_depth / max_size / exclude 透传给 collect_files（递归深度上限 / 体积上限 / 排除模式）。

    R1 新能力：各路径片段支持 glob 模式（如 `dir/*.png`、`imgs/**/*.jpg`），
    自动展开为匹配文件逐个处理，省去用户先 `ls` 再粘贴文件列表。
    """
    import glob as _glob

    files = []
    skipped = []
    for part in re.split(r"[,\n]", input_spec or ""):
        part = part.strip()
        if not part:
            continue
        # R1：glob 模式展开（含 * ? [ ]）。此前这类输入会被当字面路径，
        # 命中「路径不存在」分支静默失败（R2 隐性可用性缺陷）。
        if any(ch in part for ch in "*?["):
            # R2 修复（隐性一致性缺陷）：原实现硬编码 recursive=True，
            # 导致 '**' 这类递归通配即便用户未传 --recursive 也会下钻子目录，
            # 与 --recursive 开关语义矛盾。现透传 collect_all 收到的 recursive
            # 参数，使 glob 展开与目录遍历的递归行为保持一致。
            matched = _glob.glob(part, recursive=recursive)
            if matched:
                for m in matched:
                    f, s = collect_files(
                        m, recursive, include=include, exts=exts,
                        max_depth=max_depth, max_size=max_size, exclude=exclude,
                    )
                    files.extend(f)
                    skipped.extend(s)
                continue
            else:
                # glob 无匹配：视为「类型不支持」计入跳过，保持透明
                skipped.append(Path(part))
                continue
        f, s = collect_files(
            part, recursive, include=include, exts=exts,
            max_depth=max_depth, max_size=max_size, exclude=exclude,
        )
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


# 各输出格式的能力说明（供 --list-formats 展示与单测）
FORMAT_INFO = {
    "md": "逐文件 Markdown + 合并 _combined.md（人名/标题小标题，适合人读）",
    "txt": "逐文件纯文本 .txt + 合并 _combined.txt",
    "json": "整批 results.json + 合并 _combined.json（机器可读数组）",
    "jsonl": "results.jsonl 每行一条 JSON（便于 grep/awk/jq 与流式消费）",
    "csv": "results.csv 表头化 + 合并 _combined.csv（Excel/表格友好）",
}
DEFAULT_FORMAT = "md"


def format_format_list() -> str:
    """返回 --list-formats 的可读表格文本（纯函数，便于单测）。"""
    lines = ["支持的输出格式（--format 取值）：",
             f"{'格式':<8}{'说明'}"]
    for fmt, desc in FORMAT_INFO.items():
        lines.append(f"{fmt:<8}{desc}")
    lines.append(f"\n默认格式：{DEFAULT_FORMAT}（{FORMAT_INFO[DEFAULT_FORMAT]}）")
    return "\n".join(lines)


def available_backends() -> dict:
    """探测各 OCR 后端依赖是否可用，返回 {backend: bool}。

    R1 新能力：用户离线/未安装时，先跑 `--list-backends` 即可知道自己
    能用哪些后端（例如只有 tesseract、没有 paddle），无需等到真正跑批
    才因缺依赖而报错。探测采用「尝试导入」而非「执行识别」，避免误触发
    重模型下载或实际拉起后端。
    """
    avail = {}
    try:
        import paddleocr  # noqa: F401
        avail["paddle"] = True
    except Exception:
        avail["paddle"] = False
    try:
        import pytesseract  # noqa: F401
        from PIL import Image  # noqa: F401
        avail["tesseract"] = True
    except Exception:
        avail["tesseract"] = False
    try:
        import pdf2image  # noqa: F401
        avail["pdf2image"] = True
    except Exception:
        avail["pdf2image"] = False
    return avail


def format_backends_list() -> str:
    """返回 --list-backends 的可读文本（纯函数，便于单测）。"""
    avail = available_backends()
    lines = ["已安装的后端依赖："]
    lines.append(
        f"  paddle    : {'已安装' if avail['paddle'] else '未安装（pip install paddleocr paddlepaddle）'}"
    )
    lines.append(
        f"  tesseract : {'已安装' if avail['tesseract'] else '未安装（pip install pytesseract + Tesseract 引擎）'}"
    )
    lines.append(
        f"  pdf2image : {'已安装' if avail['pdf2image'] else '未安装（pip install pdf2image + poppler）'}"
    )
    return "\n".join(lines)


def format_version() -> str:
    """返回工具版本的可读文本（R1 新能力：--version 的纯函数，便于单测）。"""
    return f"paddleocr-tool {TOOL_VERSION}"


def version_info() -> dict:
    """返回工具版本信息的结构化字典（R1 新能力：--version --json 的纯函数）。"""
    import platform

    return {
        "version": TOOL_VERSION,
        "python": platform.python_version(),
        "backends": available_backends(),
    }


def lang_list_data() -> list:
    """返回支持语言的结构化列表（R1 新能力：--lang-list --json 的纯函数）。"""
    return [
        {"code": c, "name": i["name"], "paddle": i["paddle"], "tesseract": i["tesseract"]}
        for c, i in SUPPORTED_LANGS.items()
    ]


def format_list_data() -> dict:
    """返回支持输出格式的结构化字典（R1 新能力：--list-formats --json 的纯函数）。"""
    return dict(FORMAT_INFO)


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


def process_file(file_path, recognizer, backend_name, retries: int = 0, normalize: bool = False):
    """处理单个文件，返回结果字典。

    对 PDF 会先转图再逐页识别，合并文本。
    retries：单页/单图识别失败时的最大重试次数（R1 新能力，默认 0 不重试）。
    normalize：是否对识别文本做空白规整（R1 新能力，--normalize）。
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
                    if normalize:
                        ptext = normalize_text(ptext or "")
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
            if normalize:
                text = normalize_text(text or "")
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
    # R2 修复（隐性噪声）：仅含空白的文本（如 "   \n  "）此前因字符串非空，
    # 被误判为 ok 并计入成功数（chars>0），污染统计；现按「去除首尾空白后为空」
    # 判定 empty，与真实空结果一致处理。
    if not text.strip() and status == "ok":
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


def normalize_text(text: str) -> str:
    """规整识别文本：折叠行内连续空白、删除空行、去首尾空白。

    R1 新能力：OCR / 识别输出常含多余空格、Tab 与空行，折叠后更利于下游
    消费与阅读（如喂给 LLM、入库、比对），也避免「空行噪声」污染结果。
    不改变文字内容，仅清理空白排版。
    """
    if not text:
        return ""
    out = []
    for line in text.splitlines():
        s = " ".join(line.split())  # 折叠行内连续空白 + 去首尾空白
        if s:
            out.append(s)
    return "\n".join(out)


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


def write_combined(results, output_dir, fmt, status_filter=None):
    """把所有成功结果按文件顺序拼接成单个合并文件，格式跟随 fmt。

    status_filter：与 write_outputs 一致的状态过滤；为空 / None 时按默认
    （status==ok 且有文本）合并。提供时仅合并指定状态的结果。

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
    allowed = None
    if status_filter:
        allowed = {x.strip() for x in status_filter if x and x.strip()}
    ok_results = [
        r for r in results
        if r.get("text")
        and (
            # 未指定 status_filter：按默认「成功且有文本」合并
            (allowed is None and r.get("status") == "ok")
            # 指定了 status_filter：严格按用户选择的状态集合合并
            # （R2 修复：原实现硬编码 status=="ok"，使 status_filter 仅能在
            #  ok 内收窄、永远无法合并 empty/filtered 等状态，与 write_outputs
            #  逐文件产物的 status_filter 行为不一致且形同虚设）
            or (allowed is not None and r.get("status") in allowed)
        )
    ]
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


def write_outputs(results, output_dir, fmt, combine=False, status_filter=None):
    """根据格式写出结果文件。combine=True 时额外写出合并文件。

    status_filter：可选状态集合（str 列表，如 ["ok"]）；仅这些状态的结果
    会写出逐文件产物（md/txt/jsonl）与合并文件，便于「只导出成功结果 / 把
    empty·filtered·error 留待重跑」。results.json / results.csv / summary.txt /
    stats.json 始终写入全量结果，保证审计信息不被筛选影响。
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # R1 新能力：逐文件产物与合并产物按状态过滤；审计类产物仍用全量结果
    per_file_results = filter_results_by_status(results, status_filter)

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

    # 按用户指定格式写出逐文件结果（受 status_filter 约束）
    if fmt == "md":
        for r in per_file_results:
            write_markdown(r, output_dir)
    elif fmt == "txt":
        for r in per_file_results:
            write_text(r, output_dir)
    elif fmt == "jsonl":
        # 每行一条 JSON，便于 grep/awk/jq 等行式工具与流式消费
        jsonl_path = output_dir / "results.jsonl"
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for r in per_file_results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[完成] 已写出：{jsonl_path}")

    # 始终写出人类可读的汇总（新产物：一眼看清本次跑批结果）
    summary_path = output_dir / "summary.txt"
    s = summarize_statuses(results)
    ok = s["ok"]
    skipped = s["skipped_pdf"]
    errored = s["error"]
    total_chars = sum(len(r["text"]) for r in results)
    total_time = round(sum(r["elapsed"] for r in results), 3)
    summary_lines = [
        "PaddleOCR 批量图文抽取 · 运行汇总",
        f"时间：{time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"文件总数：{len(results)}",
        f"  成功(ok)：{ok}",
        f"  空白(empty)：{s['empty']}",
        f"  噪声(filtered)：{s['filtered']}",
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
        "total": s["total"],
        "ok": ok,
        "empty": s["empty"],
        "filtered": s["filtered"],
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
        write_combined(per_file_results, output_dir, fmt, status_filter)

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


def summarize_statuses(results: list) -> dict:
    """汇总各状态计数，供 summary.txt / stats.json 复用（R1 新能力 + R2 修复）。

    原实现只在 summary/stats 里散落统计 ok / skipped_pdf / error 三类状态，
    遗漏了 empty（识别不到字但无异常）与 filtered（低于 --min-chars 被过滤）
    两类——这两类恰恰是「噪声比例」「待重跑清单」的关键信号，漏统计会误导
    用户对本次跑批质量的判断（隐性可观测性缺口）。这里集中统计全部五类状态，
    作为单一事实来源，避免多处计数口径不一致。
    """
    from collections import Counter

    c = Counter(r.get("status") for r in results)
    return {
        "total": len(results),
        "ok": c.get("ok", 0),
        "empty": c.get("empty", 0),
        "filtered": c.get("filtered", 0),
        "error": c.get("error", 0),
        "skipped_pdf": c.get("skipped_pdf", 0),
    }


def filter_results_by_status(results: list, statuses) -> list:
    """按状态过滤结果（R1 新能力：--status-filter 的纯函数实现）。

    仅保留 status 在 statuses 集合内的结果；statuses 为空 / None / 空列表时
    原样返回全部。用于「只导出成功结果」「把 empty/filtered 留待重跑」等场景，
    不影响 results.json / stats.json 中全量审计信息（那些仍保留所有结果）。
    """
    if not statuses:
        return list(results)
    allowed = {s.strip() for s in statuses if s and s.strip()}
    if not allowed:
        return list(results)
    return [r for r in results if r.get("status") in allowed]


def build_run_report(results: list) -> dict:
    """构建机器可读的运行报告（R1 新能力 + R2 修复）。

    返回：
      {
        "counts": <summarize_statuses 结果>,
        "by_status": {status: [文件路径...]},   # 各状态对应的具体文件清单
        "rerun_candidates": [文件路径...],        # 建议重跑的 empty/filtered/error 文件
      }

    R2 隐性可观测性缺口：原 summary.txt / stats.json 只给「计数」，当一批里有
    若干 empty / filtered / error 时，用户必须自己 grep results.json 才能知道
    到底是哪些文件出了问题、该重跑哪些——毫无头绪。这里把具体文件路径按状态
    分组并给出「建议重跑清单」，配合 --report 直接落盘，省去手动排查。
    """
    by_status: dict = {}
    for r in results:
        st = r.get("status", "unknown")
        by_status.setdefault(st, []).append(r.get("file", ""))
    rerun = [
        r.get("file", "") for r in results
        if r.get("status") in ("empty", "filtered", "error")
    ]
    return {
        "counts": summarize_statuses(results),
        "by_status": by_status,
        "rerun_candidates": rerun,
    }


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
        if args.json:
            print(json.dumps(lang_list_data(), ensure_ascii=False, indent=2))
        else:
            print(format_lang_list())
        return 0
    # R1 新能力：--list-formats 仅列出支持的 --format 输出格式即退出
    if args.list_formats:
        if args.json:
            print(json.dumps(format_list_data(), ensure_ascii=False, indent=2))
        else:
            print(format_format_list())
        return 0
    # R1 新能力：--list-backends 仅列出本机已安装的后端依赖即退出，
    # 不需 --input/--output（R2 修复：信息类标志不应要求这两个必填参数，
    # 否则 `ocr --list-backends` 会先被「缺少 --input/--output」拦截而误报）。
    if args.list_backends:
        if args.json:
            print(json.dumps(available_backends(), ensure_ascii=False, indent=2))
        else:
            print(format_backends_list())
        return 0
    # R1 新能力：--version 仅打印工具版本号即退出，不要求 --input/--output
    if args.version:
        if args.json:
            print(json.dumps(version_info(), ensure_ascii=False, indent=2))
        else:
            print(format_version())
        return 0
    # 缺少必要参数时给出明确错误。
    # R2 修复（隐性 UX 缺陷）：--dry-run 仅做预检、不需要 --output，但原实现
    # 把「缺少 --output」放在 --dry-run 处理之前，导致 `ocr --dry-run --input dir`
    # 被「缺少 --output」误拦截而报错。现改为：--input 始终必填；--output 仅在
    # 非 dry-run 时必填（R1 的 --list-* 信息标志已在更靠前提前返回）。
    if not args.input:
        print("[错误] 缺少 --input 参数。运行 --lang-list 可查看支持的语言。")
        return 1
    if not args.dry_run and not args.output:
        print("[错误] 缺少 --output 参数（--dry-run 预检模式可省略 --output）。")
        return 1
    # R2 修复：未知语言代码显式告警（此前会静默透传、报错晦涩或静默用错语言）
    if args.lang not in SUPPORTED_LANGS:
        print(f"[提示] 语言代码 '{args.lang}' 不在官方支持列表，请运行 --lang-list 查看；"
              f"将按原样传给后端，可能因不支持而失败。")
    # R1/R2 输入护栏：--min-conf / --low-conf-threshold 是 0~1 的置信度「比例」，
    # 用户常误用 0~100 的「百分比」（如 --min-conf 60），导致所有识别行被过滤、
    # 结果为空且无任何提示（隐性可观测性缺口）。现越界时钳制到 [0,1] 并告警，
    # 避免「静默全丢」；合法区间内的取值不动。
    if args.min_conf < 0 or args.min_conf > 1:
        clamped = max(0.0, min(1.0, args.min_conf))
        print(f"[提示] --min-conf 应在 0~1 之间（置信度比例，非百分比），"
              f"已钳制为 {clamped}（原值 {args.min_conf}）")
        args.min_conf = clamped
    if args.low_conf_threshold > 1:
        print(f"[提示] --low-conf-threshold 应在 0~1 之间（置信度比例，非百分比），"
              f"已钳制为 1.0（原值 {args.low_conf_threshold}）")
        args.low_conf_threshold = 1.0
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

    files, skipped = collect_all(
        args.input, args.recursive, include=args.include, exts=args.ext,
        max_depth=args.max_depth, max_size=args.max_size, exclude=args.exclude,
    )
    if not files:
        # 隐性可观测性：原实现把 collect_all 返回的 skipped 直接丢弃，
        # 用户只能看到「未找到」却不知为何被排除；这里显式说明跳过情况。
        if skipped:
            example = skipped[0].name
            print(f"[提示] 未找到可处理的图片 / PDF 文件；另有 {len(skipped)} 个文件因类型不支持、隐藏或超过体积上限被跳过（例如：{example}）。")
        else:
            print("[提示] 未找到可处理的图片 / PDF 文件。")
        # 仍创建输出目录，避免下游报错（dry-run 无 --output 时跳过）
        if args.output:
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

    # R2 修复（隐性采样缺陷）：--max-files 截断必须发生在 --sort 之后，
    # 否则「size/mtime 排序优先处理」的抽样意图被破——先按名称顺序截断 N 个、
    # 再对这 N 个排序，导致「大文件优先 / 最新优先」形同虚设。现先排序、再截断。
    # R1 新能力：按 --sort 指定的顺序处理（默认 name 保证确定性；
    # size=大文件优先利于并行吞吐；mtime=最新改动优先）。
    files = sort_files(files, args.sort)

    # 抽样 / 试跑：限制处理文件数（默认 0 不限制）。超出部分计入 skipped 保持透明。
    if args.max_files and args.max_files > 0 and len(files) > args.max_files:
        extra = files[args.max_files:]
        files = files[:args.max_files]
        skipped.extend(extra)
        print(f"[提示] --max-files {args.max_files}：仅处理前 {args.max_files} 个文件，"
              f"其余 {len(extra)} 个计入跳过。")

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
    if (args.min_conf > 0 or args.low_conf_threshold > 0) and backend_name != "paddle":
        print(f"[提示] 置信度相关选项（--min-conf / --low-conf-threshold）仅 paddle 后端生效，"
              f"当前后端为 {backend_name}，这些选项不会生效。")

    print(f"[信息] 使用后端：{backend_name}，共 {len(files)} 个文件")
    results = []
    if args.workers and args.workers > 1:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = [ex.submit(process_file, f, recognizer, backend_name, args.retries, args.normalize) for f in files]
            for i, fut in enumerate(futures, 1):
                res = fut.result()
                if not args.quiet:
                    print(f"[进度] 完成 {i}/{len(files)}：{res['file']} ({res['status']})")
                results.append(res)
    else:
        for i, f in enumerate(files, 1):
            if not args.quiet:
                print(f"[进度] 处理第 {i}/{len(files)} 个：{f}")
            res = process_file(f, recognizer, backend_name, args.retries, args.normalize)
            results.append(res)

    # R2 修复（隐性一致性 bug）：原实现先 write_outputs 再 apply_min_chars，
    # 导致逐文件 .md / results.json / summary.txt 已按状态 "ok" 落盘，随后 status
    # 被改为 "filtered"，最终 stats.json 与打印的成功数却更低——磁盘产物与统计不一致。
    # 现改为先标记 filtered，再统一写出，保证所有产物状态一致。
    if args.min_chars > 0:
        filtered = apply_min_chars(results, args.min_chars)
        if filtered:
            print(f"[提示] {len(filtered)} 个结果因识别字符数低于 {args.min_chars} 被标记为 filtered（不计入成功）")
    # R1 新能力：--status-filter 只把指定状态写出逐文件/合并产物
    status_filter = None
    if args.status_filter:
        status_filter = [s.strip() for s in args.status_filter.split(",") if s.strip()]
        if not status_filter:
            status_filter = None
        else:
            print(f"[提示] 仅写出状态为 {','.join(status_filter)} 的逐文件/合并产物（审计文件仍含全部结果）")
    write_outputs(results, args.output, args.format, combine=args.combine, status_filter=status_filter)
    # R1 新能力：写出机器可读运行报告（含各状态文件清单 + 建议重跑清单），
    # 弥补 summary 只给计数、不给具体文件的隐性可观测性缺口。
    if args.report:
        report = build_run_report(results)
        Path(args.report).write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"[完成] 已写出运行报告：{args.report}")
    ok = sum(1 for r in results if r["status"] == "ok")
    errored = sum(1 for r in results if r["status"] == "error")
    print(f"[汇总] 成功 {ok}/{len(results)}，结果见：{args.output}")
    # R1 新能力：--fail-on-error 把「部分文件识别失败」升级为非零退出码，
    # 便于 CI / 流水线把静默的部分失败暴露为构建失败，而非默认吞掉（exit 0）。
    if errored and args.fail_on_error:
        print(f"[失败] 有 {errored} 个文件识别失败，因 --fail-on-error 退出码置为 1。", file=sys.stderr)
        return 1

    # 低置信度清单：把质量存疑的结果单独列出，便于重识别/人工核对
    # R2 一致性修复：low_conf_threshold <= 0 视为「关闭」（与 -1.0 默认语义一致，
    # 且 avg_conf 恒 >= 0，阈值 0 永远收集不到任何文件）；此前用 `>= 0` 会在
    # 阈值=0 时仍无意义进入收集分支，并在非 paddle 后端触发虚假的「置信度选项
    # 不生效」告警。现统一用 `> 0` 判定启用。
    if args.low_conf_threshold > 0:
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

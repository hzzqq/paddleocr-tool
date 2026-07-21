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
import json
import os
import sys
import threading
import time
from pathlib import Path

# 支持的图片扩展名（小写）
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
# 支持的 PDF 扩展名
PDF_EXTS = {".pdf"}

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
    parser.add_argument("--input", required=True,
                        help="输入目录或单个文件（图片或 PDF）")
    parser.add_argument("--output", required=True,
                        help="输出目录")
    parser.add_argument("--lang", default="ch",
                        help="识别语言，默认 ch（中文）。tesseract 可用 chi_sim 等")
    parser.add_argument("--format", choices=["md", "json", "csv"],
                        default="md", help="输出格式，默认 md")
    parser.add_argument("--recursive", action="store_true",
                        help="递归遍历子目录")
    parser.add_argument("--backend", choices=["paddle", "tesseract"],
                        default="paddle", help="OCR 后端，默认 paddle")
    parser.add_argument("--mock", action="store_true",
                        help="使用 mock 模式返回假文本（无需任何 OCR 依赖，用于演示流程）")
    parser.add_argument("--workers", type=int, default=1,
                        help="并行识别的线程数（默认 1，串行）；多图时显著提速")
    parser.add_argument("--min-conf", type=float, default=0.0,
                        help="最低置信度阈值（仅 paddle 后端生效，0~1，低于则丢弃该行）")
    return parser.parse_args(argv)


def collect_files(input_path, recursive):
    """收集需要处理的文件列表（图片 + PDF）。

    返回 (文件列表, 跳过的 PDF 列表)。
    """
    p = Path(input_path)
    files = []
    skipped_pdf = []
    if p.is_file():
        ext = p.suffix.lower()
        if ext in IMAGE_EXTS or ext in PDF_EXTS:
            files.append(p)
        else:
            print(f"[跳过] 不支持的文件类型：{p}")
        return files, skipped_pdf

    if not p.is_dir():
        print(f"[错误] 输入路径不存在：{input_path}")
        return files, skipped_pdf

    # 目录：按扩展名收集
    pattern = "**/*" if recursive else "*"
    for f in sorted(p.glob(pattern)):
        if not f.is_file():
            continue
        if f.name.startswith("."):
            continue  # 跳过隐藏文件（如 .DS_Store）
        ext = f.suffix.lower()
        if ext in IMAGE_EXTS:
            files.append(f)
        elif ext in PDF_EXTS:
            files.append(f)
        # 其他类型忽略
    return files, skipped_pdf


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
_paddle_lock = threading.Lock()
_paddle_recognizer = None


def recognize_paddle(recognizer_cls, image_input, lang, min_conf=0.0):
    """用 PaddleOCR 识别单张图片（image_input 可为路径或 PIL.Image）。

    recognizer_cls：PaddleOCR 类（由 build_recognizer 传入）。
    内部以「懒加载单例 + 锁」复用同一个识别器实例，既消除逐图重建的
    性能悬崖，又用锁保证 ThreadPoolExecutor 并行时不会并发踩同一实例。
    min_conf：最低置信度阈值（0~1），低于该值的识别行将被丢弃（仅 paddle 生效）。
    """
    global _paddle_recognizer
    with _paddle_lock:
        if _paddle_recognizer is None:
            _paddle_recognizer = recognizer_cls(use_angle_cls=True, lang=lang)
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
    if result and result[0]:
        for line in result[0]:
            # line 形如 [bbox, ('文本', 置信度)]
            text = line[1][0]
            conf = line[1][1] if len(line[1]) > 1 else 1.0
            if min_conf and conf < min_conf:
                continue
            lines.append(text)
    return "\n".join(lines)


def recognize_tesseract(pytesseract, Image, image_input, lang):
    """用 pytesseract 识别单张图片。"""
    if hasattr(image_input, "save"):
        img = image_input
    else:
        img = Image.open(str(image_input))
    # tesseract 语言映射：ch -> chi_sim
    tlang = "chi_sim+eng" if lang in ("ch", "chi_sim") else lang
    return pytesseract.image_to_string(img, lang=tlang)


def recognize_mock(image_input, lang):
    """mock 模式：返回假文本，用于无依赖演示完整流程。"""
    name = getattr(image_input, "filename", str(image_input))
    return f"[MOCK] 这是 {Path(str(name)).name} 的模拟识别文本（语言={lang}）。\n欢迎使用 PaddleOCR 批量图文抽取工具。"


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
        return (lambda img: recognize_paddle(cls, img, args.lang, args.min_conf), "paddle")

    if args.backend == "tesseract":
        try:
            pytesseract, Image = load_tesseract_backend()
        except RuntimeError as e:
            print(f"[错误] 无法加载 tesseract 后端：\n{e}")
            raise
        return (lambda img: recognize_tesseract(pytesseract, Image, img, args.lang), "tesseract")


def process_file(file_path, recognizer, backend_name):
    """处理单个文件，返回结果字典。

    对 PDF 会先转图再逐页识别，合并文本。
    """
    ext = file_path.suffix.lower()
    start = time.time()
    try:
        if ext in PDF_EXTS:
            images = pdf_to_images(file_path)
            if images is None:
                return {
                    "file": str(file_path),
                    "text": "",
                    "elapsed": round(time.time() - start, 3),
                    "status": "skipped_pdf",
                }
            pages = []
            for idx, img in enumerate(images, 1):
                pages.append(f"--- 第 {idx} 页 ---\n" + recognizer(img))
            text = "\n\n".join(pages)
        else:
            text = recognizer(file_path)
        status = "ok"
    except Exception as e:
        text = ""
        status = "error"
        print(f"[错误] 处理失败 {file_path}：{e}")
    elapsed = round(time.time() - start, 3)
    return {
        "file": str(file_path),
        "text": text,
        "elapsed": elapsed,
        "status": status,
    }


def write_markdown(result, output_dir):
    """写入单个文件的 .md 结果。"""
    src = Path(result["file"])
    out_name = src.stem + ".md"
    out_path = Path(output_dir) / out_name
    header = f"# {src.name}\n\n> 耗时：{result['elapsed']}s　状态：{result['status']}\n\n"
    body = result["text"] if result["text"] else "（无识别结果）"
    out_path.write_text(header + body + "\n", encoding="utf-8")
    return out_path


def write_outputs(results, output_dir, fmt):
    """根据格式写出结果文件。"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 始终写出合并的 json 与 csv（便于下游消费）
    json_path = output_dir / "results.json"
    csv_path = output_dir / "results.csv"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "text", "elapsed", "status"])
        writer.writeheader()
        for r in results:
            writer.writerow(r)

    # 按用户指定格式写出逐文件 md
    if fmt == "md":
        for r in results:
            write_markdown(r, output_dir)

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

    print(f"[完成] 已写出：{json_path}")
    print(f"[完成] 已写出：{csv_path}")
    print(f"[完成] 已写出汇总：{summary_path}")
    if fmt == "md":
        print(f"[完成] 已写出逐文件 .md 到：{output_dir}")


def main(argv=None):
    args = parse_args(argv)
    print(f"=== PaddleOCR 批量图文抽取工具 ===")
    print(f"输入：{args.input}　输出：{args.output}")
    print(f"后端：{'mock' if args.mock else args.backend}　语言：{args.lang}　格式：{args.format}"
          f"　递归：{args.recursive}　并行：{args.workers}　最低置信度：{args.min_conf}")

    files, _ = collect_files(args.input, args.recursive)
    if not files:
        print("[提示] 未找到可处理的图片 / PDF 文件。")
        # 仍创建输出目录，避免下游报错
        Path(args.output).mkdir(parents=True, exist_ok=True)
        write_outputs([], args.output, args.format)
        return 0

    try:
        recognizer, backend_name = build_recognizer(args)
    except RuntimeError:
        return 1

    print(f"[信息] 使用后端：{backend_name}，共 {len(files)} 个文件")
    results = []
    if args.workers and args.workers > 1:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = [ex.submit(process_file, f, recognizer, backend_name) for f in files]
            for i, fut in enumerate(futures, 1):
                res = fut.result()
                print(f"[进度] 完成 {i}/{len(files)}：{res['file']} ({res['status']})")
                results.append(res)
    else:
        for i, f in enumerate(files, 1):
            print(f"[进度] 处理第 {i}/{len(files)} 个：{f}")
            res = process_file(f, recognizer, backend_name)
            results.append(res)

    write_outputs(results, args.output, args.format)
    ok = sum(1 for r in results if r["status"] == "ok")
    print(f"[汇总] 成功 {ok}/{len(results)}，结果见：{args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

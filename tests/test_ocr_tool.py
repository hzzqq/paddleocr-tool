"""paddleocr-tool 单元测试（无 OCR 重依赖，纯 Python 可跑）。

运行：pytest paddleocr-tool/tests
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import ocr_tool  # noqa: E402


def test_parse_args_defaults():
    args = ocr_tool.parse_args(["--input", "in", "--output", "out"])
    assert args.input == "in"
    assert args.output == "out"
    assert args.backend == "paddle"
    assert args.format == "md"
    assert args.min_conf == 0.0
    assert args.workers == 1


def test_collect_files_finds_images(tmp_path):
    (tmp_path / "a.png").write_bytes(b"\x89PNG")
    (tmp_path / "b.txt").write_text("x")
    (tmp_path / ".hidden.png").write_bytes(b"")
    files, skipped = ocr_tool.collect_files(str(tmp_path), recursive=False)
    names = {f.name for f in files}
    assert "a.png" in names
    assert "b.txt" not in names
    assert ".hidden.png" not in names  # 隐藏文件跳过


def _make_line(text, conf):
    # 真实 PaddleOCR 结构：result 是「行列表」，每行 = [bbox, (文本, 置信度)]
    return [[[0, 0], [1, 0], [1, 1], [0, 1]], (text, conf)]


class _FakeOCR:
    instances = 0

    def __init__(self, use_angle_cls=True, lang="ch"):
        _FakeOCR.instances += 1
        self.lang = lang

    def ocr(self, img_path, cls=True):
        # result 是「行列表」的包裹：result[0] 才是行列表，每行=[bbox,(文本,置信度)]
        return [[_make_line("你好世界", 0.99)]]


def test_recognize_paddle_singleton_and_text():
    ocr_tool._paddle_recognizer = None
    before = _FakeOCR.instances
    txt1 = ocr_tool.recognize_paddle(_FakeOCR, "fake1.png", "ch", min_conf=0.0)
    txt2 = ocr_tool.recognize_paddle(_FakeOCR, "fake2.png", "ch", min_conf=0.0)
    assert txt1 == "你好世界"
    assert txt2 == "你好世界"
    # 两次调用应复用同一识别器实例（单例），不重复构造
    assert _FakeOCR.instances - before == 1


class _LowConfOCR:
    instances = 0

    def __init__(self, use_angle_cls=True, lang="ch"):
        _LowConfOCR.instances += 1

    def ocr(self, img_path, cls=True):
        # 置信度 0.2，低于阈值应被丢弃，返回空串
        return [[_make_line("低置信文本", 0.2)]]


def test_recognize_paddle_min_conf_filter():
    ocr_tool._paddle_recognizer = None
    out = ocr_tool.recognize_paddle(_LowConfOCR, "x.png", "ch", min_conf=0.5)
    assert out == ""


def test_process_file_captures_error_and_chars():
    """隐性问题修复验证：处理异常应被捕获并写入 error 字段，同时记录 chars。"""
    def boom(img):
        raise ValueError("识别引擎炸了")

    res = ocr_tool.process_file(Path("x.png"), boom, "mock")
    assert res["status"] == "error"
    assert res["error"] == "识别引擎炸了"
    assert res["chars"] == 0
    assert res["elapsed"] >= 0


def test_write_outputs_txt_format(tmp_path):
    """R1 新需求：--format txt 应写出逐文件纯文本。"""
    results = [
        {"file": "a.png", "text": "你好世界", "chars": 4,
         "elapsed": 0.1, "status": "ok", "error": ""}
    ]
    ocr_tool.write_outputs(results, str(tmp_path), "txt")
    txt = (tmp_path / "a.txt").read_text(encoding="utf-8")
    assert txt == "你好世界\n"
    # json / csv / summary 仍应生成
    assert (tmp_path / "results.json").exists()
    assert (tmp_path / "results.csv").exists()
    assert (tmp_path / "summary.txt").exists()


def test_write_outputs_csv_includes_chars_and_error(tmp_path):
    """可观测性：csv 表头应含 chars/error 字段。"""
    results = [
        {"file": "a.png", "text": "hi", "chars": 2,
         "elapsed": 0.1, "status": "ok", "error": ""},
        {"file": "b.png", "text": "", "chars": 0,
         "elapsed": 0.0, "status": "error", "error": "boom"},
    ]
    ocr_tool.write_outputs(results, str(tmp_path), "md")
    csv_text = (tmp_path / "results.csv").read_text(encoding="utf-8-sig")
    assert "chars" in csv_text.splitlines()[0]
    assert "error" in csv_text.splitlines()[0]
    assert "boom" in csv_text


def test_collect_all_multiple_inputs(tmp_path):
    """R1 新需求验证：--input 支持逗号/换行分隔多路径并去重。"""
    (tmp_path / "a.png").write_bytes(b"x")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "b.png").write_bytes(b"x")
    spec = f"{tmp_path / 'a.png'},{tmp_path / 'sub'}"
    files, _ = ocr_tool.collect_all(spec, recursive=False)
    names = {f.name for f in files}
    assert names == {"a.png", "b.png"}


def test_recognize_paddle_rebuild_on_lang_change():
    """隐性正确性 bug 验证：切换语言应重建识别器，同语言复用。"""
    ocr_tool._paddle_recognizer = None
    ocr_tool._paddle_recognizer_lang = None
    before = _FakeOCR.instances
    ocr_tool.recognize_paddle(_FakeOCR, "x.png", "ch")
    ocr_tool.recognize_paddle(_FakeOCR, "y.png", "en")  # 换语言 -> 重建
    assert _FakeOCR.instances - before == 2
    ocr_tool.recognize_paddle(_FakeOCR, "z.png", "en")  # 同语言 -> 复用
    assert _FakeOCR.instances - before == 2

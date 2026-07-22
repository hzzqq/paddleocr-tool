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
    txt1, conf1 = ocr_tool.recognize_paddle(_FakeOCR, "fake1.png", "ch", min_conf=0.0)
    txt2, conf2 = ocr_tool.recognize_paddle(_FakeOCR, "fake2.png", "ch", min_conf=0.0)
    assert txt1 == "你好世界"
    assert txt2 == "你好世界"
    assert conf1 == [0.99]
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
    out, conf = ocr_tool.recognize_paddle(_LowConfOCR, "x.png", "ch", min_conf=0.5)
    assert out == ""
    assert conf == []


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


def test_collect_files_reports_skipped(tmp_path):
    """可观测性：不支持/隐藏文件应进入 skipped 列表。"""
    (tmp_path / "a.png").write_bytes(b"x")
    (tmp_path / "b.txt").write_text("x")
    (tmp_path / ".hidden.png").write_bytes(b"")
    files, skipped = ocr_tool.collect_files(str(tmp_path), recursive=False)
    names = {f.name for f in files}
    skip_names = {f.name for f in skipped}
    assert "a.png" in names
    assert "b.txt" in skip_names
    assert ".hidden.png" in skip_names


def test_main_dry_run_counts_only(tmp_path, capsys):
    """R1 新需求验证：--dry-run 只统计不执行 OCR。"""
    (tmp_path / "a.png").write_bytes(b"x")
    (tmp_path / "b.jpg").write_bytes(b"x")
    out = tmp_path / "out"
    rc = ocr_tool.main([
        "--input", str(tmp_path), "--output", str(out), "--dry-run",
    ])
    assert rc == 0
    out_text = capsys.readouterr().out
    assert "dry-run" in out_text
    assert "2 个文件" in out_text
    # 不应生成任何 OCR 产物
    assert not (out / "results.json").exists()


def test_parse_progress():
    """UI 流式进度的纯函数解析（DRY + 可单测）。"""
    assert ocr_tool.parse_progress("[进度] 完成 3/10：x.png (ok)") == (3, 10)
    assert ocr_tool.parse_progress("普通日志行") is None
    assert ocr_tool.parse_progress("") is None


def test_write_combined_md_excludes_failed(tmp_path):
    """R1 新需求验证：--combine 应把所有成功结果按序合并，失败项不计入。"""
    results = [
        {"file": "a.png", "text": "第一张", "chars": 3,
         "elapsed": 0.1, "status": "ok", "error": ""},
        {"file": "b.png", "text": "第二张", "chars": 3,
         "elapsed": 0.1, "status": "ok", "error": ""},
        {"file": "c.png", "text": "", "chars": 0,
         "elapsed": 0.0, "status": "error", "error": "boom"},
    ]
    path = ocr_tool.write_combined(results, str(tmp_path), "md")
    txt = path.read_text(encoding="utf-8")
    assert "## a.png" in txt and "第一张" in txt
    assert "## b.png" in txt and "第二张" in txt
    assert "c.png" not in txt  # 失败项不计入合并


def test_write_combined_txt_extension(tmp_path):
    """--combine 在 txt 格式下应生成 _combined.txt。"""
    results = [
        {"file": "a.png", "text": "内容", "chars": 2,
         "elapsed": 0.1, "status": "ok", "error": ""}
    ]
    path = ocr_tool.write_combined(results, str(tmp_path), "txt")
    assert path.name == "_combined.txt"
    assert "===== a.png =====" in path.read_text(encoding="utf-8")


def test_main_combine_and_workers_cap(tmp_path, capsys):
    """R1 新需求 + R2 隐性安全：--combine 生成合并文件；--workers 超限被钳制。"""
    img = tmp_path / "a.png"
    img.write_bytes(b"fake")
    out = tmp_path / "out"
    rc = ocr_tool.main([
        "--input", str(img), "--output", str(out),
        "--mock", "--combine", "--workers", "999",
    ])
    assert rc == 0
    assert (out / "_combined.md").exists()
    assert (out / "results.json").exists()
    out_text = capsys.readouterr().out
    assert "安全上限" in out_text  # workers 被钳制提示


def test_main_skip_existing(tmp_path, capsys):
    """R1 新需求验证：--skip-existing 跳过已有结果文件，实现断点续跑。"""
    img = tmp_path / "a.png"
    img.write_bytes(b"fake")
    out = tmp_path / "out"
    rc1 = ocr_tool.main(["--input", str(img), "--output", str(out), "--mock"])
    assert rc1 == 0
    assert (out / "a.md").exists()
    # 第二次运行应跳过（不报错、不覆盖）
    rc2 = ocr_tool.main([
        "--input", str(img), "--output", str(out), "--mock", "--skip-existing",
    ])
    assert rc2 == 0
    assert (out / "a.md").exists()
    out_text = capsys.readouterr().out
    assert "跳过" in out_text or "已存在" in out_text
    # 全部跳过时不应 overwrite 成空结果
    assert (out / "results.json").exists()


def test_process_file_aggregates_confidence(tmp_path):
    """R1 新需求验证：识别器返回 (text, confs)，process_file 聚合 avg/min 置信度。"""
    from pathlib import Path as _P

    f = _P(tmp_path / "x.png")
    f.write_bytes(b"fake")
    # 假识别器返回文本 + 逐行置信度
    def fake_recognizer(p):
        return "hello world", [0.9, 0.8, 0.7]

    res = ocr_tool.process_file(f, fake_recognizer, "fake")
    assert res["status"] == "ok"
    assert res["avg_conf"] == 0.8
    assert res["min_conf"] == 0.7


def test_main_emits_confidence_field(tmp_path):
    """置信度字段进入 results.json（mock 后端无可置信度时为 None）。"""
    import json as _json

    img = tmp_path / "a.png"
    img.write_bytes(b"fake")
    out = tmp_path / "out"
    rc = ocr_tool.main(["--input", str(img), "--output", str(out), "--mock"])
    assert rc == 0
    data = _json.loads((out / "results.json").read_text(encoding="utf-8"))
    assert "avg_conf" in data[0]
    assert data[0]["avg_conf"] is None  # mock 不产出置信度


def test_main_jsonl_format(tmp_path):
    """R1 新需求验证：--format jsonl 写出 results.jsonl（每行一条 JSON）。"""
    import json as _json

    img = tmp_path / "a.png"
    img.write_bytes(b"fake")
    out = tmp_path / "out"
    rc = ocr_tool.main([
        "--input", str(img), "--output", str(out), "--mock", "--format", "jsonl",
    ])
    assert rc == 0
    assert (out / "results.jsonl").exists()
    lines = (out / "results.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    # 每行都是合法 JSON，且含 file 字段
    obj = _json.loads(lines[0])
    assert obj["file"].endswith("a.png")
    # md 逐文件产物不应在 jsonl 模式下写出
    assert not (out / "a.md").exists()


def test_quiet_suppresses_progress(tmp_path, capsys):
    """R1 新需求验证：--quiet 不打印逐文件进度。"""
    img = tmp_path / "a.png"
    img.write_bytes(b"fake")
    out = tmp_path / "out"
    rc = ocr_tool.main([
        "--input", str(img), "--output", str(out), "--mock", "--quiet",
    ])
    assert rc == 0
    out_text = capsys.readouterr().out
    assert "[进度]" not in out_text
    # 关键结果（完成/汇总）仍应输出
    assert "results.json" in out_text


def test_main_surfaces_skipped_when_no_files(tmp_path, capsys):
    """R2 隐性可观测性验证：无可处理文件但存在被跳过的文件时，应说明跳过原因。"""
    (tmp_path / "note.txt").write_text("not an image")
    out = tmp_path / "out"
    rc = ocr_tool.main(["--input", str(tmp_path), "--output", str(out), "--mock"])
    assert rc == 0
    out_text = capsys.readouterr().out
    assert "跳过" in out_text
    assert "note.txt" in out_text  # 给出被跳过文件的例子，避免「未找到」无下文
    # 空结果产物仍应落盘（不崩溃）
    assert (out / "results.json").exists()


def test_collect_files_include_filter(tmp_path):
    """R1 新需求验证：--include 只收集文件名匹配的文件。"""
    (tmp_path / "cover.png").write_bytes(b"x")
    (tmp_path / "page2.png").write_bytes(b"x")
    (tmp_path / "other.jpg").write_bytes(b"x")
    files, _ = ocr_tool.collect_files(str(tmp_path), recursive=False, include="*page*")
    names = {f.name for f in files}
    assert names == {"page2.png"}
    # 子串匹配也应生效
    files2, _ = ocr_tool.collect_files(str(tmp_path), recursive=False, include="cover")
    assert {f.name for f in files2} == {"cover.png"}


def test_main_include_narrows_files(tmp_path, capsys):
    import json as _json

    (tmp_path / "a.png").write_bytes(b"x")
    (tmp_path / "b.png").write_bytes(b"x")
    out = tmp_path / "out"
    rc = ocr_tool.main([
        "--input", str(tmp_path), "--output", str(out), "--mock", "--include", "a",
    ])
    assert rc == 0
    data = _json.loads((out / "results.json").read_text(encoding="utf-8"))
    assert [r["file"].endswith("a.png") for r in data] == [True]


def test_main_warns_confidence_noop_on_mock(tmp_path, capsys):
    """R2 隐性问题验证：--min-conf 在 mock 后端下静默失效，应给出提示。"""
    img = tmp_path / "a.png"
    img.write_bytes(b"fake")
    out = tmp_path / "out"
    rc = ocr_tool.main([
        "--input", str(img), "--output", str(out), "--mock", "--min-conf", "0.5",
    ])
    assert rc == 0
    out_text = capsys.readouterr().out
    assert "仅 paddle 后端生效" in out_text


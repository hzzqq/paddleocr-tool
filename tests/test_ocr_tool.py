"""paddleocr-tool 单元测试（无 OCR 重依赖，纯 Python 可跑）。

运行：pytest paddleocr-tool/tests
"""

from __future__ import annotations

import json
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


def test_collect_files_ext_filter(tmp_path):
    """R1 新需求验证：--ext 只保留指定扩展名，覆盖默认图片/PDF 白名单。"""
    (tmp_path / "a.png").write_bytes(b"\x89PNG")
    (tmp_path / "b.jpg").write_bytes(b"\xff\xd8")
    (tmp_path / "c.pdf").write_bytes(b"%PDF")
    files, _ = ocr_tool.collect_files(str(tmp_path), recursive=False, exts=".png,.pdf")
    names = {f.name for f in files}
    assert names == {"a.png", "c.pdf"}
    assert "b.jpg" not in names  # jpg 被 --ext 排除


def test_collect_files_skips_hidden_dir(tmp_path):
    """R2 隐性健壮性验证：隐藏目录（如 .git）内的文件不应被收集。"""
    hidden = tmp_path / ".git"
    hidden.mkdir()
    (hidden / "config.png").write_bytes(b"\x89PNG")
    (tmp_path / "keep.png").write_bytes(b"\x89PNG")
    files, skipped = ocr_tool.collect_files(str(tmp_path), recursive=True)
    names = {f.name for f in files}
    assert "keep.png" in names
    assert "config.png" not in names  # 位于隐藏目录内，应跳过
    assert any(f.name == "config.png" for f in skipped)


def test_process_file_retries_then_succeeds(tmp_path):
    """R1 新需求验证：--retries 让偶发失败的单图识别自动重试直至成功。"""
    calls = {"n": 0}

    def flaky(img):
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("transient")
        return ("恢复的文本", [0.9])

    f = tmp_path / "a.png"
    f.write_bytes(b"\x89PNG")
    res = ocr_tool.process_file(f, flaky, "paddle", retries=2)
    assert res["status"] == "ok"
    assert res["text"] == "恢复的文本"
    assert calls["n"] == 3  # 重试两次后第 3 次成功


def test_process_file_retries_exhausted(tmp_path):
    """重试耗尽仍失败应标记为 error（而非静默成功）。"""

    def flaky(img):
        raise RuntimeError("boom")

    f = tmp_path / "a.png"
    f.write_bytes(b"\x89PNG")
    res = ocr_tool.process_file(f, flaky, "paddle", retries=1)
    assert res["status"] == "error"


def test_process_file_pdf_partial_page_failure(tmp_path, monkeypatch):
    """R2 隐性健壮性验证：PDF 仅部分页失败时不应丢弃其余页，整体仍标 ok。"""
    pages = {"n": 0}

    def rec(img):
        pages["n"] += 1
        if pages["n"] == 1:
            return ("第一页内容", [0.9])
        raise RuntimeError("page2 failed")

    monkeypatch.setattr(ocr_tool, "pdf_to_images", lambda p: [object(), object()])
    f = tmp_path / "doc.pdf"
    f.write_bytes(b"%PDF")
    res = ocr_tool.process_file(f, rec, "paddle", retries=0)
    assert res["status"] == "ok"  # 仅一页失败，整体仍可用
    assert "第一页内容" in res["text"]
    assert "第 2 页" in res["text"]  # 失败页以占位说明呈现
    assert "识别失败" in res["text"]


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


def test_write_combined_respects_format_json_and_jsonl(tmp_path):
    """R1 新能力：--format json/jsonl 的合并文件应为对应格式，而非 _combined.md。
    R2 验证：此前无论何种格式都只写 _combined.md，json 流水线混入多余 md。"""
    results = [
        {"file": "a.png", "text": "甲", "chars": 1,
         "elapsed": 0.1, "status": "ok", "error": ""},
        {"file": "b.png", "text": "乙", "chars": 1,
         "elapsed": 0.1, "status": "ok", "error": ""},
    ]
    p_json = ocr_tool.write_combined(results, str(tmp_path), "json")
    assert p_json.name == "_combined.json"
    arr = json.loads(p_json.read_text(encoding="utf-8"))
    assert isinstance(arr, list) and len(arr) == 2
    assert (tmp_path / "_combined.md").exists() is False  # 不产生多余 md

    p_jsonl = ocr_tool.write_combined(results, str(tmp_path / "jl"), "jsonl")
    assert p_jsonl.name == "_combined.jsonl"
    lines = p_jsonl.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["file"] == "a.png"


def test_write_combined_respects_format_csv(tmp_path):
    """R1 新能力：--format csv 的合并文件应为 _combined.csv（与 results.csv 同表头）。"""
    results = [
        {"file": "a.png", "text": "甲", "chars": 1,
         "elapsed": 0.1, "status": "ok", "error": "",
         "avg_conf": None, "min_conf": None},
    ]
    p_csv = ocr_tool.write_combined(results, str(tmp_path), "csv")
    assert p_csv.name == "_combined.csv"
    content = p_csv.read_text(encoding="utf-8-sig")
    assert "file" in content and "a.png" in content
    assert (tmp_path / "_combined.md").exists() is False


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


def test_list_formats_exits_and_lists(capsys):
    """R1 新能力：--list-formats 列出支持的 --format 输出格式即退出（返回0）。"""
    rc = ocr_tool.main(["--list-formats"])
    assert rc == 0
    out_text = capsys.readouterr().out
    assert "支持的输出格式" in out_text
    for fmt in ("md", "txt", "json", "jsonl", "csv"):
        assert fmt in out_text


def test_format_format_list_pure():
    """format_format_list 纯函数：默认格式为 md 且覆盖全部 5 种格式。"""
    text = ocr_tool.format_format_list()
    assert ocr_tool.DEFAULT_FORMAT == "md"
    for fmt in ("md", "txt", "json", "jsonl", "csv"):
        assert fmt in text
        assert ocr_tool.FORMAT_INFO[fmt]


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


def test_process_file_empty_marked_not_ok(tmp_path):
    """R2 修复验证：识别不到任何字（空文本、无异常）应标记为 empty，而非 ok。"""
    from pathlib import Path as _P

    f = _P(tmp_path / "blank.png")
    f.write_bytes(b"fake")
    res = ocr_tool.process_file(f, lambda p: ("", None), "fake")
    assert res["status"] == "empty"  # 此前会被误记为 ok
    assert res["chars"] == 0


def test_output_filename_collision_safe(tmp_path):
    """R1 新需求验证：不同目录的同名文件输出不再静默覆盖。"""
    out = tmp_path / "out"
    out.mkdir()
    # 模拟两个不同来源的同名文件（如 a/x.png 与 b/x.png）
    r1 = {"file": "dirA/x.png", "text": "结果A", "chars": 3,
           "elapsed": 0.1, "status": "ok", "error": ""}
    r2 = {"file": "dirB/x.png", "text": "结果B", "chars": 3,
           "elapsed": 0.1, "status": "ok", "error": ""}
    p1 = ocr_tool.write_markdown(r1, out)
    p2 = ocr_tool.write_markdown(r2, out)
    assert p1 != p2                      # 必须落到不同文件
    assert p1.read_text(encoding="utf-8").count("结果A") == 1
    assert p2.read_text(encoding="utf-8").count("结果B") == 1


def test_apply_min_chars_filters_noise():
    """R1 新需求验证：--min-chars 把过短的成功结果标记为 filtered。"""
    results = [
        {"file": "a.png", "text": "你好世界", "chars": 4, "status": "ok", "elapsed": 0.1},
        {"file": "b.png", "text": "x", "chars": 1, "status": "ok", "elapsed": 0.1},
        {"file": "c.png", "text": "", "chars": 0, "status": "empty", "elapsed": 0.0},
    ]
    filtered = ocr_tool.apply_min_chars(results, 3)
    assert results[0]["status"] == "ok"      # 4 字，保留
    assert results[1]["status"] == "filtered"  # 1 字，被过滤
    assert results[2]["status"] == "empty"    # 已是 empty，不受影响
    assert len(filtered) == 1


def test_apply_min_chars_noop_when_zero():
    results = [{"file": "a.png", "text": "x", "chars": 1, "status": "ok"}]
    assert ocr_tool.apply_min_chars(results, 0) == []
    assert results[0]["status"] == "ok"


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


def test_write_outputs_emits_stats_json(tmp_path):
    """R1 新需求：write_outputs 应额外写出机器可读的 stats.json 运行统计。"""
    import json as _json

    results = [
        {"file": "a.png", "text": "你好世界", "chars": 4, "elapsed": 0.2,
         "status": "ok", "error": "", "avg_conf": 0.9, "min_conf": 0.8},
        {"file": "b.png", "text": "hi", "chars": 2, "elapsed": 0.1,
         "status": "ok", "error": "", "avg_conf": 0.7, "min_conf": 0.6},
        {"file": "c.png", "text": "", "chars": 0, "elapsed": 0.0,
         "status": "error", "error": "boom", "avg_conf": None, "min_conf": None},
    ]
    ocr_tool.write_outputs(results, str(tmp_path), "md")
    stats = _json.loads((tmp_path / "stats.json").read_text(encoding="utf-8"))
    assert stats["total"] == 3
    assert stats["ok"] == 2
    assert stats["error"] == 1
    assert stats["total_chars"] == 6
    # 仅对有置信度的 ok 文件求平均：(0.9+0.7)/2 = 0.8
    assert stats["avg_conf_overall"] == 0.8
    assert stats["format"] == "md"
    # 人读 summary 仍应同时产出
    assert (tmp_path / "summary.txt").exists()


def test_write_combined_skips_when_empty(tmp_path, capsys):
    """R2 隐性问题：无成功结果时 write_combined 不应写出空合并文件，应跳过并提示。"""
    results = [
        {"file": "c.png", "text": "", "chars": 0, "elapsed": 0.0,
         "status": "error", "error": "boom"},
    ]
    path = tmp_path / "_combined.md"
    returned = ocr_tool.write_combined(results, str(tmp_path), "md")
    assert returned is None
    assert not path.exists()  # 不应留下误导性的空合并文件
    assert "没有可合并的成功结果" in capsys.readouterr().out  # R2：明确提示已跳过


def test_main_emits_stats_and_no_empty_combined(tmp_path, capsys):
    """R1+R2 端到端：零可处理文件时仍产出 stats.json，且合并分支不写出空文件。"""
    (tmp_path / "note.txt").write_text("not an image")
    out = tmp_path / "out"
    rc = ocr_tool.main([
        "--input", str(tmp_path), "--output", str(out), "--mock", "--combine",
    ])
    assert rc == 0
    assert (out / "stats.json").exists()  # R1：运行统计始终产出
    assert not (out / "_combined.md").exists()  # R2：无内容时不应产生合并文件


def test_resolve_lang_known_maps_per_backend():
    """R2 修复验证：已知语言按后端解析为正确引擎代码。"""
    assert ocr_tool.resolve_lang("fr", "paddle") == "french"
    assert ocr_tool.resolve_lang("fr", "tesseract") == "fra"   # 此前会原样传 "fr" 导致失败
    assert ocr_tool.resolve_lang("ch", "tesseract") == "chi_sim+eng"
    assert ocr_tool.resolve_lang("en", "tesseract") == "eng"


def test_resolve_lang_unknown_passthrough():
    """未知语言原样透传（由后端决定，并在 main 中告警）。"""
    assert ocr_tool.resolve_lang("zzz", "paddle") == "zzz"
    assert ocr_tool.resolve_lang("zzz", "tesseract") == "zzz"


def test_format_lang_list_contains_default():
    """R1 新需求：--lang-list 文本应含默认语言与多个官方支持代码。"""
    text = ocr_tool.format_lang_list()
    assert "ch" in text and "fr" in text and "fra" in text
    assert ocr_tool.DEFAULT_LANG in text


def test_main_lang_list_returns_zero(tmp_path, capsys):
    """R1 新需求验证：--lang-list 仅列出语言并退出，无需 --input/--output。"""
    rc = ocr_tool.main(["--lang-list"])
    assert rc == 0
    out_text = capsys.readouterr().out
    assert "支持的识别语言" in out_text
    assert "fra" in out_text  # tesseract 映射


def test_main_invalid_lang_warns(tmp_path, capsys):
    """R2 修复验证：未知 --lang 应显式告警，而非静默透传。"""
    img = tmp_path / "a.png"
    img.write_bytes(b"fake")
    out = tmp_path / "out"
    rc = ocr_tool.main([
        "--input", str(img), "--output", str(out), "--mock", "--lang", "chh",
    ])
    assert rc == 0
    out_text = capsys.readouterr().out
    assert "不在官方支持列表" in out_text
    assert "chh" in out_text


def test_main_valid_lang_no_warning(tmp_path, capsys):
    """已知语言（en）不应触发未知语言告警。"""
    img = tmp_path / "a.png"
    img.write_bytes(b"fake")
    out = tmp_path / "out"
    rc = ocr_tool.main([
        "--input", str(img), "--output", str(out), "--mock", "--lang", "en",
    ])
    assert rc == 0
    out_text = capsys.readouterr().out
    assert "不在官方支持列表" not in out_text


def test_main_missing_required_args_returns_error():
    """可观测性：未提供 --input/--output 时给出明确错误码 1。"""
    rc = ocr_tool.main([])
    assert rc == 1


def test_main_max_files_limits_processing(tmp_path, capsys):
    """R1 新需求验证：--max-files N 只处理前 N 个文件（其余计入跳过）。"""
    for i in range(5):
        (tmp_path / f"img{i}.png").write_bytes(b"x")
    out = tmp_path / "out"
    rc = ocr_tool.main([
        "--input", str(tmp_path), "--output", str(out), "--mock", "--max-files", "2",
    ])
    assert rc == 0
    data = json.loads((out / "results.json").read_text(encoding="utf-8"))
    assert len(data) == 2  # 仅处理前 2 个
    out_text = capsys.readouterr().out
    assert "max-files" in out_text
    assert "3 个" in out_text  # 其余 3 个计入跳过


def test_main_min_chars_reflected_in_results_json(tmp_path):
    """R2 验证：--min-chars 标记 filtered 必须在 write_outputs 之前完成，
    保证 results.json / 逐文件产物状态与最终统计一致（原实现顺序相反，磁盘产物报 ok
    但统计报 filtered，互相矛盾）。"""
    (tmp_path / "a.png").write_bytes(b"x")
    (tmp_path / "b.png").write_bytes(b"x")
    out = tmp_path / "out"
    rc = ocr_tool.main([
        "--input", str(tmp_path), "--output", str(out), "--mock", "--min-chars", "9999",
    ])
    assert rc == 0
    data = json.loads((out / "results.json").read_text(encoding="utf-8"))
    # mock 返回的文本较短，应全部被标记 filtered，而非 ok
    assert all(d["status"] == "filtered" for d in data)
    # summary/stats 的成功数也应与产物一致（均为 0）
    summary = (out / "summary.txt").read_text(encoding="utf-8")
    assert "成功(ok)：0" in summary



def test_main_fail_on_error_returns_nonzero(tmp_path, monkeypatch, capsys):
    """R1 新需求验证：--fail-on-error 时，任一文件识别失败应返回退出码 1。"""
    img = tmp_path / "a.png"
    img.write_bytes(b"fake")
    out = tmp_path / "out"

    def boom_recognizer(*a, **k):
        raise RuntimeError("模拟识别失败")

    def fake_build(args):
        return boom_recognizer, "mock"

    monkeypatch.setattr(ocr_tool, "build_recognizer", fake_build)

    # 不带 --fail-on-error：默认吞掉失败，退出 0
    rc0 = ocr_tool.main(["--input", str(img), "--output", str(out), "--mock"])
    assert rc0 == 0
    # 带 --fail-on-error：升级为非零退出码
    rc1 = ocr_tool.main([
        "--input", str(img), "--output", str(out), "--mock", "--fail-on-error",
    ])
    assert rc1 == 1
    err = capsys.readouterr().err
    assert "退出码置为 1" in err


def test_collect_files_missing_path_reports_stderr(tmp_path, capsys):
    """R2 隐性问题验证：输入路径不存在时，错误应打印到 stderr 而非 stdout。"""
    bad = tmp_path / "nope" / "x.png"
    out = tmp_path / "out"
    rc = ocr_tool.main(["--input", str(bad), "--output", str(out), "--mock"])
    captured = capsys.readouterr()
    assert "输入路径不存在" in captured.err
    assert "输入路径不存在" not in captured.out
    # 退出码仍为 0（保持原有行为：无文件可处理也算正常收尾）
    assert rc == 0


def test_sort_files_orders_by_name_and_size(tmp_path):
    """R1：sort_files 按 name 字典序、按 size 降序正确排序，且不修改入参。"""
    f1 = tmp_path / "b.png"
    f2 = tmp_path / "a.png"
    f3 = tmp_path / "c.png"
    f1.write_bytes(b"x" * 100)
    f2.write_bytes(b"x" * 10)
    f3.write_bytes(b"x" * 50)
    files = [f1, f2, f3]
    by_name = ocr_tool.sort_files(files, "name")
    assert [p.name for p in by_name] == ["a.png", "b.png", "c.png"]
    by_size = ocr_tool.sort_files(files, "size")
    assert [p.name for p in by_size] == ["b.png", "c.png", "a.png"]  # 体积 100/50/10
    # 不修改原列表
    assert [p.name for p in files] == ["b.png", "a.png", "c.png"]


def test_sort_files_empty_is_safe():
    assert ocr_tool.sort_files([], "name") == []
    assert ocr_tool.sort_files([], "size") == []


def test_process_file_none_text_does_not_crash(tmp_path):
    """R2 防护验证：识别器返回 (None, None) 时不应抛 TypeError，
    而应被规整为空文本并标记为 empty。"""
    img = tmp_path / "x.png"
    img.write_bytes(b"fake")
    res = ocr_tool.process_file(img, lambda x: (None, None), "mock")
    assert res["status"] == "empty"
    assert res["chars"] == 0
    assert res["text"] == ""


def test_summarize_statuses_counts_all_five(tmp_path):
    """R2 修复验证：汇总应覆盖 ok/empty/filtered/error/skipped_pdf 五类状态，
    而非只统计 ok/error（此前 empty/filtered 被漏算，误导质量判断）。"""
    results = [
        {"file": "a.png", "text": "x", "chars": 1, "status": "ok", "elapsed": 0.1},
        {"file": "b.png", "text": "y", "chars": 1, "status": "ok", "elapsed": 0.1},
        {"file": "c.png", "text": "", "chars": 0, "status": "empty", "elapsed": 0.0},
        {"file": "d.png", "text": "z", "chars": 1, "status": "filtered", "elapsed": 0.1},
        {"file": "e.pdf", "text": "", "chars": 0, "status": "skipped_pdf", "elapsed": 0.0},
        {"file": "f.png", "text": "", "chars": 0, "status": "error", "elapsed": 0.0, "error": "boom"},
    ]
    s = ocr_tool.summarize_statuses(results)
    assert s["total"] == 6
    assert s["ok"] == 2
    assert s["empty"] == 1
    assert s["filtered"] == 1
    assert s["skipped_pdf"] == 1
    assert s["error"] == 1


def test_filter_results_by_status():
    """R1 新需求验证：--status-filter 的纯函数只保留指定状态。"""
    results = [
        {"file": "a.png", "text": "x", "status": "ok"},
        {"file": "b.png", "text": "", "status": "empty"},
        {"file": "c.png", "text": "z", "status": "filtered"},
    ]
    only_ok = ocr_tool.filter_results_by_status(results, ["ok"])
    assert [r["file"] for r in only_ok] == ["a.png"]
    ok_empty = ocr_tool.filter_results_by_status(results, ["ok", "empty"])
    assert {r["file"] for r in ok_empty} == {"a.png", "b.png"}
    # 空过滤条件应原样返回全部
    assert ocr_tool.filter_results_by_status(results, None) == results
    assert ocr_tool.filter_results_by_status(results, []) == results


def test_write_outputs_stats_includes_empty_filtered(tmp_path):
    """R2 修复验证：stats.json 与 summary.txt 应呈现 empty/filtered 计数。"""
    import json as _json

    results = [
        {"file": "a.png", "text": "你好世界", "chars": 4, "elapsed": 0.2,
         "status": "ok", "error": "", "avg_conf": 0.9, "min_conf": 0.8},
        {"file": "b.png", "text": "", "chars": 0, "elapsed": 0.0,
         "status": "empty", "error": "", "avg_conf": None, "min_conf": None},
        {"file": "c.png", "text": "短", "chars": 1, "elapsed": 0.1,
         "status": "filtered", "error": ""},
    ]
    ocr_tool.write_outputs(results, str(tmp_path), "md")
    stats = _json.loads((tmp_path / "stats.json").read_text(encoding="utf-8"))
    assert stats["empty"] == 1
    assert stats["filtered"] == 1
    assert stats["ok"] == 1
    summary = (tmp_path / "summary.txt").read_text(encoding="utf-8")
    assert "空白(empty)：1" in summary
    assert "噪声(filtered)：1" in summary


def test_main_status_filter_limits_per_file_output(tmp_path, capsys):
    """R1 新需求端到端：--status-filter ok 只写出成功结果的逐文件产物，
    empty/filtered 文件无对应 .md，但 results.json 仍含全部结果。"""
    import json as _json

    (tmp_path / "a.png").write_bytes(b"x")
    (tmp_path / "b.png").write_bytes(b"x")
    out = tmp_path / "out"
    rc = ocr_tool.main([
        "--input", str(tmp_path), "--output", str(out), "--mock",
        "--min-chars", "9999", "--status-filter", "ok",
    ])
    assert rc == 0
    data = _json.loads((out / "results.json").read_text(encoding="utf-8"))
    # 全量审计：mock 文本较短，全部被标 filtered，但因 status-filter=ok 不写逐文件产物
    assert all(d["status"] == "filtered" for d in data)
    assert not (out / "a.md").exists()
    assert not (out / "b.md").exists()
    assert (out / "stats.json").exists()  # 审计信息仍保留
    out_text = capsys.readouterr().out
    assert "status-filter" in out_text or "仅写出" in out_text


def test_write_combined_respects_status_filter(tmp_path):
    """R1 新需求验证：--status-filter 下合并文件只含指定状态。"""
    results = [
        {"file": "a.png", "text": "成功一", "chars": 3, "status": "ok", "error": ""},
        {"file": "b.png", "text": "成功二", "chars": 3, "status": "ok", "error": ""},
        {"file": "c.png", "text": "被过滤", "chars": 3, "status": "filtered", "error": ""},
    ]
    path = ocr_tool.write_combined(results, str(tmp_path), "md", status_filter=["ok"])
    txt = path.read_text(encoding="utf-8")
    assert "成功一" in txt and "成功二" in txt
    assert "c.png" not in txt and "被过滤" not in txt


def test_build_run_report_groups_files(tmp_path):
    """R1 新需求验证：build_run_report 应给出各状态文件清单与建议重跑清单。"""
    results = [
        {"file": "a.png", "text": "x", "chars": 1, "status": "ok", "elapsed": 0.1},
        {"file": "b.png", "text": "", "chars": 0, "status": "empty", "elapsed": 0.0},
        {"file": "c.png", "text": "短", "chars": 1, "status": "filtered", "elapsed": 0.1},
        {"file": "d.png", "text": "", "chars": 0, "status": "error", "elapsed": 0.0, "error": "boom"},
    ]
    rep = ocr_tool.build_run_report(results)
    assert rep["counts"]["ok"] == 1
    assert set(rep["by_status"]["empty"]) == {"b.png"}
    assert set(rep["by_status"]["filtered"]) == {"c.png"}
    assert set(rep["by_status"]["error"]) == {"d.png"}
    # 建议重跑 = empty / filtered / error 三类
    assert set(rep["rerun_candidates"]) == {"b.png", "c.png", "d.png"}


def test_main_report_writes_json(tmp_path, capsys):
    """R1 新需求端到端：--report 把运行报告写入指定 JSON 文件。"""
    import json as _json

    (tmp_path / "a.png").write_bytes(b"x")
    (tmp_path / "b.png").write_bytes(b"x")
    out = tmp_path / "out"
    rep_path = tmp_path / "report.json"
    rc = ocr_tool.main([
        "--input", str(tmp_path), "--output", str(out), "--mock",
        "--min-chars", "9999", "--report", str(rep_path),
    ])
    assert rc == 0
    rep = _json.loads(rep_path.read_text(encoding="utf-8"))
    assert rep["counts"]["filtered"] == 2
    # 建议重跑清单应包含两个被过滤（短文本）的文件（file 为完整路径）
    assert len(rep["rerun_candidates"]) == 2
    assert set(rep["rerun_candidates"]) == {
        str(tmp_path / "a.png"), str(tmp_path / "b.png")
    }
    assert "已写出运行报告" in capsys.readouterr().out

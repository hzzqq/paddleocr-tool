"""paddleocr-tool 单元测试（无 OCR 重依赖，纯 Python 可跑）。

运行：pytest paddleocr-tool/tests
"""

from __future__ import annotations

import json
import os
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


def test_parse_manifest(tmp_path):
    """R1 新需求验证：@清单解析——空行与 # 注释跳过，逐行展开。"""
    mf = tmp_path / "batch.txt"
    mf.write_text("\n# 这是注释\na.png\n  b.png  \n\nc.pdf\n", encoding="utf-8")
    (tmp_path / "a.png").write_bytes(b"\x89PNG")
    (tmp_path / "b.png").write_bytes(b"\x89PNG")
    (tmp_path / "c.pdf").write_bytes(b"%PDF")
    parts, skipped = ocr_tool.parse_manifest(str(mf))
    # 相对清单内的相对路径按清单目录解析为绝对路径（自包含）
    expected = [str(tmp_path / n) for n in ("a.png", "b.png", "c.pdf")]
    assert parts == expected  # 注释/空行被剔除，空白被 strip
    assert skipped == []  # 清单可读，无跳过


def test_parse_manifest_unreadable(tmp_path):
    """R2 健壮化：清单文件不存在/不可读时安全返回空片段+跳过项，不抛错。"""
    parts, skipped = ocr_tool.parse_manifest(str(tmp_path / "nope.txt"))
    assert parts == []
    assert len(skipped) == 1


def test_collect_all_manifest(tmp_path):
    """R1 新需求验证：--input @清单 与字面/额外路径组合批次。"""
    (tmp_path / "a.png").write_bytes(b"\x89PNG")
    (tmp_path / "b.png").write_bytes(b"\x89PNG")
    (tmp_path / "c.pdf").write_bytes(b"%PDF")
    (tmp_path / "skip.txt").write_text("x")
    mf = tmp_path / "batch.txt"
    mf.write_text("a.png\nb.png\n", encoding="utf-8")
    # c.pdf 字面路径 + 一个不应被收纳的 skip.txt（不在白名单）一起传入验证跳过
    files, skipped = ocr_tool.collect_all(
        f"@{mf},{tmp_path / 'c.pdf'},{tmp_path / 'skip.txt'}", False, exts=None
    )
    names = {f.name for f in files}
    # 清单里的 a/b 与额外的 c.pdf 都应被纳入
    assert names == {"a.png", "b.png", "c.pdf"}
    assert any(s.name == "skip.txt" for s in skipped)  # txt 不在白名单，计入跳过


def test_collect_all_manifest_missing_is_skipped(tmp_path):
    """R2 隐性可观测性：不可读清单计入 skipped 而非静默崩溃。"""
    files, skipped = ocr_tool.collect_all("@missing.txt", False)
    assert files == []
    assert any(s.name == "missing.txt" for s in skipped)


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


def test_redact_text_masks_patterns():
    """R1 新需求验证：redact_text 按正则打码敏感信息。"""
    text = "身份证 110105199003078888 手机 13812345678"
    out, n = ocr_tool.redact_text(
        text, [r"\d{17}[\dX]", r"1[3-9]\d{9}"]
    )
    assert n == 2
    assert out.count("***") == 2
    assert "110105199003078888" not in out
    assert "13812345678" not in out


def test_redact_text_invalid_pattern_skipped():
    """R2 隐性健壮性：非法正则仅告警跳过，不中断其余规则、不抛错。"""
    out, n = ocr_tool.redact_text("订单 12345 备注 ok", [r"([", r"\d+"])
    assert n == 1  # 仅合法 \d+ 生效
    assert "12345" not in out
    assert "ok" in out


def test_redact_text_empty_safe():
    """R2 边界：空文本 / 空规则安全返回。"""
    assert ocr_tool.redact_text("", [r"\d+"]) == ("", 0)
    assert ocr_tool.redact_text("abc", None) == ("abc", 0)
    assert ocr_tool.redact_text("abc", []) == ("abc", 0)


def test_process_file_redacts(tmp_path):
    """R1 新需求验证：process_file 把 redact 规则落到最终 text 并报告 redact_count。"""
    def fake_recognizer(img):
        return ("客户手机 13812345678 谢谢", [0.95])

    f = tmp_path / "a.png"
    f.write_bytes(b"\x89PNG")
    res = ocr_tool.process_file(f, fake_recognizer, "paddle", redact=[r"1[3-9]\d{9}"])
    assert res["status"] == "ok"
    assert res["redact_count"] == 1
    assert "13812345678" not in res["text"]
    assert "***" in res["text"]



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


def test_no_angle_flag_parsing():
    """R1 新需求：--no-angle 关闭角度分类、--angle 显式开启，且默认开启。"""
    assert ocr_tool.parse_args(["--no-angle"]).use_angle_cls is False
    assert ocr_tool.parse_args(["--angle"]).use_angle_cls is True
    assert ocr_tool.parse_args([]).use_angle_cls is True


class _AngleSpyOCR:
    instances = 0
    last_cls = None

    def __init__(self, use_angle_cls=True, lang="ch"):
        _AngleSpyOCR.instances += 1
        self.use_angle_cls = use_angle_cls
        self.lang = lang

    def ocr(self, img_path, cls=True):
        _AngleSpyOCR.last_cls = cls
        return [[_make_line("文本", 0.9)]]


def test_recognize_paddle_angle_rebuilds_singleton():
    """R2 修复验证：单例缓存键必须包含 use_angle_cls，否则切换 --no-angle
    后仍复用旧（角度开启）实例、导致开关失效。这里断言切换后识别器被重建，
    且 ocr 调用的 cls 参数与 use_angle_cls 一致。"""
    ocr_tool._paddle_recognizer = None
    ocr_tool._paddle_recognizer_lang = None
    ocr_tool._paddle_recognizer_angle = None
    before = _AngleSpyOCR.instances
    ocr_tool.recognize_paddle(_AngleSpyOCR, "a.png", "ch", use_angle_cls=True)
    built_with_true = _AngleSpyOCR.instances - before
    # 切换到 --no-angle：应重建实例并透传 cls=False
    ocr_tool.recognize_paddle(_AngleSpyOCR, "b.png", "ch", use_angle_cls=False)
    assert _AngleSpyOCR.instances - before == built_with_true + 1
    assert _AngleSpyOCR.last_cls is False
    # 再次以 True 调用：又应重建一次（缓存键含 angle）
    ocr_tool.recognize_paddle(_AngleSpyOCR, "c.png", "ch", use_angle_cls=True)
    assert _AngleSpyOCR.instances - before == built_with_true + 2
    assert _AngleSpyOCR.last_cls is True


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


def test_write_outputs_custom_summary_name(tmp_path):
    """R1 新需求：--summary-file 自定义汇总文件名（相对 output 的子路径）。"""
    results = [
        {"file": "a.png", "text": "你好世界", "chars": 4,
         "elapsed": 0.1, "status": "ok", "error": ""}
    ]
    ocr_tool.write_outputs(results, str(tmp_path), "md", summary_name="run1_summary.txt")
    assert not (tmp_path / "summary.txt").exists()  # 默认名不再写出
    custom = tmp_path / "run1_summary.txt"
    assert custom.exists()
    assert "成功(ok)：1" in custom.read_text(encoding="utf-8")


def test_write_outputs_summary_creates_parent_dir(tmp_path):
    """R2 边界：--summary-file 指向不存在的嵌套目录时，父目录应自动创建。"""
    results = [
        {"file": "a.png", "text": "x", "chars": 1,
         "elapsed": 0.1, "status": "ok", "error": ""}
    ]
    nested = tmp_path / "nested" / "deep" / "my_summary.txt"
    ocr_tool.write_outputs(results, str(tmp_path), "md", summary_name=str(nested))
    assert nested.exists()


def test_write_outputs_summary_absolute_path(tmp_path):
    """R2 边界：--summary-file 为绝对路径时，写到该绝对位置（不限于 output_dir）。"""
    import tempfile, os
    results = [
        {"file": "a.png", "text": "x", "chars": 1,
         "elapsed": 0.1, "status": "ok", "error": ""}
    ]
    ext = tempfile.mkdtemp()
    abs_sum = os.path.join(ext, "abs_summary.txt")
    try:
        ocr_tool.write_outputs(results, str(tmp_path), "md", summary_name=abs_sum)
        assert os.path.exists(abs_sum)
    finally:
        if os.path.exists(abs_sum):
            os.remove(abs_sum)
        os.rmdir(ext)


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


def test_main_skip_existing_resume_recognizes_new_file(tmp_path, capsys):
    """R2 回归：目录新增文件在已有 results.json 时，--skip-existing 仍应识别它。

    旧实现：done_paths 对汇总格式一律返回「results.json 是否存在」，等于
    「只要跑过一次，之后任何文件都算已完成」——新加入目录的文件永远不会被
    识别，中断后续跑原地空转，与断点续跑承诺完全相反。
    """
    for n in ("a", "b", "c"):
        (tmp_path / f"{n}.png").write_bytes(b"fake")
    out = tmp_path / "out"
    rc1 = ocr_tool.main(["--input", str(tmp_path), "--output", str(out), "--mock"])
    assert rc1 == 0
    assert (out / "results.json").exists()
    # 新增 d.png 后带 --skip-existing 续跑
    (tmp_path / "d.png").write_bytes(b"fake")
    rc2 = ocr_tool.main([
        "--input", str(tmp_path), "--output", str(out), "--mock", "--skip-existing",
    ])
    assert rc2 == 0
    # 新增文件必须被实际处理并产出（旧实现下 d.md 永远不会生成）
    assert (out / "d.md").exists()
    out_text = capsys.readouterr().out
    assert "跳过" in out_text  # 旧的 a/b/c 被正确跳过


def test_main_skip_existing_resume_preserves_history(tmp_path):
    """R2 回归：--skip-existing 续跑后 results.json 仍含历史全部文件记录。

    旧实现：write_outputs 整体重写 results.json，仅写本轮新处理的文件，
    把之前已识别的 a/b/c 记录静默抹掉（「断点续跑」反把已得成果删了）。
    """
    for n in ("a", "b", "c"):
        (tmp_path / f"{n}.png").write_bytes(b"fake")
    out = tmp_path / "out"
    ocr_tool.main(["--input", str(tmp_path), "--output", str(out), "--mock"])
    (tmp_path / "d.png").write_bytes(b"fake")
    ocr_tool.main([
        "--input", str(tmp_path), "--output", str(out), "--mock", "--skip-existing",
    ])
    data = json.loads((out / "results.json").read_text(encoding="utf-8"))
    files = sorted(os.path.basename(r["file"]) for r in data)
    assert files == ["a.png", "b.png", "c.png", "d.png"]


def test_merge_previous_results_unit():
    """merge_previous_results：本轮结果按路径覆盖旧记录，其余旧记录保留。"""
    prev = [
        {"file": "a.png", "status": "ok", "text": "old-a"},
        {"file": "b.png", "status": "ok", "text": "old-b"},
    ]
    curr = [{"file": "b.png", "status": "ok", "text": "new-b"}]
    merged = ocr_tool.merge_previous_results(prev, curr)
    by_file = {os.path.normcase(r["file"]): r for r in merged}
    assert len(merged) == 2
    assert by_file["a.png"]["text"] == "old-a"
    assert by_file["b.png"]["text"] == "new-b"


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


def test_normalize_text_pure():
    """R1 验证：normalize_text 折叠空白、删除空行、去首尾空白。"""
    assert ocr_tool.normalize_text("  hello   world  \n") == "hello world"
    assert ocr_tool.normalize_text("a\n\n\n\nb\n\n") == "a\nb"
    assert ocr_tool.normalize_text("\t x \t y \t") == "x y"
    assert ocr_tool.normalize_text("") == ""


def test_normalize_text_strips_invisible_chars():
    """R2 修复验证：零宽/不可见字符（\u200b 等）会被剔除，否则下游串比对出现
    「看起来一样实则不等」的隐性 bug。"""
    assert ocr_tool.normalize_text("a\u200bb") == "ab"  # 零宽空格被去掉（相邻字母直接拼接）
    assert ocr_tool.normalize_text("c\ufeffd") == "cd"  # BOM 残留被去掉
    assert ocr_tool.normalize_text("\u200b\u200c\u200dx") == "x"


def test_dedup_lines_pure():
    """R1 验证：dedup_lines 仅折叠严格相邻的重复行。"""
    assert ocr_tool.dedup_lines("a\na\nb") == "a\nb"
    assert ocr_tool.dedup_lines("a\na\na") == "a"
    assert ocr_tool.dedup_lines("") == ""
    # 被其它行隔开的相同行各自保留（不误删正文正常重复）
    assert ocr_tool.dedup_lines("a\n\nb\na") == "a\n\nb\na"


def test_process_file_dedup_flag(tmp_path):
    """R1 验证：--dedup-lines（process_file dedup=True）删除连续重复行。"""
    from pathlib import Path as _P

    f = _P(tmp_path / "dup.png")
    f.write_bytes(b"fake")
    repeated = "标题\n标题\n正文\n正文\n结尾"
    res = ocr_tool.process_file(f, lambda p: (repeated, None), "fake", dedup=True)
    assert res["status"] == "ok"
    assert res["text"] == "标题\n正文\n结尾"
    # 不带 --dedup-lines 时保留原样
    res2 = ocr_tool.process_file(f, lambda p: (repeated, None), "fake")
    assert res2["text"] == repeated


def test_process_file_normalize_flag(tmp_path):
    """R1 验证：--normalize（process_file normalize=True）规整识别文本。"""
    from pathlib import Path as _P

    f = _P(tmp_path / "messy.png")
    f.write_bytes(b"fake")
    messy = "  第  一行   \n\n\n   第二行   \n"
    res = ocr_tool.process_file(f, lambda p: (messy, None), "fake", normalize=True)
    assert res["status"] == "ok"
    assert res["text"] == "第 一行\n第二行"


def test_process_file_whitespace_only_marked_empty(tmp_path):
    """R2 修复验证：仅含空白的文本应判为 empty（此前误记为 ok 污染成功数）。"""
    from pathlib import Path as _P

    f = _P(tmp_path / "ws.png")
    f.write_bytes(b"fake")
    res = ocr_tool.process_file(f, lambda p: ("   \n  \t ", None), "fake")
    assert res["status"] == "empty"  # 此前因字符串非空被误判为 ok
    res2 = ocr_tool.process_file(f, lambda p: ("   \n  \t ", None), "fake", normalize=True)
    assert res2["status"] == "empty"


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


def test_main_min_conf_out_of_range_clamped(tmp_path, capsys):
    """R1/R2 护栏：--min-conf 误用百分比(60>1) 时应钳制到 [0,1] 并告警。

    修复前：用户传 --min-conf 60（误以为百分比）会在 paddle 后端下过滤掉
    全部识别行、结果为空且无任何提示（静默全丢）。
    """
    img = tmp_path / "a.png"
    img.write_bytes(b"fake")
    out = tmp_path / "out"
    rc = ocr_tool.main([
        "--input", str(img), "--output", str(out), "--mock", "--min-conf", "60",
    ])
    assert rc == 0
    out_text = capsys.readouterr().out
    assert "0~1 之间" in out_text and "钳制" in out_text


def test_main_low_conf_threshold_out_of_range_clamped(tmp_path, capsys):
    """R1/R2 护栏：--low-conf-threshold 越界(>1) 时钳制到 1.0 并告警。"""
    img = tmp_path / "a.png"
    img.write_bytes(b"fake")
    out = tmp_path / "out"
    rc = ocr_tool.main([
        "--input", str(img), "--output", str(out), "--mock", "--low-conf-threshold", "5",
    ])
    assert rc == 0
    out_text = capsys.readouterr().out
    assert "0~1 之间" in out_text and "钳制" in out_text


def test_main_min_conf_valid_range_no_clamp_warning(tmp_path, capsys):
    """合法区间内的 --min-conf（如 0.5）不应触发钳制告警。"""
    img = tmp_path / "a.png"
    img.write_bytes(b"fake")
    out = tmp_path / "out"
    rc = ocr_tool.main([
        "--input", str(img), "--output", str(out), "--mock", "--min-conf", "0.5",
    ])
    assert rc == 0
    out_text = capsys.readouterr().out
    assert "0~1 之间" not in out_text


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


def test_available_backends_shape():
    """R1 新能力：探测各后端依赖可用性，返回固定三键的布尔字典。"""
    b = ocr_tool.available_backends()
    assert set(b.keys()) == {"paddle", "tesseract", "pdf2image"}
    assert all(isinstance(v, bool) for v in b.values())


def test_format_backends_list_is_string():
    """format_backends_list 返回可读字符串且含三个后端名。"""
    s = ocr_tool.format_backends_list()
    assert isinstance(s, str)
    assert "paddle" in s and "tesseract" in s and "pdf2image" in s


def test_list_backends_returns_0_without_io_args(capsys):
    """R1+R2：信息类标志 --list-backends 不应要求 --input/--output，
    直接返回 0（此前会被「缺少 --input/--output」拦截而误报）。"""
    rc = ocr_tool.main(["--list-backends"])
    assert rc == 0
    assert "后端依赖" in capsys.readouterr().out


def test_list_formats_returns_0_without_io_args(capsys):
    """回归：--list-formats 同样不应要求 --input/--output。"""
    rc = ocr_tool.main(["--list-formats"])
    assert rc == 0
    assert "支持的" in capsys.readouterr().out


def test_format_version_contains_tool_version():
    """R1 新能力：format_version 返回含版本号的字符串。"""
    text = ocr_tool.format_version()
    assert ocr_tool.TOOL_VERSION in text
    assert "paddleocr-tool" in text


def test_main_version_returns_zero_without_io_args(capsys):
    """R1 新能力：--version 仅打印版本号并退出，无需 --input/--output。"""
    rc = ocr_tool.main(["--version"])
    assert rc == 0
    out_text = capsys.readouterr().out
    assert ocr_tool.TOOL_VERSION in out_text


def test_main_version_does_not_require_input():
    """--version 不应被「缺少 --input」拦截（与 --list-* 信息标志一致）。"""
    # 通过不抛异常 + 返回 0 验证（不依赖 capsys，避免与其它用例耦合）
    import io
    import contextlib

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = ocr_tool.main(["--version"])
    assert rc == 0
    assert ocr_tool.TOOL_VERSION in buf.getvalue()


def test_collect_all_glob_honors_recursive_flag(tmp_path):
    """R2 修复验证：glob 展开应尊重 --recursive（此前硬编码 recursive=True，
    导致 '**' 模式即便未传 --recursive 也会下钻两级子目录）。"""
    (tmp_path / "a.png").write_bytes(b"x")
    sub = tmp_path / "sub" / "deep"
    sub.mkdir(parents=True)
    (sub / "c.png").write_bytes(b"x")
    # 未递归：'**/*.png' 只匹配一级，拿不到两级深的 c.png，也不应拿到根 a.png
    files_off, _ = ocr_tool.collect_all(str(tmp_path / "**/*.png"), recursive=False)
    assert files_off == []
    # 递归：应能找到根 a.png 与两级深 c.png
    files_on, _ = ocr_tool.collect_all(str(tmp_path / "**/*.png"), recursive=True)
    assert {f.name for f in files_on} == {"a.png", "c.png"}


def test_collect_files_ext_without_leading_dot(tmp_path):
    """R2 修复验证：--ext 传 'png'（无前导点）应与 '.png' 等价，不应静默跳过全部。"""
    (tmp_path / "a.png").write_bytes(b"\x89PNG")
    (tmp_path / "b.jpg").write_bytes(b"\xff\xd8")
    # 无前导点
    files, _ = ocr_tool.collect_files(str(tmp_path), recursive=False, exts="png")
    assert {f.name for f in files} == {"a.png"}
    # 混用：'png, .JPG' 都应规整生效
    files2, _ = ocr_tool.collect_files(str(tmp_path), recursive=False, exts="png, .JPG")
    assert {f.name for f in files2} == {"a.png", "b.jpg"}


def test_version_info_structure():
    """R1 新能力：version_info 返回含 version 与 backends 三键的字典。"""
    info = ocr_tool.version_info()
    assert info["version"] == ocr_tool.TOOL_VERSION
    assert set(info["backends"].keys()) == {"paddle", "tesseract", "pdf2image"}


def test_lang_list_data_structure():
    """R1 新能力：lang_list_data 返回结构化语言列表（含后端映射）。"""
    data = ocr_tool.lang_list_data()
    codes = {d["code"] for d in data}
    assert "ch" in codes and "fr" in codes
    fr = next(d for d in data if d["code"] == "fr")
    assert fr["tesseract"] == "fra"  # 结构携带正确引擎代码


def test_main_info_commands_emit_json(capsys):
    """R1 新能力：--json 让信息类命令输出合法 JSON（便于脚本解析）。"""
    import json as _json

    for cmd in (["--version", "--json"], ["--lang-list", "--json"],
                ["--list-formats", "--json"], ["--list-backends", "--json"]):
        rc = ocr_tool.main(cmd)
        assert rc == 0
        out = capsys.readouterr().out
        parsed = _json.loads(out)  # 必须可解析
        assert parsed


def test_main_version_json_contains_version(capsys):
    """--version --json 的 JSON 应含 TOOL_VERSION。"""
    import json as _json

    rc = ocr_tool.main(["--version", "--json"])
    assert rc == 0
    parsed = _json.loads(capsys.readouterr().out)
    assert parsed["version"] == ocr_tool.TOOL_VERSION


def test_max_depth_limits_recursion(tmp_path):
    """R1：--max-depth 限制递归深度，1 只取顶层、不进子目录。"""
    (tmp_path / "top.png").write_bytes(b"x")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "deep.png").write_bytes(b"x")
    sub2 = sub / "nested"
    sub2.mkdir()
    (sub2 / "deeper.png").write_bytes(b"x")

    # max-depth=1：仅顶层 top.png
    files, skipped = ocr_tool.collect_files(str(tmp_path), recursive=True, max_depth=1)
    names = {f.name for f in files}
    assert names == {"top.png"}

    # max-depth=2：顶层 + 一级子目录
    files2, _ = ocr_tool.collect_files(str(tmp_path), recursive=True, max_depth=2)
    names2 = {f.name for f in files2}
    assert names2 == {"top.png", "deep.png"}

    # max-depth=0 / None：不限深度（全量）
    files_all, _ = ocr_tool.collect_files(str(tmp_path), recursive=True, max_depth=0)
    assert {f.name for f in files_all} == {"top.png", "deep.png", "deeper.png"}


def test_dry_run_without_output(tmp_path, capsys):
    """R2 修复：--dry-run 预检不应要求 --output，返回 0 并打印待处理文件统计。"""
    (tmp_path / "a.png").write_bytes(b"x")
    (tmp_path / "b.png").write_bytes(b"x")
    rc = ocr_tool.main(["--input", str(tmp_path), "--dry-run", "--mock"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "dry-run" in out
    assert "2" in out  # 两个待处理文件


def test_collect_all_glob_expands(tmp_path):
    """R1 验证：--input 支持 glob 模式（如 dir/*.png），展开为匹配文件。"""
    (tmp_path / "a.png").write_bytes(b"x")
    (tmp_path / "b.png").write_bytes(b"x")
    (tmp_path / "c.jpg").write_bytes(b"x")
    (tmp_path / "d.txt").write_text("x")
    files, skipped = ocr_tool.collect_all(str(tmp_path / "*.png"), recursive=False)
    names = {f.name for f in files}
    assert names == {"a.png", "b.png"}  # 仅 png 命中 glob
    assert "c.jpg" not in names
    assert "d.txt" not in names


def test_collect_all_glob_no_match_counts_skipped(tmp_path):
    """R2 验证：无匹配的 glob 应计入 skipped（透明），而非静默丢失。"""
    (tmp_path / "a.png").write_bytes(b"x")
    files, skipped = ocr_tool.collect_all(str(tmp_path / "*.pdf"), recursive=False)
    assert files == []
    assert len(skipped) == 1  # 无匹配 glob 计入跳过


def test_main_glob_end_to_end_mock(tmp_path, capsys):
    """R1 端到端验证：以 glob 作为输入跑 mock 批处理，仅命中匹配文件。"""
    out = tmp_path / "out"
    (tmp_path / "a.png").write_bytes(b"x")
    (tmp_path / "b.png").write_bytes(b"x")
    (tmp_path / "c.jpg").write_bytes(b"x")
    rc = ocr_tool.main([
        "--input", str(tmp_path / "*.png"),
        "--output", str(out), "--mock", "--format", "txt",
    ])
    assert rc == 0
    # 仅 png 被处理（summary.txt 也以 .txt 结尾，需排除）
    assert (out / "a.txt").exists()
    assert (out / "b.txt").exists()
    assert not (out / "c.txt").exists()


def test_collect_files_max_size(tmp_path):
    """R1 验证：--max-size 跳过超过体积上限的文件，仅保留较小者。"""
    small = tmp_path / "small.png"
    small.write_bytes(b"x" * 100)
    big = tmp_path / "big.png"
    big.write_bytes(b"x" * 10000)
    files, skipped = ocr_tool.collect_files(
        str(tmp_path), recursive=False, max_size=1000
    )
    names = {f.name for f in files}
    assert names == {"small.png"}
    assert "big.png" in {s.name for s in skipped}


def test_collect_files_max_size_single_file(tmp_path):
    """R2 验证：单文件输入超过 --max-size 应计入跳过（而非被处理）。"""
    big = tmp_path / "big.png"
    big.write_bytes(b"x" * 5000)
    files, skipped = ocr_tool.collect_files(
        str(big), recursive=False, max_size=1000
    )
    assert files == []
    assert len(skipped) == 1


def test_main_max_size_end_to_end_mock(tmp_path, capsys):
    """R1 端到端验证：mock 批处理中 --max-size 仅处理较小的图片。"""
    out = tmp_path / "out"
    (tmp_path / "small.png").write_bytes(b"x" * 100)
    (tmp_path / "big.png").write_bytes(b"x" * 10000)
    rc = ocr_tool.main([
        "--input", str(tmp_path), "--output", str(out),
        "--mock", "--format", "txt", "--max-size", "1000",
    ])
    assert rc == 0
    assert (out / "small.txt").exists()
    assert not (out / "big.txt").exists()


def test_collect_files_multi_include(tmp_path):
    """R1 新需求验证：--include 支持逗号分隔的多个模式（OR 语义）。"""
    (tmp_path / "cover.png").write_bytes(b"x")
    (tmp_path / "page2.png").write_bytes(b"x")
    (tmp_path / "other.jpg").write_bytes(b"x")
    files, _ = ocr_tool.collect_files(str(tmp_path), recursive=False, include="cover,page2")
    assert {f.name for f in files} == {"cover.png", "page2.png"}


def test_write_combined_respects_status_filter(tmp_path):
    """R2 验证：write_combined 的 status_filter 真正生效（可合并非 ok 状态）。

    原实现硬编码 status=='ok'，status_filter 对 empty/filtered 永远无效，
    本测试确认 --combine --status-filter empty 能合并空识别结果、排除 ok。
    """
    out = tmp_path / "out"
    results = [
        {"file": "a.png", "text": "成功文本", "chars": 4, "elapsed": 0.1,
         "status": "ok", "error": "", "avg_conf": 0.9, "min_conf": 0.8},
        {"file": "b.png", "text": "空白内容", "chars": 4, "elapsed": 0.1,
         "status": "empty", "error": "", "avg_conf": None, "min_conf": None},
    ]
    path = ocr_tool.write_combined(results, str(out), "md", status_filter=["empty"])
    assert path is not None
    content = Path(path).read_text(encoding="utf-8")
    assert "空白内容" in content
    assert "成功文本" not in content


def test_sort_files_mtime_newest_first(tmp_path):
    """R1 验证：--sort mtime 按修改时间降序（最新优先）。"""
    import os
    old = tmp_path / "old.png"
    new = tmp_path / "new.png"
    old.write_bytes(b"\x89PNG")
    new.write_bytes(b"\x89PNG")
    # 让 old 明显更早
    os.utime(old, (1000.0, 1000.0))
    os.utime(new, (2000.0, 2000.0))
    out = ocr_tool.sort_files([old, new], "mtime")
    assert out[0] == new  # 最新改动在前
    assert out[1] == old


def test_sort_then_max_files_picks_largest():
    """R2 验证：--sort size 后接 --max-files 截断应得到「体积最大」的 N 个
    （此前 main 先按名称截断再排序，导致 size 优先采样形同虚设）。"""
    class _F:
        def __init__(self, name, size):
            self.name = name
            self._size = size
        def __fspath__(self):
            return self.name
        def stat(self):
            class _S:
                st_size = self._size
                st_mtime = 0.0
            return _S()
    small = _F("a.png", 10)
    big = _F("b.png", 1000)
    mid = _F("c.png", 100)
    # 模拟 main 的「先排序、再截断」
    sorted_files = ocr_tool.sort_files([small, big, mid], "size")
    top2 = sorted_files[:2]
    assert top2 == [big, mid]  # 体积最大的两个，而非按名称截取的 [small, big]


def test_collect_files_exclude(tmp_path):
    """R1 验证：--exclude 命中排除模式的文件被跳过。"""
    (tmp_path / "cover.png").write_bytes(b"x")
    (tmp_path / "page1.png").write_bytes(b"x")
    (tmp_path / "page2.png").write_bytes(b"x")
    files, skipped = ocr_tool.collect_files(str(tmp_path), recursive=False, exclude="cover")
    assert {f.name for f in files} == {"page1.png", "page2.png"}
    assert any(s.name == "cover.png" for s in skipped)


def test_collect_files_multi_exclude_comma(tmp_path):
    """R1 验证：--exclude 支持逗号分隔的多个模式（OR 语义）。"""
    (tmp_path / "cover.png").write_bytes(b"x")
    (tmp_path / "draft.png").write_bytes(b"x")
    (tmp_path / "page1.png").write_bytes(b"x")
    files, _ = ocr_tool.collect_files(str(tmp_path), recursive=False, exclude="cover,draft")
    assert {f.name for f in files} == {"page1.png"}


def test_collect_files_include_then_exclude(tmp_path):
    """R1 验证：--include 先收窄、--exclude 再剔除指定文件。"""
    (tmp_path / "cover_a.png").write_bytes(b"x")
    (tmp_path / "cover_b.png").write_bytes(b"x")
    (tmp_path / "page1.png").write_bytes(b"x")
    files, _ = ocr_tool.collect_files(
        str(tmp_path), recursive=False, include="cover*", exclude="*_b*"
    )
    assert {f.name for f in files} == {"cover_a.png"}


def test_main_low_conf_threshold_zero_no_spurious_warning(tmp_path, capsys):
    """R2 验证：--low-conf-threshold 0 视为关闭，mock 后端下不应打印
    「置信度选项不生效」的虚假告警（此前用 `>= 0` 会在阈值=0 时误报）。"""
    out = tmp_path / "out"
    out.mkdir()
    (tmp_path / "a.png").write_bytes(b"\x89PNG")
    rc = ocr_tool.main([
        "--input", str(tmp_path), "--output", str(out),
        "--mock", "--low-conf-threshold", "0",
    ])
    assert rc == 0
    captured = capsys.readouterr()
    assert "置信度相关选项" not in captured.out
    assert "置信度相关选项" not in captured.err


def test_main_low_conf_threshold_positive_warns_on_mock(tmp_path, capsys):
    """R2 回归：--low-conf-threshold 0.5（真正启用）在 mock 后端下仍应提示
    置信度选项仅 paddle 生效（确保 `> 0` 修复不会误吞真实告警）。"""
    out = tmp_path / "out"
    out.mkdir()
    (tmp_path / "a.png").write_bytes(b"\x89PNG")
    rc = ocr_tool.main([
        "--input", str(tmp_path), "--output", str(out),
        "--mock", "--low-conf-threshold", "0.5",
    ])
    assert rc == 0
    captured = capsys.readouterr()
    assert "置信度相关选项" in (captured.out + captured.err)


def test_main_low_conf_threshold_zero_no_report_written(tmp_path):
    """R2 验证：阈值=0 时不写 low_confidence.txt（视为关闭）。"""
    out = tmp_path / "out"
    out.mkdir()
    (tmp_path / "a.png").write_bytes(b"\x89PNG")
    rc = ocr_tool.main([
        "--input", str(tmp_path), "--output", str(out),
        "--mock", "--low-conf-threshold", "0",
    ])
    assert rc == 0
    assert not (out / "low_confidence.txt").exists()


def test_dedup_lines_ignores_whitespace_only_diff():
    """R2 验证：仅首尾/行内空白不同的同一行也应被去重。

    注意：dedup_lines 仅在「比较」阶段按空白规整（与 normalize_text 口径一致），
    输出保留首次出现的原始行文本（不篡改正文格式），因此下方期望保留原始空白。
    """
    # "公司名称\t" 与 "公司名称" 规整后相同 -> 去重保留首条（原始空白）
    assert ocr_tool.dedup_lines("公司名称\t\n公司名称\n正文") == "公司名称\t\n正文"
    # 行内连续空白规整后与下行相同 -> 去重，保留首次出现的原始行
    assert ocr_tool.dedup_lines("a  b\n a b\nc") == "a  b\nc"


def test_dedup_lines_preserves_separated_duplicates():
    """回归：被空行/其它行隔开的相同行仍保留。"""
    assert ocr_tool.dedup_lines("a\n\nb\na") == "a\n\nb\na"


def test_merge_results_text_skips_errors():
    """R1 验证：merge_results_text 默认跳过 error/空文本，仅合并 ok。"""
    results = [
        {"file": "a.png", "text": "合同第一条", "status": "ok"},
        {"file": "b.png", "text": "", "status": "empty"},
        {"file": "c.png", "text": "[第 1 页识别失败：超时]", "status": "error"},
    ]
    merged = ocr_tool.merge_results_text(results)
    assert "合同第一条" in merged
    assert "识别失败" not in merged          # 错误占位被排除（R2 防污染）
    assert "c.png" not in merged
    assert "a.png" in merged


def test_merge_results_text_include_filtered():
    """R1 验证：include_statuses 可放宽到 filtered/empty（按需）。"""
    results = [
        {"file": "a.png", "text": "x", "status": "ok"},
        {"file": "b.png", "text": "y", "status": "filtered"},
        {"file": "c.png", "text": "", "status": "empty"},
    ]
    merged = ocr_tool.merge_results_text(results, include_statuses=("ok", "filtered"))
    assert "a.png" in merged and "b.png" in merged
    assert "c.png" not in merged            # empty 仍被空文本跳过


def test_merge_results_text_no_mutate():
    """R3 纯度：不修改入参。"""
    results = [{"file": "a.png", "text": "x", "status": "ok"}]
    before = [dict(r) for r in results]
    ocr_tool.merge_results_text(results)
    assert results == before


def test_main_output_dash_pipes_jsonl(tmp_path, capsys):
    """R1 新需求验证：--output - 把每条结果以 JSON 行输出到 stdout（管道友好）；
    R2 验证：诊断/进度信息改走 stderr，stdout 只有纯净 JSONL（无 '===' 横幅污染）。"""
    (tmp_path / "a.png").write_bytes(b"\x89PNG")
    (tmp_path / "b.png").write_bytes(b"\x89PNG")
    rc = ocr_tool.main([
        "--input", str(tmp_path), "--output", "-", "--mock",
    ])
    assert rc == 0
    out = capsys.readouterr().out
    # stdout 不应包含横幅/进度等诊断文本
    assert "===" not in out
    assert "[进度]" not in out
    lines = [ln for ln in out.splitlines() if ln.strip()]
    assert len(lines) == 2
    parsed = [json.loads(ln) for ln in lines]
    assert {Path(r["file"]).name for r in parsed} == {"a.png", "b.png"}
    assert all(r["status"] == "ok" for r in parsed)
    # 确认 cwd 没有被创建字面量 "-" 目录（R2 隐性 bug 修复）
    assert not os.path.exists("-")


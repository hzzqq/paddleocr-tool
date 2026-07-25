"""log_utils 行为测试（R1 新能力：paddleocr-tool 可观测性基础设施）。"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from log_utils import setup_logging, capture_logs
import ocr_tool


def test_setup_logging_writes_to_file(tmp_path):
    log_file = tmp_path / "run.log"
    setup_logging("INFO", log_file=str(log_file))
    logging.getLogger("paddleocr").info("识别开始")
    assert log_file.exists()
    assert "识别开始" in log_file.read_text(encoding="utf-8")


def test_setup_logging_case_insensitive():
    setup_logging("warning")
    assert logging.getLogger().level == logging.WARNING


def test_capture_logs_captures_records():
    with capture_logs() as buf:
        logging.getLogger("t").error("boom")
    assert "boom" in buf.getvalue()
    assert "ERROR" in buf.getvalue()


def test_parse_args_accepts_log_options():
    # R2 验收：旧调用方式不受影响，新 --log-level/--log-file 被接受
    args = ocr_tool.parse_args(["--input", "x", "--log-level", "DEBUG", "--log-file", "x.log"])
    assert args.log_level == "DEBUG"
    assert args.log_file == "x.log"


def test_main_version_writes_log(tmp_path):
    # R1 验收：--version 仍正常退出，且 --log-file 被创建
    log_file = tmp_path / "ocr.log"
    rc = ocr_tool.main(["--version", "--log-file", str(log_file)])
    assert rc == 0
    assert log_file.exists()

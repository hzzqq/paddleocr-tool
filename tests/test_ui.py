"""paddleocr ui.py 接线回归测试（不导入 streamlit，仅静态校验源码）。

运行：pytest paddleocr-tool/tests
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

UI_SRC = (ROOT / "ui.py").read_text(encoding="utf-8")


def test_ui_exposes_combine_and_dryrun():
    """R1 新需求验证：UI 应暴露 --combine 与 --dry-run 开关并传给子进程。"""
    assert '"--combine"' in UI_SRC
    assert '"--dry-run"' in UI_SRC
    assert "combine =" in UI_SRC
    assert "dry_run =" in UI_SRC


def test_ui_defensive_json_parse():
    """R2 隐性健壮性：结果文件解析应有异常兜底，避免 Streamlit 崩溃。"""
    assert "json.JSONDecodeError" in UI_SRC
    assert "结果文件解析失败" in UI_SRC
    # 字段访问应走 .get，避免 KeyError
    assert 'item.get("file"' in UI_SRC


def test_ui_exposes_include_and_quiet():
    """R1 新需求验证：UI 暴露 --include / --quiet 开关并传给子进程。"""
    assert '"--include"' in UI_SRC
    assert '"--quiet"' in UI_SRC
    assert "include =" in UI_SRC
    assert "quiet =" in UI_SRC


def test_ui_warns_min_conf_on_non_paddle():
    """R2 隐性问题验证：tesseract 后端下设置 min_conf 时 UI 给出告警。"""
    assert "最低置信度仅在 paddle 后端生效" in UI_SRC
    assert "backend != \"paddle\"" in UI_SRC

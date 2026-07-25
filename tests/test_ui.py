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


def test_ui_exposes_exclude_normalize_dedup():
    """R1 新需求验证：UI 应暴露后端已支持但此前缺失的 --exclude / --normalize /
    --dedup-lines 开关，并透传给子进程命令。"""
    assert '"--exclude"' in UI_SRC
    assert '"--normalize"' in UI_SRC
    assert '"--dedup-lines"' in UI_SRC
    assert "exclude =" in UI_SRC
    assert "normalize =" in UI_SRC
    assert "dedup_lines =" in UI_SRC


def test_ui_counts_empty_and_filtered():
    """R2 验证：UI 指标卡与徽章应覆盖后端已产出的 empty / filtered 状态，
    而非只统计 ok / skipped_pdf / error（避免空结果/过滤项被标成未知且与产物脱节）。"""
    assert 'd.get("status") == "empty"' in UI_SRC
    assert 'd.get("status") == "filtered"' in UI_SRC
    assert '"empty"' in UI_SRC and '"filtered"' in UI_SRC  # 徽章映射含两类状态
    # 六类指标（成功/跳过/失败/空结果/已过滤/总计）
    assert 'c1, c2, c3, c4, c5, c6' in UI_SRC

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PaddleOCR 批量图文抽取工具 —— Streamlit Web 界面（可选）
==========================================================

通过浏览器选择输入目录 / 输出目录 / 输出格式，点击「开始识别」，
逐文件展示识别结果与下载链接。

依赖：streamlit（见 requirements.txt）
运行：streamlit run ui.py
"""

import json
import subprocess
import sys
from pathlib import Path

import streamlit as st

from ocr_tool import parse_progress  # 进度解析纯函数（DRY + 可单测）

st.set_page_config(page_title="PaddleOCR 批量图文抽取", page_icon="📄")

st.title("📄 PaddleOCR 批量图文抽取工具")
st.caption("支持图片（jpg/png/bmp）与 PDF 批量识别，输出 Markdown / JSON / CSV")

with st.sidebar:
    st.header("参数设置")
    input_dir = st.text_input("输入目录", placeholder="例如 D:/docs/images")
    output_dir = st.text_input("输出目录", placeholder="例如 D:/docs/out")
    lang = st.text_input("识别语言", value="ch")
    fmt = st.selectbox("输出格式", ["md", "json", "csv", "txt"])
    recursive = st.checkbox("递归遍历子目录", value=False)
    backend = st.selectbox("OCR 后端", ["paddle", "tesseract"])
    use_angle = st.checkbox("文字方向分类（--no-angle 取消；纯水平排版关掉可提速）", value=True)
    workers = st.number_input("并行线程数", min_value=1, max_value=8, value=1, step=1)
    min_conf = st.slider("最低置信度（仅 paddle 生效）", 0.0, 1.0, 0.0, 0.05)
    include = st.text_input("文件名过滤（--include，可选）", placeholder="如 *page* 或 封面")
    max_files = st.number_input("最多处理文件数（0=不限制）", min_value=0, max_value=1000, value=0, step=1)
    use_mock = st.checkbox("Mock 模式（无需 OCR 依赖，演示流程）", value=False)
    combine = st.checkbox("合并输出（生成 _combined.md/.txt）", value=False)
    dry_run = st.checkbox("仅预检（统计待处理文件，不执行 OCR）", value=False)
    quiet = st.checkbox("静默模式（--quiet，减少逐文件日志）", value=False)
    exclude = st.text_input("排除文件（--exclude，可选）", placeholder="如 *tmp* 或 草稿")
    normalize = st.checkbox("规整空白（--normalize，折叠换行/去首尾空白）", value=False)
    dedup_lines = st.checkbox("删除连续重复行（--dedup-lines，去页眉水印）", value=False)
    redact_patterns = st.text_input("隐私脱敏正则（--redact-pattern，换行分隔，可选）", placeholder="每行一个正则，如\n\\d{17}[\\dX]\n1[3-9]\\d{9}")

    start = st.button("开始识别", type="primary")

if start:
    # R2 修复（c166）：防止「运行中重复点击」排队第二次完整 OCR（同输出目录
    # 重复处理、_unique_path 竞态）。用 session_state 标记运行态；完成后清除。
    if st.session_state.get("_ocr_running"):
        st.warning("⚠️ 已有识别任务在运行，请等待完成后再点击。")
        st.stop()
    st.session_state["_ocr_running"] = True
    # 隐性问题：最低置信度仅 paddle 后端生效，但 UI 允许在 tesseract 下设置，
    # 用户会误以为已过滤。这里在运行前显式告警，避免「静默失效」的误导。
    if backend != "paddle" and min_conf > 0:
        st.warning("⚠️ 最低置信度仅在 paddle 后端生效；当前选择 tesseract，该选项不会生效。")
    if backend != "paddle" and not use_angle:
        st.warning("⚠️ 文字方向分类（--no-angle）仅 paddle 后端生效；当前选择 tesseract，该选项不会生效。")
    if not input_dir or not output_dir:
        st.error("请先填写输入目录与输出目录。")
    elif not Path(input_dir).exists():
        st.error(f"输入路径不存在：{input_dir}")
    else:
        # 复用 CLI 主程序，避免逻辑重复
        cmd = [
            sys.executable, str(Path(__file__).with_name("ocr_tool.py")),
            "--input", input_dir,
            "--output", output_dir,
            "--lang", lang,
            "--format", fmt,
            "--backend", backend,
            "--workers", str(workers),
            "--min-conf", str(min_conf),
        ]
        if recursive:
            cmd.append("--recursive")
        if include:
            cmd += ["--include", include]
        if max_files and max_files > 0:
            cmd += ["--max-files", str(max_files)]
        if use_mock:
            cmd.append("--mock")
        if combine:
            cmd.append("--combine")
        if dry_run:
            cmd.append("--dry-run")
        if quiet:
            cmd.append("--quiet")
        if exclude:
            cmd += ["--exclude", exclude]
        if normalize:
            cmd.append("--normalize")
        if dedup_lines:
            cmd.append("--dedup-lines")
        if redact_patterns:
            # R2 修复（c166）：正则本身可能含逗号（\d{1,3}、{2,}），原实现
            # 按 split(",") 切割会把 \d{1,3} 拆成 \d{1 与 3} 两段非法正则；
            # redact_text 编译失败仅 log.warning，脱敏静默失效——结果仍含
            # 完整身份证/手机号。改用换行作为分隔符（一行一个正则），与正则
            # 语法不冲突；placeholder 示例相应更新。
            for pat in [p.strip() for p in redact_patterns.splitlines() if p.strip()]:
                cmd += ["--redact-pattern", pat]
        if not use_angle:
            cmd.append("--no-angle")

        log_box = st.empty()
        progress = st.progress(0, text="准备中…")
        lines_seen = []
        # 用 Popen 逐行读取子进程输出，实现真实流式进度（而非一次性阻塞）
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
            # R2 修复（c166）：Windows 默认 GBK；子进程输出含 ✅ 或 emoji 文件名时
            # 编码不一致会抛 UnicodeDecodeError 卡死进度。统一 UTF-8 + errors=replace。
            encoding="utf-8", errors="replace",
        )
        total = 0
        done = 0
        while True:
            line = proc.stdout.readline()
            if not line:
                if proc.poll() is not None:
                    break
                continue
            lines_seen.append(line.rstrip("\n"))
            log_box.code("\n".join(lines_seen[-200:]), language="text")
            prog = parse_progress(line)
            if prog:
                done, total = prog
                if total:
                    progress.progress(min(done / total, 1.0), text=f"识别中 {done}/{total}")
        proc.wait()
        progress.progress(1.0, text="完成")
        st.subheader("运行日志")
        log_box.code("\n".join(lines_seen), language="text")

        if proc.returncode != 0:
            st.error("识别失败，请查看上方日志。")
        else:
            st.success("识别完成！")
            out = Path(output_dir)
            if out.exists():
                rj = out / "results.json"
                if rj.exists():
                    # 防御性解析：结果文件可能为空 / 损坏，避免 Streamlit 崩溃
                    try:
                        data = json.loads(rj.read_text(encoding="utf-8"))
                    except (json.JSONDecodeError, OSError) as e:
                        st.error(f"结果文件解析失败：{e}")
                        data = []
                    if data:
                        ok = sum(1 for d in data if d.get("status") == "ok")
                        skp = sum(1 for d in data if d.get("status") == "skipped_pdf")
                        err = sum(1 for d in data if d.get("status") == "error")
                        emp = sum(1 for d in data if d.get("status") == "empty")
                        flt = sum(1 for d in data if d.get("status") == "filtered")
                        c1, c2, c3, c4, c5, c6 = st.columns(6)
                        c1.metric("成功", ok)
                        c2.metric("跳过", skp)
                        c3.metric("失败", err)
                        c4.metric("空结果", emp)
                        c5.metric("已过滤", flt)
                        c6.metric("总计", len(data))

                        # 状态徽章着色，一眼区分各类结果：
                        # R2 补齐 empty / filtered 两类后端已产出的状态
                        # （此前 UI 只认 ok/skipped_pdf/error，空结果与过滤项
                        # 被标成未知 ⚪，且指标卡不展示，与实际产物口径不一致）。
                        badge = {
                            "ok": "🟢",
                            "skipped_pdf": "🟡",
                            "error": "🔴",
                            "empty": "⚪",
                            "filtered": "🟣",
                        }
                        st.subheader("逐文件结果")
                        for item in data:
                            sts = item.get("status", "")
                            icon = badge.get(sts, "⚪")
                            name = Path(item.get("file", "")).name
                            with st.expander(f"{icon} {name} （{item.get('elapsed', 0)}s / {sts}）"):
                                st.text(item.get("text", "") or "（无识别结果）")

                st.subheader("结果文件下载")
                files = sorted(out.rglob("*")) if recursive else sorted(out.glob("*"))
                for f in files:
                    if f.is_file():
                        with open(f, "rb") as fh:
                            st.download_button(
                                label=f"下载 {f.name}",
                                data=fh.read(),
                                file_name=f.name,
                            )

    # R2 修复（c166）：无论成功/失败都清除运行态标记，允许下一次识别
    st.session_state["_ocr_running"] = False


if __name__ == "__main__":
    import sys
    from streamlit.web.cli import main as _st_main
    sys.argv = ["streamlit", "run", __file__, "--server.port", "8502"]
    _st_main()

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
import re
import subprocess
import sys
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="PaddleOCR 批量图文抽取", page_icon="📄")

st.title("📄 PaddleOCR 批量图文抽取工具")
st.caption("支持图片（jpg/png/bmp）与 PDF 批量识别，输出 Markdown / JSON / CSV")

with st.sidebar:
    st.header("参数设置")
    input_dir = st.text_input("输入目录", placeholder="例如 D:/docs/images")
    output_dir = st.text_input("输出目录", placeholder="例如 D:/docs/out")
    lang = st.text_input("识别语言", value="ch")
    fmt = st.selectbox("输出格式", ["md", "json", "csv"])
    recursive = st.checkbox("递归遍历子目录", value=False)
    backend = st.selectbox("OCR 后端", ["paddle", "tesseract"])
    workers = st.number_input("并行线程数", min_value=1, max_value=8, value=1, step=1)
    min_conf = st.slider("最低置信度（仅 paddle 生效）", 0.0, 1.0, 0.0, 0.05)
    use_mock = st.checkbox("Mock 模式（无需 OCR 依赖，演示流程）", value=False)

    start = st.button("开始识别", type="primary")

if start:
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
        if use_mock:
            cmd.append("--mock")

        log_box = st.empty()
        progress = st.progress(0, text="准备中…")
        lines_seen = []
        # 用 Popen 逐行读取子进程输出，实现真实流式进度（而非一次性阻塞）
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
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
            m = re.search(r"\[进度\].*?(\d+)/(\d+)", line)
            if m:
                done, total = int(m.group(1)), int(m.group(2))
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
                    data = json.loads(rj.read_text(encoding="utf-8"))
                    ok = sum(1 for d in data if d.get("status") == "ok")
                    skp = sum(1 for d in data if d.get("status") == "skipped_pdf")
                    err = sum(1 for d in data if d.get("status") == "error")
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("成功", ok)
                    c2.metric("跳过", skp)
                    c3.metric("失败", err)
                    c4.metric("总计", len(data))

                    # 状态徽章着色，一眼区分 ok / 跳过 / 失败
                    badge = {"ok": "🟢", "skipped_pdf": "🟡", "error": "🔴"}
                    st.subheader("逐文件结果")
                    for item in data:
                        sts = item.get("status", "")
                        icon = badge.get(sts, "⚪")
                        name = Path(item["file"]).name
                        with st.expander(f"{icon} {name} （{item['elapsed']}s / {sts}）"):
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

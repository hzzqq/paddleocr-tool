#!/usr/bin/env bash
# PaddleOCR 批量图文抽取 —— 启动 Streamlit 界面
# CLI 用法见 README；直接跑：python ocr_tool.py --help
# 依赖：pip install streamlit paddleocr paddlepaddle pytesseract pdf2image
streamlit run ui.py --server.port 8502

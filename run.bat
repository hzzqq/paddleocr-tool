@echo off
REM PaddleOCR 批量图文抽取 —— 启动 Streamlit 界面
where python >nul 2>nul || (echo [错误] 未检测到 python，请先安装 Python 并勾选 "Add to PATH"。 & pause & exit /b)
streamlit run ui.py --server.port 8502
pause

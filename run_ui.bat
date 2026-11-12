@echo off
rem Double-click to open the QB Ranker in your browser.
cd /d "%~dp0"
"C:\Users\Will\Field Session\.venv\Scripts\python.exe" -m streamlit run app.py

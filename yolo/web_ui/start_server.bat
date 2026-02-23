@echo off
echo Starting Auto-Label Web Server...
echo.
cd /d F:\dev\bot

REM Activate the virtual environment
call darkorbit_bot\.venv\Scripts\activate.bat

REM Run the server
python yolo\web_ui\autolabel_server.py

pause

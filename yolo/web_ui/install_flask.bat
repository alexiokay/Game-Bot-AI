@echo off
echo Installing Flask web server dependencies...
echo.

echo Installing Flask...
uv pip install flask

echo.
echo Installing Flask-CORS (for cross-origin requests)...
uv pip install flask-cors

echo.
echo Installing Pillow (image handling)...
uv pip install pillow

echo.
echo Done! You can now run the web UI:
echo   python autolabel_server.py
echo   Then open http://localhost:5000 in your browser
echo.
pause

@echo off
cd /d "%~dp0"
echo ============================================
echo   VC Cabin - Face Recognition
echo   Dashboard: http://127.0.0.1:5000
echo   Press Ctrl+C to stop
echo ============================================
where py >nul 2>nul
if %errorlevel%==0 (
    rem Prefer 3.14 if installed (3.15+ may lack the project packages)
    py -3.14 app.py 2>nul || py app.py
) else (
    python app.py
)
pause

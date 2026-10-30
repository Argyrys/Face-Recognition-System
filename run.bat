@echo off
setlocal EnableExtensions
cd /d "%~dp0"

rem ============================================
rem  VC Cabin - Face Recognition
rem  Double-click to install dependencies (first run)
rem  and start the dashboard.
rem ============================================

rem --- 1. find a usable Python ---------------------------------------
rem Prefer 3.14 if installed (3.15+ may lack the project packages).
set "PY="
py -3.14 -c "import sys" >nul 2>nul
if not errorlevel 1 set "PY=py -3.14"
if not defined PY (
    python -c "import sys" >nul 2>nul
    if not errorlevel 1 set "PY=python"
)
if not defined PY (
    py -c "import sys" >nul 2>nul
    if not errorlevel 1 set "PY=py"
)
if not defined PY (
    echo.
    echo  [ERROR] No Python found on this machine.
    echo          Install Python 3.11+ from https://python.org and tick
    echo          "Add python.exe to PATH" during setup, then run this again.
    echo.
    pause
    exit /b 1
)

rem --- 2. install dependencies if missing (first run only) -----------
%PY% -c "import cv2, numpy, flask, waitress" >nul 2>nul
if %errorlevel% neq 0 (
    echo.
    echo  [SETUP] Dependencies missing - installing from requirements.txt ...
    echo          This runs only once and needs internet access.
    echo.
    %PY% -m pip install -r requirements.txt
    if errorlevel 1 (
        echo.
        echo  [ERROR] Dependency install failed. Check your internet connection
        echo          and try again, or run manually:
        echo              %PY% -m pip install -r requirements.txt
        echo.
        pause
        exit /b 1
    )
    rem verify the install actually worked
    %PY% -c "import cv2, numpy, flask, waitress" >nul 2>nul
    if errorlevel 1 (
        echo.
        echo  [ERROR] Install finished but imports still fail - the packages may
        echo          have gone to a different Python. Run manually and check:
        echo              %PY% -m pip install -r requirements.txt
        echo.
        pause
        exit /b 1
    )
    echo  [OK] Dependencies ready.
)

rem --- 3. sanity-check the models (tracked in git, but ZIP exports can omit them)
if not exist "models\face_detection_yunet_2023mar.onnx" goto :missing_models
if not exist "models\face_recognition_sface_2021dec.onnx" goto :missing_models

rem --- 4. launch ------------------------------------------------------
echo.
echo  ============================================
echo    VC Cabin - Face Recognition
echo    Python:    %PY%
echo    Dashboard: http://127.0.0.1:5000
echo    Stop:      Ctrl+C
echo  ============================================
echo.
%PY% app.py %*
set "RC=%errorlevel%"
echo.
echo  [EXIT] app.py stopped (code %RC%).
pause
exit /b %RC%

:missing_models
echo.
echo  [ERROR] Model files are missing from models\.
echo          If you downloaded the project as a ZIP, clone it with git instead:
echo              git clone https://github.com/Argyrys/Face-Recognition-System.git
echo          or download the two .onnx files manually (URLs in README.md).
echo.
pause
exit /b 1

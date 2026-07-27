@echo off
setlocal
cd /d "%~dp0"
title PIIP Setup

if not exist ".deps_installed" (
    cls
    echo.
    echo   ============================================================
    echo    Setting up PIIP for the first time
    echo   ============================================================
    echo.
    echo    PIIP runs entirely on YOUR computer, so the first time it
    echo    opens it needs to install a handful of standard, publicly
    echo    available Python packages - the same ones used every day
    echo    by data scientists and researchers. This is completely
    echo    normal for new software and only happens ONCE.
    echo.
    echo    This needs an internet connection and usually takes a
    echo    couple of minutes. Please don't close this window.
    echo   ============================================================
    echo.
    echo    Installing...
    echo.
    "python-embed\python.exe" -m pip install --no-warn-script-location -q -q -r requirements.txt > setup_log.txt 2>&1
    if errorlevel 1 (
        echo.
        echo   Something went wrong during setup.
        echo   Check your internet connection and try running this again.
        echo   ^(Full details were saved to setup_log.txt if you need them.^)
        echo.
        pause
        exit /b 1
    )
    echo done > ".deps_installed"
    echo    Setup complete!
    echo.
    rem ping, not `timeout` -- timeout needs an interactive console input handle and can fail
    rem in some launch contexts; pinging localhost a few times is a plain time-based pause with
    rem no such dependency, a well-known robust substitute in batch scripts.
    ping -n 3 127.0.0.1 > nul
)

cls
title PIIP - Running
echo.
echo   ============================================================
echo    PIIP is starting...
echo   ============================================================
echo.
echo    This app runs ENTIRELY on YOUR computer. It is NOT a hosted
echo    website - there is no PIIP server anywhere else. Your data
echo    and any API keys you enter stay on this machine.
echo.
echo    Your browser will open automatically to PIIP.
echo    Do NOT close this window while using PIIP - closing it
echo    stops the app.
echo   ============================================================
echo.
"python-embed\python.exe" -m streamlit run app.py

pause

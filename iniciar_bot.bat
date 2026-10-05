@echo off
REM Mantiene el bot de Telegram corriendo: si se cae (sin internet, error), espera 15 s y lo vuelve a iniciar.
cd /d "%~dp0"
:inicio
python bot.py >> bot.log 2>&1
timeout /t 15 /nobreak >nul
goto inicio

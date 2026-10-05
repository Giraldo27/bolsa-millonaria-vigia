@echo off
rem Abre el dashboard de Bolsa Millonaria en el navegador (http://localhost:8501). Cierra esta ventana para apagarlo.
rem Para verlo desde el celular en la misma red wifi, usa la "Network URL" que aparece abajo.
cd /d "%~dp0"
python -m streamlit run app.py --server.headless false --browser.gatherUsageStats false
pause

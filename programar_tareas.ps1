# Programa en Windows las tareas del sistema (ejecútalo UNA vez, como tu usuario):
#     powershell -ExecutionPolicy Bypass -File .\programar_tareas.ps1
# Para ver qué haría sin instalar nada:   ... -Simular
# Para borrar las tareas:                 ... -Quitar
#
#   BM-Monitor : cada 15 min de lunes a viernes entre 08:30 y 16:00 (hora de Bogotá). El propio monitor.py no hace nada si el
#                mercado de trii está cerrado o es festivo, así que el mismo horario sirve para octubre (15:00) y noviembre (16:00).
#   BM-Radar   : lunes a jueves 19:30.
#   BM-Bot     : al iniciar sesión, con reinicio automático si se cae (iniciar_bot.bat).
# El reloj del PC debe estar en hora de Colombia. Si el PC está apagado a esa hora, la tarea se ejecuta al encenderlo (StartWhenAvailable);
# para cubrir el PC apagado usa la nube (GitHub Actions, ver README).
param([switch]$Simular, [switch]$Quitar)

$ErrorActionPreference = "Stop"
$dir = Split-Path -Parent $MyInvocation.MyCommand.Path
$py = (Get-Command python).Source
$tareas = @("BM-Monitor", "BM-Radar", "BM-Bot")

if ($Quitar) {
    foreach ($t in $tareas) { Unregister-ScheduledTask -TaskName $t -Confirm:$false -ErrorAction SilentlyContinue; "Borrada: $t" }
    return
}

$ajustes = New-ScheduledTaskSettingsSet -StartWhenAvailable -WakeToRun -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Minutes 10)
$dias = "Monday", "Tuesday", "Wednesday", "Thursday", "Friday"

function Accion($script) {
    New-ScheduledTaskAction -Execute "cmd.exe" -Argument "/c cd /d `"$dir`" && `"$py`" $script >> alertas.log 2>&1"
}

# 1) Monitor: lunes a viernes 08:30, repetido cada 15 min durante 7 h 30 min
$trigMon = New-ScheduledTaskTrigger -Weekly -DaysOfWeek $dias -At "08:30"
$trigMon.Repetition = (New-ScheduledTaskTrigger -Once -At "08:30" -RepetitionInterval (New-TimeSpan -Minutes 15) -RepetitionDuration (New-TimeSpan -Hours 7 -Minutes 30)).Repetition
# 2) Radar: lunes a jueves 19:30
$trigRad = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday, Tuesday, Wednesday, Thursday -At "19:30"
# 3) Bot: al iniciar sesión (la ventana queda minimizada; el .bat lo reinicia si se cae)
$accBot = New-ScheduledTaskAction -Execute "cmd.exe" -Argument "/c start `"BM-Bot`" /min `"$dir\iniciar_bot.bat`""
$trigBot = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME

$plan = @(
    @{ n = "BM-Monitor"; a = (Accion "monitor.py"); t = $trigMon; d = "Semáforo cada 15 min en horario de bolsa" },
    @{ n = "BM-Radar";   a = (Accion "radar.py");   t = $trigRad; d = "Radar nocturno 19:30 lunes a jueves" },
    @{ n = "BM-Bot";     a = $accBot;               t = $trigBot; d = "Bot de Telegram (al iniciar sesión)" }
)
foreach ($x in $plan) {
    if ($Simular) { "[simulación] $($x.n): $($x.d)"; continue }
    Register-ScheduledTask -TaskName $x.n -Action $x.a -Trigger $x.t -Settings $ajustes -Description $x.d -Force | Out-Null
    "Programada: $($x.n) — $($x.d)"
}
if (-not $Simular) {
    "`nListo. Pruebas rápidas:  Start-ScheduledTask BM-Monitor ; Start-ScheduledTask BM-Radar ; Start-ScheduledTask BM-Bot"
    "Registro de salidas: $dir\alertas.log"
}

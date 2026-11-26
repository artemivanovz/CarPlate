# Создание окружения ANPR-Gate (Windows, Python 3.11/3.12 x64).
param(
    [ValidateSet("auto", "cpu", "cu126")][string]$Device = "auto",
    [string]$Python = ""
)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if ($Python) {
    & $Python "$PSScriptRoot\scripts\setup_env.py" --device $Device
} elseif (Get-Command py -ErrorAction SilentlyContinue) {
    & py -3 "$PSScriptRoot\scripts\setup_env.py" --device $Device
} else {
    & python "$PSScriptRoot\scripts\setup_env.py" --device $Device
}
if ($LASTEXITCODE -ne 0) {
    throw "Setup failed (exit code $LASTEXITCODE). See the error above."
}

# Окно оператора. Камеру и ролик выбирают кнопками в окне.
# Необязательно: .\START.ps1 -Source data\videos\ролик.mp4
param([string]$Source = "")
Set-Location $PSScriptRoot
$pyw = Join-Path $PSScriptRoot ".venv\Scripts\pythonw.exe"
if (-not (Test-Path $pyw)) {
    Write-Host "Не найдено окружение .venv. Один раз выполните .\SETUP.ps1"
    exit 1
}
$args = @("-m", "anprgate", "ui")
if ($Source) { $args += @("--source", $Source) }
Start-Process -FilePath $pyw -ArgumentList $args -WorkingDirectory $PSScriptRoot

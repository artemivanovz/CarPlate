@echo off
rem Окно оператора. Камеру и ролик выбирают кнопками в окне.
rem Работает на любом ПК: папка скрипта, не текущий каталог и не чужой диск.
chcp 65001 >nul
cd /d "%~dp0"

if not exist ".venv\Scripts\pythonw.exe" (
  echo Не найдено окружение .venv в этой папке.
  echo Один раз выполните: powershell -ExecutionPolicy Bypass -File .\SETUP.ps1
  pause
  exit /b 1
)

start "" "%~dp0.venv\Scripts\pythonw.exe" -m anprgate ui

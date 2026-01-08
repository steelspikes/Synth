@echo off
REM Script para activar el entorno virtual

SET ENV_NAME=venv

IF NOT EXIST "%ENV_NAME%\Scripts\activate.bat" (
    echo No se encontro el entorno virtual "%ENV_NAME%".
    echo Asegurate de haberlo creado primero.
    exit /b 1
)

echo Activando entorno virtual "%ENV_NAME%"...
call "%ENV_NAME%\Scripts\activate.bat"

echo Entorno virtual activado.
pause

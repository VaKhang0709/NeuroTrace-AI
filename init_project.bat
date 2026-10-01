@echo off
REM ============================================================================
REM  NeuroTrace AI - Project scaffolding script (Windows)
REM ----------------------------------------------------------------------------
REM  Creates the full directory tree with .gitkeep placeholders so empty folders
REM  survive in Git. The data\ folder is created locally but is fully gitignored.
REM
REM  Usage:
REM      init_project.bat
REM ============================================================================
setlocal
cd /d "%~dp0"

echo.
echo   ================================================
echo    NeuroTrace AI - initializing project structure
echo   ================================================
echo.

REM --- 1) Create every directory ---------------------------------------------
for %%d in (
  "assets\banner"
  "assets\diagrams"
  "assets\logo"
  "data\raw"
  "data\processed"
  "data\external"
  "docs\papers"
  "docs\proposal"
  "docs\references"
  "docs\slides"
  "src\ai_model\notebooks"
  "src\ai_model\scripts"
  "src\ai_model\configs"
  "src\ai_model\checkpoints"
  "src\ai_model\exports"
  "src\backend\app\api"
  "src\backend\app\core"
  "src\backend\app\models"
  "src\backend\app\schemas"
  "src\backend\app\services"
  "src\backend\functions"
  "src\backend\tests"
  "src\frontend\public"
  "src\frontend\src\components\canvas"
  "src\frontend\src\hooks"
  "src\frontend\src\pages"
  "src\frontend\src\services"
  "src\frontend\src\store"
  "src\frontend\src\utils"
  "src\mobile\android"
  "src\mobile\ios"
  "src\mobile\www"
) do (
  if not exist "%%~d" mkdir "%%~d" && echo   [dir] %%~d
)

REM --- 2) .gitkeep placeholders (data\ is skipped - it is fully gitignored) --
for %%d in (
  "assets\banner"
  "assets\diagrams"
  "assets\logo"
  "docs\papers"
  "docs\proposal"
  "docs\references"
  "docs\slides"
  "src\ai_model\notebooks"
  "src\ai_model\scripts"
  "src\ai_model\configs"
  "src\ai_model\checkpoints"
  "src\ai_model\exports"
  "src\backend\app\api"
  "src\backend\app\core"
  "src\backend\app\models"
  "src\backend\app\schemas"
  "src\backend\app\services"
  "src\backend\functions"
  "src\backend\tests"
  "src\frontend\public"
  "src\frontend\src\components\canvas"
  "src\frontend\src\hooks"
  "src\frontend\src\pages"
  "src\frontend\src\services"
  "src\frontend\src\store"
  "src\frontend\src\utils"
  "src\mobile\android"
  "src\mobile\ios"
  "src\mobile\www"
) do (
  if not exist "%%~d\.gitkeep" type nul > "%%~d\.gitkeep"
)

echo.
echo   Done! Directory tree created.
echo.
echo   Next steps:
echo     - Frontend : cd src\frontend ^&^& npm install ^&^& npm run dev
echo     - Backend  : cd src\backend ^&^& python -m venv .venv ^&^& pip install -r requirements.txt
echo     - AI model : cd src\ai_model (Jupyter notebooks ^& training scripts)
echo.
endlocal

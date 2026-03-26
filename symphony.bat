@echo off
setlocal
set "ym=%~1"
set "maxDay=%~2"
if "%ym%"=="" (
    echo Usage: %~nx0 YYYY-MM MAX_DAY
    exit /b 1
)

if "%maxDay%"=="" (
    echo Error: Missing maximum day val.
    echo Usage: %~nx0 YYYY-MM MAX_DAY
    exit /b 1
)

echo Let the symphony begin for %ym% with %maxDay% days

for /L %%i in (1,1,%maxDay%) do (
    setlocal enabledelayedexpansion
    if %%i LSS 10 (
        set "day=0%%i"
    ) else (
        set "day=%%i"
    )
    echo Running proxy_api.py
    python proxy_api.py
    echo Running FetchPrevData.py for %ym%-!day!
    python FetchPrevData.py %ym%-!day!
    echo Running FetchSpecialData.py for %ym%-!day!
    python FetchSpecialData.py %ym%-!day!
    endlocal
)

echo.
echo ============================================================
echo   DATA COLLECTION DONE — Starting processing pipeline
echo ============================================================

python FillData.py
python Predict.py
python FetchRealData.py
python ErrorAnalysis.py
python Display.py

endlocal

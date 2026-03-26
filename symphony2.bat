@echo off
echo "Let the symphony begin"
for /L %%i in (1,1,24) do (
    setlocal enabledelayedexpansion
    if %%i LSS 10 (
        set "day=0%%i"
    ) else (
        set "day=%%i"
    )
    echo Running proxy_api.py
    python proxy_api.py
    echo Running FetchPrevData.py for 2025-05-!day!
    python FetchPrevData.py 2025-05-!day!
    echo Running FetchSpecialData.py for 2025-05-!day!
    python FetchSpecialData.py 2025-05-!day!
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

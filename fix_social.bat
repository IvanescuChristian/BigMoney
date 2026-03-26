@echo off
echo.
echo ============================================================
echo   FIX SOCIAL DATA — Re-extract with corrected slug mapping
echo ============================================================
echo.

setlocal enabledelayedexpansion

echo Scanning historical_hourly/ for CSV files...
set count=0
for %%f in (historical_hourly\*.csv) do (
    set /a count+=1
    set "fname=%%~nf"
    echo [!count!] Re-extracting social for !fname!...
    python FetchSpecialData.py !fname!
)

echo.
echo ============================================================
echo   DONE — Re-extracted social data for !count! days
echo   Now run: python FillData.py
echo ============================================================
pause

@echo off
echo.
echo ============================================================
echo   VERIFY PIPELINE — Predict, Fetch Real, Analyze Errors
echo ============================================================
echo.

echo [1/4] Processing historical data (FillData)...
python FillData.py
if %ERRORLEVEL% NEQ 0 (echo FAILED at FillData.py & pause & exit /b 1)

echo.
echo [2/4] Running 3-stage predictions (Social + Magnitude + Price)...
python Predict.py
if %ERRORLEVEL% NEQ 0 (echo FAILED at Predict.py & pause & exit /b 1)

echo.
echo [3/4] Fetching real prices + social for comparison...
python FetchRealData.py
if %ERRORLEVEL% NEQ 0 (echo FAILED at FetchRealData.py & pause & exit /b 1)

echo.
echo [4/4] Analyzing prediction accuracy...
python ErrorAnalysis.py
if %ERRORLEVEL% NEQ 0 (echo FAILED at ErrorAnalysis.py & pause & exit /b 1)

echo.
echo ============================================================
echo   ALL DONE — Check error_analysis/ folder for results
echo ============================================================
pause

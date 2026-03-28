@echo off
echo ============================================================
echo   MASS COLLECT -- Run and leave it
echo   Safe to close anytime. Resume by running this again.
echo ============================================================
echo.

python mass_collect.py %*

echo.
echo   Done. Run again to continue if interrupted.
pause

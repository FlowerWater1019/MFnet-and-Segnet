@ECHO OFF

PUSHD %~dp0

REM MMIF-CDDFuse
git clone https://github.com/Zhaozixiang1228/MMIF-CDDFuse

REM Dif-Fusion
git clone https://github.com/GeoVectorMatrix/Dif-Fusion

POPD

ECHO Done!
ECHO.

PAUSE

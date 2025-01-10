@ECHO OFF

PUSHD %~dp0

REM MMIF-CDDFuse
git clone https://github.com/Zhaozixiang1228/MMIF-CDDFuse

REM Dif-Fusion
git clone https://github.com/GeoVectorMatrix/Dif-Fusion

REM A2RNet
git clone https://github.com/lok-18/A2RNet

POPD

ECHO Done!
ECHO.

PAUSE

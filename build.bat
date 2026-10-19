@echo off
rem Builds applloopback.exe (per-app audio capture helper). Needs VS 2022 C++ Build Tools.
cd /d "%~dp0"
where cl >nul 2>nul
if %errorlevel%==0 goto build
for %%d in (
  "C:\BuildTools"
  "%ProgramFiles%\Microsoft Visual Studio\2022\BuildTools"
  "%ProgramFiles%\Microsoft Visual Studio\2022\Community"
  "%ProgramFiles%\Microsoft Visual Studio\2022\Professional"
  "%ProgramFiles%\Microsoft Visual Studio\2022\Enterprise"
  "%ProgramFiles(x86)%\Microsoft Visual Studio\2022\BuildTools"
) do if exist "%%~d\Common7\Tools\VsDevCmd.bat" (
  call "%%~d\Common7\Tools\VsDevCmd.bat" -arch=x64 -no_logo
  goto build
)
echo Visual Studio 2022 C++ Build Tools not found. Install "Desktop development with C++".
exit /b 1
:build
cl /nologo /O2 /EHsc /std:c++17 /DUNICODE /D_UNICODE applloopback.cpp /Fe:applloopback.exe /link ole32.lib mmdevapi.lib
del applloopback.obj 2>nul

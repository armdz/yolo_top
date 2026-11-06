@echo off
rem Loads the Visual Studio 2022 x64 environment (any edition or Build Tools).
rem Called with "call" from build.bat / build_cpu.bat (no setlocal: the environment has to stay loaded).
if defined VSCMD_VER exit /b 0
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%VSWHERE%" (
	echo ERROR: Visual Studio 2022 not found. Install Visual Studio or the Build Tools with "Desktop development with C++".
	exit /b 1
)
set "VSDIR="
for /f "usebackq delims=" %%i in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VSDIR=%%i"
if not defined VSDIR (
	echo ERROR: Visual Studio does not have the x64 C++ tools installed.
	exit /b 1
)
set "PATH=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer;%PATH%"
call "%VSDIR%\VC\Auxiliary\Build\vcvars64.bat" >nul || exit /b 1
exit /b 0

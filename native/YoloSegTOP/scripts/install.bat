@echo off
rem install.bat NAME FOLDER  ->  FOLDER\NAME.new.dll becomes FOLDER\NAME.dll
rem Windows can't overwrite a loaded DLL but can rename it: the old one is kept as .old
rem (TouchDesigner keeps using it until restarted; .old files are deleted on the next install).
del /q "%~2\%~1.dll.old*" 2>nul
if exist "%~2\%~1.dll" (
	move /y "%~2\%~1.dll" "%~2\%~1.dll.old" >nul 2>nul || move /y "%~2\%~1.dll" "%~2\%~1.dll.old%RANDOM%" >nul || (
		echo ERROR: could not replace %~2\%~1.dll
		exit /b 1
	)
)
move /y "%~2\%~1.new.dll" "%~2\%~1.dll" >nul || exit /b 1
exit /b 0

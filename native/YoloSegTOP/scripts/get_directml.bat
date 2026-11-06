@echo off
rem Downloads ONNX Runtime with DirectML (GPU on AMD, Intel or NVIDIA) for the portable Windows version
rem and puts it in Documents\Derivative\Plugins\YoloSeg\ (onnxruntime.dll + DirectML.dll).
rem Without it, the portable version uses TouchDesigner's own onnxruntime.dll, which runs on the CPU.
rem To go back to CPU, delete the Plugins\YoloSeg folder.
setlocal
set ORT_VERSION=1.24.4
set DML_VERSION=1.15.4
set DEST=%USERPROFILE%\Documents\Derivative\Plugins\YoloSeg
set TMPD=%TEMP%\yoloseg_directml
if exist "%TMPD%" rmdir /s /q "%TMPD%"
mkdir "%TMPD%\ort" "%TMPD%\dml" || exit /b 1

echo Downloading ONNX Runtime DirectML %ORT_VERSION%...
curl -fsSL -o "%TMPD%\ort.zip" https://api.nuget.org/v3-flatcontainer/microsoft.ml.onnxruntime.directml/%ORT_VERSION%/microsoft.ml.onnxruntime.directml.%ORT_VERSION%.nupkg || goto :fail
echo Downloading DirectML %DML_VERSION%...
curl -fsSL -o "%TMPD%\dml.zip" https://api.nuget.org/v3-flatcontainer/microsoft.ai.directml/%DML_VERSION%/microsoft.ai.directml.%DML_VERSION%.nupkg || goto :fail
tar -xf "%TMPD%\ort.zip" -C "%TMPD%\ort" || goto :fail
tar -xf "%TMPD%\dml.zip" -C "%TMPD%\dml" || goto :fail

if not exist "%DEST%" mkdir "%DEST%"
copy /y "%TMPD%\ort\runtimes\win-x64\native\onnxruntime.dll" "%DEST%\" >nul || goto :fail
copy /y "%TMPD%\dml\bin\x64-win\DirectML.dll" "%DEST%\" >nul || goto :fail
rmdir /s /q "%TMPD%"
echo OK: DirectML installed in %DEST%
echo Restart TouchDesigner. With Device = Auto, the YOLO Seg Track info popup shows "Backend: ONNX Runtime DirectML".
exit /b 0

:fail
echo ERROR: could not download or copy DirectML (if TouchDesigner is open, close it and try again).
exit /b 1

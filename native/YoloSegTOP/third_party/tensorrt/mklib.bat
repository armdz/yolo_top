@echo off
rem Builds the TensorRT import libraries (.lib) from the .def files.
rem The .def files list the exports of nvinfer_10.dll / nvonnxparser_10.dll from TouchDesigner 2025.33230 (TensorRT 10.9).
rem If TD ships another TensorRT version, regenerate the .def files with:
rem   dumpbin /exports "<TD>\bin\nvinfer_10.dll"  (and the same for nvonnxparser_10.dll)
setlocal
call "%~dp0..\..\scripts\vsenv.bat" || exit /b 1
cd /d "%~dp0lib"
for %%N in (nvinfer_10 nvonnxparser_10) do (
	lib /nologo /def:%%N.def /out:%%N.lib /machine:x64 >nul || exit /b 1
)
del /q *.exp 2>nul
echo OK: TensorRT import libraries generated

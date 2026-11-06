@echo off
rem Builds the YOLO Seg Track plugin (YoloSegTOP.dll) and installs it in Documents\Derivative\Plugins.
rem The DLL has both backends and picks one when TouchDesigner loads it:
rem   NVIDIA GPU (RTX 20 or newer) -> CUDA + TensorRT
rem   anything else                -> ONNX Runtime (DirectML if installed, otherwise CPU)
rem Also builds bench.exe and cpu_bench.exe (tests outside TD).
rem Requires Visual Studio 2022 (any edition or Build Tools) and CUDA Toolkit 12.x.
rem TensorRT and ONNX Runtime come from TouchDesigner's own DLLs.
rem   build.bat            build and install
rem   build.bat noinstall  build only (DLLs stay in build\)
setlocal
cd /d "%~dp0"
call scripts\vsenv.bat || exit /b 1
where nvcc >nul 2>nul || (
	echo ERROR: nvcc not found. Install the CUDA Toolkit 12.x.
	exit /b 1
)
if not exist build mkdir build
if not exist third_party\tensorrt\lib\nvinfer_10.lib call third_party\tensorrt\mklib.bat || exit /b 1

rem Turing (RTX 20), Ampere (RTX 30 / A-series), Ada (RTX 40) and, with CUDA 12.8+, Blackwell (RTX 50).
rem PTX for the last architecture lets it run on newer GPUs (compiled when loaded).
set ARCH=-gencode arch=compute_75,code=sm_75 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_89,code=sm_89
set LASTARCH=89
nvcc --list-gpu-arch 2>nul | findstr /x "compute_120" >nul && set LASTARCH=120
if "%LASTARCH%"=="120" set ARCH=%ARCH% -gencode arch=compute_120,code=sm_120
set ARCH=%ARCH% -gencode arch=compute_%LASTARCH%,code=compute_%LASTARCH%

set INC=-Isrc -Isrc\cpu -Itd -Ithird_party\tensorrt\include -Ithird_party\onnxruntime\include
set FLAGS=-O3 -std=c++17 -cudart shared %ARCH% -Xcompiler "/MD /O2 /EHsc /W3 /utf-8" -D_CRT_SECURE_NO_WARNINGS -DYOLOSEG_WITH_CUDA
rem The NVIDIA driver and TensorRT DLLs are loaded only when used: the plugin also loads on PCs without NVIDIA
set DELAY=-Xlinker /DELAYLOAD:nvcuda.dll -Xlinker /DELAYLOAD:nvinfer_10.dll -Xlinker /DELAYLOAD:nvonnxparser_10.dll -ldelayimp
set LIBS=-Lthird_party\tensorrt\lib -lnvinfer_10 -lnvonnxparser_10 -lcuda
set CUDA_SRC=src\Kernels.cu src\Pipeline.cpp src\TrtModel.cpp src\ByteTracker.cpp
set CPU_SRC=src\cpu\CpuPipeline.cpp src\cpu\CpuKernels.cpp src\cpu\OrtModel.cpp

rem Build to temporary names: if TD has the DLL in build\ loaded, it can't be overwritten
nvcc %FLAGS% %INC% -shared -o build\YoloSegTOP.new.dll src\YoloSegPlugin.cpp src\Backend.cpp src\YoloSegTOP.cpp src\cpu\YoloSegTOP_CPU.cpp %CUDA_SRC% %CPU_SRC% %LIBS% %DELAY% || exit /b 1
nvcc %FLAGS% %INC% -o build\bench.exe src\bench.cpp %CUDA_SRC% %LIBS% || exit /b 1
nvcc %FLAGS% %INC% -o build\cpu_bench.exe src\cpu\cpu_bench.cpp %CPU_SRC% src\ByteTracker.cpp || exit /b 1
del /q build\*.exp build\*.lib 2>nul

call scripts\install.bat YoloSegTOP build || exit /b 1
rem optional local extras (not in the repo)
if exist "%~dp0build_extra.bat" call "%~dp0build_extra.bat" %1 || exit /b 1
if "%1"=="noinstall" exit /b 0
set PLUGINS=%USERPROFILE%\Documents\Derivative\Plugins
if not exist "%PLUGINS%" mkdir "%PLUGINS%"
copy /y build\YoloSegTOP.dll "%PLUGINS%\YoloSegTOP.new.dll" >nul || exit /b 1
call scripts\install.bat YoloSegTOP "%PLUGINS%" || exit /b 1
echo OK: installed in %PLUGINS%
echo If TouchDesigner was open, restart it to use the new version.
exit /b 0

# Operadores solo con el backend ONNX Runtime: macOS (CoreML).
# En macOS usar build_mac.sh, que baja ONNX Runtime y llama a este archivo.
# En Windows tambien sirve, sin CUDA (la DLL de build.bat ya trae los dos backends):
#   cmake -B build/cmake -A x64 && cmake --build build/cmake --config Release

cmake_minimum_required(VERSION 3.20)

if(APPLE)
	# binario universal (Apple Silicon + Intel); ONNX Runtime 1.23 pide macOS 13.4
	set(CMAKE_OSX_ARCHITECTURES "arm64;x86_64" CACHE STRING "Architectures")
	set(CMAKE_OSX_DEPLOYMENT_TARGET "13.4" CACHE STRING "Minimum macOS")
endif()

project(YoloSegTOP_Portable LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
	set(CMAKE_BUILD_TYPE Release)
endif()

# carpeta de ONNX Runtime para macOS (la que baja build_mac.sh): se copia dentro del .plugin
set(ORT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/onnxruntime/osx" CACHE PATH "ONNX Runtime for macOS (with lib/libonnxruntime.dylib)")

find_package(Threads REQUIRED)

set(CORE_SRC
	src/cpu/CpuPipeline.cpp
	src/cpu/CpuKernels.cpp
	src/cpu/OrtModel.cpp
	src/ByteTracker.cpp)

add_library(YoloSegTOP MODULE src/YoloSegPlugin.cpp src/cpu/YoloSegTOP_CPU.cpp ${CORE_SRC})
set(PLUGINS YoloSegTOP)
# extras locales opcionales (no estan en el repo)
include("${CMAKE_CURRENT_SOURCE_DIR}/extra.cmake" OPTIONAL)
add_executable(cpu_bench src/cpu/cpu_bench.cpp ${CORE_SRC})

foreach(t ${PLUGINS} cpu_bench)
	target_include_directories(${t} PRIVATE src src/cpu td third_party/onnxruntime/include)
	target_link_libraries(${t} PRIVATE Threads::Threads ${CMAKE_DL_LIBS})
	if(MSVC)
		target_compile_options(${t} PRIVATE /utf-8 /W3 /EHsc)
		target_compile_definitions(${t} PRIVATE _CRT_SECURE_NO_WARNINGS)
	else()
		target_compile_options(${t} PRIVATE -O3 -Wall -Wno-unused-parameter)
	endif()
endforeach()

foreach(t ${PLUGINS})
	set_target_properties(${t} PROPERTIES PREFIX "" CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN ON)
	if(APPLE)
		# TouchDesigner carga plugins de macOS como bundles .plugin
		set_target_properties(${t} PROPERTIES
			BUNDLE TRUE
			BUNDLE_EXTENSION plugin
			MACOSX_BUNDLE_GUI_IDENTIFIER "com.yolotop.${t}"
			MACOSX_BUNDLE_BUNDLE_NAME "${t}")
	endif()
endforeach()

if(APPLE)
	if(EXISTS "${ORT_DIR}/lib/libonnxruntime.dylib")
		# la dylib va dentro del bundle; OrtModel la abre con dlopen desde Contents/Frameworks
		add_custom_command(TARGET YoloSegTOP POST_BUILD
			COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_BUNDLE_CONTENT_DIR:YoloSegTOP>/Frameworks"
			COMMAND cp -fL "${ORT_DIR}/lib/libonnxruntime.dylib" "$<TARGET_BUNDLE_CONTENT_DIR:YoloSegTOP>/Frameworks/libonnxruntime.dylib"
			COMMAND codesign --force --sign - "$<TARGET_BUNDLE_CONTENT_DIR:YoloSegTOP>/Frameworks/libonnxruntime.dylib"
			COMMAND codesign --force --sign - "$<TARGET_BUNDLE_DIR:YoloSegTOP>"
			COMMENT "Copying ONNX Runtime into YoloSegTOP.plugin")
		# cpu_bench la busca al lado del ejecutable
		add_custom_command(TARGET cpu_bench POST_BUILD
			COMMAND cp -fL "${ORT_DIR}/lib/libonnxruntime.dylib" "$<TARGET_FILE_DIR:cpu_bench>/libonnxruntime.dylib")
	else()
		message(WARNING "${ORT_DIR}/lib/libonnxruntime.dylib not found: run build_mac.sh (it downloads it)")
	endif()
endif()

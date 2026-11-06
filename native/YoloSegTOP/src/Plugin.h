// Un solo plugin con dos backends:
//   CUDA + TensorRT (Windows con NVIDIA) -> YoloSegTOP.cpp
//   ONNX Runtime (Mac, Windows sin NVIDIA) -> cpu/YoloSegTOP_CPU.cpp
// YoloSegPlugin.cpp exporta las funciones que pide TouchDesigner y elige el backend
// al cargar el plugin. El backend CUDA solo se compila con YOLOSEG_WITH_CUDA (build.bat).
#pragma once

#include "TOP_CPlusPlusBase.h"

#include <cstdint>

#ifdef _WIN32
#define PLUGIN_EXPORT DLLEXPORT
#else
#define PLUGIN_EXPORT __attribute__((visibility("default")))
#endif

#ifdef YOLOSEG_WITH_CUDA
// true si hay una GPU NVIDIA usable (driver + RTX 20 o mas nueva) y no se forzo el modo portable
// con la variable de entorno YOLOSEG_FORCE_PORTABLE=1. Se decide una vez, al cargar el plugin.
bool useCudaBackend();

TD::TOP_CPlusPlusBase* newYoloSegCuda(const TD::OP_NodeInfo* info, TD::TOP_Context* context);
void deleteYoloSegCuda(TD::TOP_CPlusPlusBase* instance);
bool yoloSegSharedCuda(uint32_t opId, void* out);   // out: YoloSegShared*
#endif

TD::TOP_CPlusPlusBase* newYoloSegCpu(const TD::OP_NodeInfo* info, TD::TOP_Context* context);
void deleteYoloSegCpu(TD::TOP_CPlusPlusBase* instance);
bool yoloSegSharedCpu(uint32_t opId, void* out);    // out: YoloSegSharedCPU*

inline bool
cudaBackend()
{
#ifdef YOLOSEG_WITH_CUDA
	return useCudaBackend();
#else
	return false;
#endif
}

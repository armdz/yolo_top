// Eleccion de backend al cargar el plugin (solo en la build de Windows con CUDA).
#include "Plugin.h"

#include <cuda_runtime.h>

bool
useCudaBackend()
{
	static int result = -1;
	if (result >= 0)
		return result == 1;
	result = 0;

	// para probar el modo portable en una PC con NVIDIA
	char env[8] = {};
	if (GetEnvironmentVariableA("YOLOSEG_FORCE_PORTABLE", env, sizeof(env)) > 0 && env[0] != '0')
		return false;

	// driver de NVIDIA (sin el, nvcuda.dll no existe) y TensorRT de TouchDesigner
	if (!LoadLibraryA("nvcuda.dll") || !LoadLibraryA("nvinfer_10.dll") || !LoadLibraryA("nvonnxparser_10.dll"))
		return false;
	int count = 0;
	if (cudaGetDeviceCount(&count) != cudaSuccess || count <= 0)
		return false;
	cudaDeviceProp prop = {};
	if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess)
		return false;
	// TensorRT 10 y los kernels compilados piden Turing (RTX 20) o mas nuevo
	result = prop.major * 10 + prop.minor >= 75 ? 1 : 0;
	return result == 1;
}

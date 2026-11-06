// Datos del frame que YOLO Seg Track (backend ONNX Runtime) comparte con otros plugins.
// Todo vive en memoria de CPU y es valido hasta el proximo cook del YOLO TOP.
#pragma once

#include "SegTypes.h"

#include <cstddef>
#include <cstdint>
#include <string>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif
#ifdef __APPLE__
#include <mach-o/dyld.h>
#endif

struct YoloSegSharedCPU
{
	uint32_t frame;              // contador de resultados del YOLO TOP
	int count;                   // personas en este frame
	int segment;                 // 1 = hay mascaras (logits), 0 = solo cajas
	int inputW, inputH;          // resolucion del frame procesado
	int modelW, modelH;          // resolucion de entrada de la red
	int protoW, protoH;          // resolucion de los logits de mascara
	const float* logits;         // count x protoH x protoW
	const PersonGPU* persons;    // count cajas (recorte en pixeles del modelo + caja del tracker)
	const PersonOut* info;       // count filas (id, cx, cy, w, h, ..., conf, area)
	const uint8_t* image;        // frame procesado (inputW x inputH, fila 0 = abajo)
	size_t imagePitch;           // bytes por fila de 'image'
	int imageBgra;               // 1 = BGRA, 0 = RGBA
};

// Devuelve false si 'opId' no es un YOLO Seg Track TOP (version ONNX Runtime) vivo.
typedef bool (*YoloSegGetSharedCPUFn)(uint32_t opId, YoloSegSharedCPU* out);

// Busca la funcion en el plugin YOLO Seg Track ya cargado por TouchDesigner.
inline YoloSegGetSharedCPUFn
findYoloSegCPU()
{
	void* sym = nullptr;
#ifdef _WIN32
	if (HMODULE m = GetModuleHandleA("YoloSegTOP.dll"))
		sym = (void*)GetProcAddress(m, "YoloSegGetSharedCPU");
#elif defined(__APPLE__)
	// TD carga cada .plugin por separado: buscar YoloSegTOP.plugin/Contents/MacOS/YoloSegTOP entre las imagenes cargadas
	for (uint32_t i = 0, n = _dyld_image_count(); i < n && !sym; i++)
	{
		const char* name = _dyld_get_image_name(i);
		std::string s = name ? name : "";
		const std::string tail = "/YoloSegTOP";
		if (s.size() < tail.size() || s.compare(s.size() - tail.size(), tail.size(), tail) != 0)
			continue;
		if (void* h = dlopen(name, RTLD_LAZY | RTLD_NOLOAD))
			sym = dlsym(h, "YoloSegGetSharedCPU");
	}
	if (!sym)
		sym = dlsym(RTLD_DEFAULT, "YoloSegGetSharedCPU");
#else
	sym = dlsym(RTLD_DEFAULT, "YoloSegGetSharedCPU");
#endif
	return (YoloSegGetSharedCPUFn)sym;
}

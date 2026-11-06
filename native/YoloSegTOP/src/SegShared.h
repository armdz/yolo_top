// Datos del frame que YoloSegTOP.dll comparte con otros plugins (backend CUDA):
// buscan la funcion exportada YoloSegGetShared en YoloSegTOP.dll.
#pragma once

#include "Kernels.h"
#include <cstdint>

struct PersonOut;

struct YoloSegShared
{
	uint32_t frame;              // contador de cooks del YOLO TOP
	int count;                   // personas en este frame
	int segment;                 // 1 = hay mascaras (logits), 0 = solo cajas
	int inputW, inputH;          // resolucion de la entrada del YOLO TOP
	int modelW, modelH;          // resolucion de entrada de la red
	int protoW, protoH;          // resolucion de los logits de mascara
	const float* logits;         // GPU: count x protoH x protoW (valido hasta el proximo cook del YOLO TOP)
	const PersonGPU* persons;    // CPU: count cajas (recorte en pixeles del modelo + caja del tracker)
	const PersonOut* info;       // CPU: count filas (id, cx, cy, w, h, ..., conf, area)
	const uchar4* image;         // GPU: copia de la entrada del YOLO TOP (inputW x inputH, fila 0 = abajo)
	size_t imagePitch;           // bytes por fila de 'image'
	int imageBgra;               // 1 = BGRA, 0 = RGBA
	cudaEvent_t ready;           // se completa cuando 'image' y 'logits' estan escritos
};

// Devuelve false si 'opId' no es un YOLO Seg Track TOP vivo.
typedef bool (*YoloSegGetSharedFn)(uint32_t opId, YoloSegShared* out);

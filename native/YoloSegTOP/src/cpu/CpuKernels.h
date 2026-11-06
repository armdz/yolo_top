// Version CPU de Kernels.cu (backend ONNX Runtime). Misma matematica que en CUDA.
// Las imagenes son RGBA8/BGRA8 en orientacion TouchDesigner (fila 0 = abajo).
#pragma once

#include "SegTypes.h"

#include <cstddef>
#include <cstdint>
#include <functional>

// Reparte las filas [0, rows) entre los nucleos del CPU
void parallelRows(int rows, const std::function<void(int y0, int y1)>& fn);

// Imagen -> tensor 1x3xHxW RGB 0-1, con resize bilineal (y del tensor desde arriba)
void cpuPreprocess(const uint8_t* img, int W, int H, size_t pitch, bool bgra, float* dst, int dstW, int dstH);

// logits = coef . proto dentro de la caja de recorte (afuera queda muy negativo).
// area = pixeles > 0 dentro de la caja; bounds = { -minX, -minY, maxX, maxY } en pixeles del proto (como en CUDA)
void cpuMaskLogits(const float* proto, int C, int pH, int pW, int modelW, int modelH,
				   const float* coeffs, const PersonGPU& person, float* logits, int& area, int bounds[4]);

// Salida BGRA8 (orientacion TD) combinando mascaras / cajas / imagen de entrada
void cpuCompose(const uint8_t* img, size_t pitch, bool bgra, uint8_t* out, int W, int H,
				const float* logits, int pH, int pW, int modelW, int modelH,
				const PersonGPU* persons, int N, int mode, bool segment, bool soft);

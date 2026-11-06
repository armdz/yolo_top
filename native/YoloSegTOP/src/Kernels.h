#pragma once

#include "SegTypes.h"
#include <cuda_runtime.h>

// Textura TD (BGRA8 o RGBA8, fila 0 = abajo) -> tensor 1x3xHxW RGB 0-1, con resize bilineal.
void launchPreprocess(cudaSurfaceObject_t in, int inW, int inH, bool inBgra,
					  float* dst, int dstW, int dstH, cudaStream_t s);

// logits[n][p] = coef[n] . proto[:, p]; cuenta pixeles > 0 dentro de la caja (area)
// y sus limites en bounds[n*4] = { -minX, -minY, maxX, maxY } en pixeles del proto
// (bounds hay que inicializarlo con memset 0x80 = muy negativo).
void launchMaskLogits(const float* proto, int C, int pH, int pW, int modelW, int modelH,
					  const float* coeffs, const PersonGPU* persons, int N,
					  float* logits, int* areaCount, int* bounds, cudaStream_t s);

// Escribe la salida BGRA8 (orientacion TD) combinando mascaras / cajas / imagen de entrada.
void launchCompose(cudaSurfaceObject_t in, bool inBgra, cudaSurfaceObject_t out, int W, int H,
				   const float* logits, int pH, int pW, int modelW, int modelH,
				   const PersonGPU* persons, int N, int mode, bool segment, bool soft, cudaStream_t s);

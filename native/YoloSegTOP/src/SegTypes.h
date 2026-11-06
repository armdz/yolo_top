// Tipos comunes a los dos backends (CUDA/TensorRT y ONNX Runtime). Sin dependencias de CUDA.
#pragma once

#include <algorithm>
#include <cmath>

struct PersonGPU
{
	float cropX1, cropY1, cropX2, cropY2;   // caja de la deteccion, pixeles del modelo (recorte de mascara)
	float boxX1, boxY1, boxX2, boxY2;       // caja del tracker normalizada 0-1, y desde arriba (dibujo)
	float r, g, b, pad;
	float maskX1, maskY1, maskX2, maskY2;   // limites de la mascara (pixeles reales), normalizado, y desde arriba
};

enum OutputMode { ModeMask = 0, ModeColors = 1, ModeOverlay = 2 };

struct PersonOut
{
	int   id;
	float cx, cy, w, h, x1, y1, x2, y2;   // normalizado 0-1, y desde arriba
	float conf, area;
};

struct PipelineSettings
{
	float confidence = 0.4f;
	bool  segment = true;   // calcular mascaras
	bool  track = true;     // IDs persistentes con ByteTrack (si no, detecciones del frame ordenadas por confianza)
	int   mode = ModeMask;
	bool  soft = true;
	int   trackBuffer = 30;
	float matchThresh = 0.8f;
};

// color estable por ID (tono por razon aurea)
inline void
idColor(int id, float& r, float& g, float& b)
{
	float h = std::fmod(id * 0.618034f, 1.0f) * 6.0f;
	float s = 0.75f, v = 1.0f;
	int i = (int)h;
	float f = h - i, p = v * (1 - s), q = v * (1 - s * f), t = v * (1 - s * (1 - f));
	switch (i % 6)
	{
		case 0: r = v; g = t; b = p; break;
		case 1: r = q; g = v; b = p; break;
		case 2: r = p; g = v; b = t; break;
		case 3: r = p; g = q; b = v; break;
		case 4: r = t; g = p; b = v; break;
		default: r = v; g = p; b = q; break;
	}
}

inline float clamp01(float v) { return std::min(std::max(v, 0.f), 1.f); }

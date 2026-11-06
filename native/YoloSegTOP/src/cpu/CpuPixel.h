// Lectura de pixeles y alpha de mascara, compartidos por los kernels de CPU.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>

namespace cpupixel
{
struct RGB
{
	float r, g, b;
};

inline RGB
readRGB(const uint8_t* img, size_t pitch, int x, int y, bool bgra)
{
	const uint8_t* c = img + (size_t)y * pitch + (size_t)x * 4;
	return bgra ? RGB{ (float)c[2], (float)c[1], (float)c[0] } : RGB{ (float)c[0], (float)c[1], (float)c[2] };
}

inline uint8_t
toByte(float v)
{
	return (uint8_t)(std::min(std::max(v, 0.f), 1.f) * 255.f + 0.5f);
}

// Logit bilineal en (mx, my) pixeles del modelo; con soft, alpha con antialias analitico de ~1.5 px de salida
inline float
maskAlpha(const float* L, int pW, int pH, int modelW, int modelH, float mx, float my, bool soft, float pixelScale)
{
	float px = std::min(std::max(mx * pW / modelW - 0.5f, 0.f), pW - 1.f);
	float py = std::min(std::max(my * pH / modelH - 0.5f, 0.f), pH - 1.f);
	int x0 = (int)px, y0 = (int)py;
	int x1 = std::min(x0 + 1, pW - 1), y1 = std::min(y0 + 1, pH - 1);
	float fx = px - x0, fy = py - y0;
	float v00 = L[y0 * pW + x0], v10 = L[y0 * pW + x1], v01 = L[y1 * pW + x0], v11 = L[y1 * pW + x1];
	float l = v00 * (1 - fx) * (1 - fy) + v10 * fx * (1 - fy) + v01 * (1 - fx) * fy + v11 * fx * fy;
	if (!soft)
		return l > 0.f ? 1.f : 0.f;
	float gx = (v10 - v00) * (1 - fy) + (v11 - v01) * fy;
	float gy = (v01 - v00) * (1 - fx) + (v11 - v10) * fx;
	float g = std::sqrt(gx * gx + gy * gy) * pixelScale + 1e-6f;
	return std::min(std::max(0.5f + l / (1.5f * g), 0.f), 1.f);
}
} // namespace cpupixel

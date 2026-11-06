#include "CpuKernels.h"
#include "CpuPixel.h"

#include <algorithm>
#include <cmath>
#include <thread>
#include <vector>

using namespace cpupixel;

void
parallelRows(int rows, const std::function<void(int y0, int y1)>& fn)
{
	int n = (int)std::min<unsigned>(std::max(std::thread::hardware_concurrency(), 1u), 16u);
	n = std::min(n, std::max(rows / 32, 1));   // no vale la pena repartir imagenes chicas
	if (n <= 1)
	{
		fn(0, rows);
		return;
	}
	std::vector<std::thread> threads;
	threads.reserve(n - 1);
	int chunk = (rows + n - 1) / n;
	for (int i = 1; i < n; i++)
	{
		int y0 = i * chunk, y1 = std::min(rows, y0 + chunk);
		if (y0 < y1)
			threads.emplace_back(fn, y0, y1);
	}
	fn(0, std::min(rows, chunk));
	for (auto& t : threads)
		t.join();
}

void
cpuPreprocess(const uint8_t* img, int inW, int inH, size_t pitch, bool bgra, float* dst, int dstW, int dstH)
{
	const int plane = dstW * dstH;
	const float k = 1.0f / 255.0f;
	parallelRows(dstH, [&](int ya, int yb) {
		for (int y = ya; y < yb; y++)   // y desde arriba
			for (int x = 0; x < dstW; x++)
			{
				float sx = (x + 0.5f) * inW / dstW - 0.5f;
				float sy = (y + 0.5f) * inH / dstH - 0.5f;
				sx = std::min(std::max(sx, 0.f), inW - 1.f);
				sy = std::min(std::max(sy, 0.f), inH - 1.f);
				int x0 = (int)sx, y0 = (int)sy;
				int x1 = std::min(x0 + 1, inW - 1), y1 = std::min(y0 + 1, inH - 1);
				float fx = sx - x0, fy = sy - y0;
				// TD guarda las filas de abajo hacia arriba
				int ty0 = inH - 1 - y0, ty1 = inH - 1 - y1;
				RGB a = readRGB(img, pitch, x0, ty0, bgra), b = readRGB(img, pitch, x1, ty0, bgra);
				RGB c = readRGB(img, pitch, x0, ty1, bgra), d = readRGB(img, pitch, x1, ty1, bgra);
				float w00 = (1 - fx) * (1 - fy), w10 = fx * (1 - fy), w01 = (1 - fx) * fy, w11 = fx * fy;
				int i = y * dstW + x;
				dst[i] = (a.r * w00 + b.r * w10 + c.r * w01 + d.r * w11) * k;
				dst[plane + i] = (a.g * w00 + b.g * w10 + c.g * w01 + d.g * w11) * k;
				dst[2 * plane + i] = (a.b * w00 + b.b * w10 + c.b * w01 + d.b * w11) * k;
			}
	});
}

void
cpuMaskLogits(const float* proto, int C, int pH, int pW, int modelW, int modelH,
			  const float* coeffs, const PersonGPU& pr, float* logits, int& area, int bounds[4])
{
	const int plane = pH * pW;
	const float sx = (float)modelW / pW, sy = (float)modelH / pH;
	std::fill(logits, logits + plane, -100.f);
	area = 0;
	int minX = pW, minY = pH, maxX = -1, maxY = -1;
	// solo hace falta la caja de recorte (+2 px para el bilineal y el gradiente del borde)
	int px0 = std::max((int)std::floor(pr.cropX1 / sx) - 2, 0), px1 = std::min((int)std::ceil(pr.cropX2 / sx) + 2, pW);
	int py0 = std::max((int)std::floor(pr.cropY1 / sy) - 2, 0), py1 = std::min((int)std::ceil(pr.cropY2 / sy) + 2, pH);
	for (int py = py0; py < py1; py++)
		for (int px = px0; px < px1; px++)
		{
			int p = py * pW + px;
			float acc = 0.f;
			for (int c = 0; c < C; c++)
				acc += coeffs[c] * proto[(size_t)c * plane + p];
			logits[p] = acc;
			// area: pixel del proto dentro de la caja con logit > 0 (igual que process_mask de Ultralytics)
			float mx = (px + 0.5f) * sx, my = (py + 0.5f) * sy;
			if (acc > 0.f && mx >= pr.cropX1 && mx < pr.cropX2 && my >= pr.cropY1 && my < pr.cropY2)
			{
				area++;
				minX = std::min(minX, px); minY = std::min(minY, py);
				maxX = std::max(maxX, px); maxY = std::max(maxY, py);
			}
		}
	bounds[0] = -minX; bounds[1] = -minY; bounds[2] = maxX; bounds[3] = maxY;
}

void
cpuCompose(const uint8_t* img, size_t pitch, bool bgra, uint8_t* out, int W, int H,
		   const float* logits, int pH, int pW, int modelW, int modelH,
		   const PersonGPU* persons, int N, int mode, bool segment, bool soft)
{
	const float pixelScale = std::max((float)pW / W, (float)pH / H);
	parallelRows(H, [&](int ya, int yb) {
		for (int yTD = ya; yTD < yb; yTD++)
		{
			int yUp = H - 1 - yTD;
			uint8_t* row = out + (size_t)yTD * W * 4;
			for (int x = 0; x < W; x++)
			{
				float u = (x + 0.5f) / W, v = (yUp + 0.5f) / H;
				float mx = u * modelW, my = v * modelH;
				float best = 0.f;
				int bestN = -1;
				for (int n = 0; n < N; n++)
				{
					const PersonGPU& p = persons[n];
					float a;
					if (segment)
					{
						if (mx < p.cropX1 || mx >= p.cropX2 || my < p.cropY1 || my >= p.cropY2)
							continue;
						a = maskAlpha(logits + (size_t)n * pH * pW, pW, pH, modelW, modelH, mx, my, soft, pixelScale);
					}
					else
						a = (u >= p.boxX1 && u < p.boxX2 && v >= p.boxY1 && v < p.boxY2) ? 1.f : 0.f;
					if (a > best)
					{
						best = a;
						bestN = n;
					}
				}
				RGB col = bestN >= 0 ? RGB{ persons[bestN].r, persons[bestN].g, persons[bestN].b } : RGB{ 0, 0, 0 };
				float r, g, b, a;
				if (mode == ModeMask)
				{
					r = g = b = a = best;
				}
				else if (mode == ModeColors)
				{
					r = col.r * best; g = col.g * best; b = col.b * best; a = best;
				}
				else
				{
					RGB src = img ? readRGB(img, pitch, x, yTD, bgra) : RGB{ 0, 0, 0 };
					const float k = 1.0f / 255.0f;
					float t = 0.5f * best;
					r = src.r * k * (1 - t) + col.r * t;
					g = src.g * k * (1 - t) + col.g * t;
					b = src.b * k * (1 - t) + col.b * t;
					a = 1.f;
					// contorno de las cajas del tracker (2 px)
					float px = (float)x, py = (float)yUp;
					for (int n = 0; n < N; n++)
					{
						const PersonGPU& p = persons[n];
						float x1 = p.boxX1 * W, x2 = p.boxX2 * W, y1 = p.boxY1 * H, y2 = p.boxY2 * H;
						bool inside = px >= x1 - 1 && px <= x2 + 1 && py >= y1 - 1 && py <= y2 + 1;
						bool edge = std::fabs(px - x1) < 1.5f || std::fabs(px - x2) < 1.5f || std::fabs(py - y1) < 1.5f || std::fabs(py - y2) < 1.5f;
						if (inside && edge)
						{
							r = p.r; g = p.g; b = p.b;
						}
					}
				}
				uint8_t* o = row + (size_t)x * 4;
				o[0] = toByte(b);
				o[1] = toByte(g);
				o[2] = toByte(r);
				o[3] = toByte(a);
			}
		}
	});
}

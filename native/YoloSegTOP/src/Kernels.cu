#include "Kernels.h"

namespace
{
__device__ __forceinline__ float3
readRGB(cudaSurfaceObject_t s, int x, int y, bool bgra)
{
	uchar4 c;
	surf2Dread(&c, s, x * 4, y, cudaBoundaryModeClamp);
	return bgra ? make_float3(c.z, c.y, c.x) : make_float3(c.x, c.y, c.z);
}

int divUp(int a, int b) { return (a + b - 1) / b; }

__global__ void
preprocessKernel(cudaSurfaceObject_t in, int inW, int inH, bool bgra, float* dst, int dstW, int dstH)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;   // y desde arriba
	if (x >= dstW || y >= dstH)
		return;
	float sx = (x + 0.5f) * inW / dstW - 0.5f;
	float sy = (y + 0.5f) * inH / dstH - 0.5f;
	sx = fminf(fmaxf(sx, 0.f), inW - 1.f);
	sy = fminf(fmaxf(sy, 0.f), inH - 1.f);
	int x0 = (int)sx, y0 = (int)sy;
	int x1 = min(x0 + 1, inW - 1), y1 = min(y0 + 1, inH - 1);
	float fx = sx - x0, fy = sy - y0;
	// TD guarda las filas de abajo hacia arriba
	int ty0 = inH - 1 - y0, ty1 = inH - 1 - y1;
	float3 a = readRGB(in, x0, ty0, bgra), b = readRGB(in, x1, ty0, bgra);
	float3 c = readRGB(in, x0, ty1, bgra), d = readRGB(in, x1, ty1, bgra);
	float w00 = (1 - fx) * (1 - fy), w10 = fx * (1 - fy), w01 = (1 - fx) * fy, w11 = fx * fy;
	const float k = 1.0f / 255.0f;
	int plane = dstW * dstH, i = y * dstW + x;
	dst[i] = (a.x * w00 + b.x * w10 + c.x * w01 + d.x * w11) * k;
	dst[plane + i] = (a.y * w00 + b.y * w10 + c.y * w01 + d.y * w11) * k;
	dst[2 * plane + i] = (a.z * w00 + b.z * w10 + c.z * w01 + d.z * w11) * k;
}

__global__ void
maskLogitsKernel(const float* proto, int C, int pH, int pW, float sx, float sy,
				 const float* coeffs, const PersonGPU* persons, float* logits, int* areaCount, int* bounds)
{
	int p = blockIdx.x * blockDim.x + threadIdx.x;
	int n = blockIdx.y;
	int plane = pH * pW;
	if (p >= plane)
		return;
	const float* cf = coeffs + n * C;
	float acc = 0.f;
	for (int c = 0; c < C; c++)
		acc += cf[c] * proto[c * plane + p];
	logits[n * plane + p] = acc;

	// area: pixel del proto dentro de la caja con logit > 0 (igual que process_mask de Ultralytics)
	float mx = ((p % pW) + 0.5f) * sx, my = ((p / pW) + 0.5f) * sy;
	const PersonGPU& pr = persons[n];
	if (acc > 0.f && mx >= pr.cropX1 && mx < pr.cropX2 && my >= pr.cropY1 && my < pr.cropY2)
	{
		atomicAdd(&areaCount[n], 1);
		int px = p % pW, py = p / pW;
		atomicMax(&bounds[n * 4 + 0], -px);
		atomicMax(&bounds[n * 4 + 1], -py);
		atomicMax(&bounds[n * 4 + 2], px);
		atomicMax(&bounds[n * 4 + 3], py);
	}
}

__global__ void
composeKernel(cudaSurfaceObject_t in, bool inBgra, cudaSurfaceObject_t out, int W, int H,
			  const float* logits, int pH, int pW, int modelW, int modelH,
			  const PersonGPU* persons, int N, int mode, bool segment, bool soft)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int yTD = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= W || yTD >= H)
		return;
	int yUp = H - 1 - yTD;
	float u = (x + 0.5f) / W, v = (yUp + 0.5f) / H;
	float mx = u * modelW, my = v * modelH;

	float best = 0.f;
	int bestN = -1;
	for (int n = 0; n < N; n++)
	{
		const PersonGPU& p = persons[n];
		float a = 0.f;
		if (segment)
		{
			if (mx < p.cropX1 || mx >= p.cropX2 || my < p.cropY1 || my >= p.cropY2)
				continue;
			float px = mx * pW / modelW - 0.5f, py = my * pH / modelH - 0.5f;
			px = fminf(fmaxf(px, 0.f), pW - 1.f);
			py = fminf(fmaxf(py, 0.f), pH - 1.f);
			int x0 = (int)px, y0 = (int)py;
			int x1 = min(x0 + 1, pW - 1), y1 = min(y0 + 1, pH - 1);
			float fx = px - x0, fy = py - y0;
			const float* L = logits + n * pH * pW;
			float v00 = L[y0 * pW + x0], v10 = L[y0 * pW + x1], v01 = L[y1 * pW + x0], v11 = L[y1 * pW + x1];
			float l = v00 * (1 - fx) * (1 - fy) + v10 * fx * (1 - fy) + v01 * (1 - fx) * fy + v11 * fx * fy;
			if (soft)
			{
				// antialias analitico: borde de ~1.5 px de salida segun el gradiente del logit
				float gx = (v10 - v00) * (1 - fy) + (v11 - v01) * fy;
				float gy = (v01 - v00) * (1 - fx) + (v11 - v10) * fx;
				float g = sqrtf(gx * gx + gy * gy) * fmaxf((float)pW / W, (float)pH / H) + 1e-6f;
				a = fminf(fmaxf(0.5f + l / (1.5f * g), 0.f), 1.f);
			}
			else
				a = l > 0.f ? 1.f : 0.f;
		}
		else
		{
			a = (u >= p.boxX1 && u < p.boxX2 && v >= p.boxY1 && v < p.boxY2) ? 1.f : 0.f;
		}
		if (a > best)
		{
			best = a;
			bestN = n;
		}
	}

	float3 col = bestN >= 0 ? make_float3(persons[bestN].r, persons[bestN].g, persons[bestN].b) : make_float3(0, 0, 0);
	float r, g, b, a;
	if (mode == ModeMask)
	{
		r = g = b = a = best;
	}
	else if (mode == ModeColors)
	{
		r = col.x * best; g = col.y * best; b = col.z * best; a = best;
	}
	else
	{
		float3 src = in ? readRGB(in, x, yTD, inBgra) : make_float3(0, 0, 0);
		const float k = 1.0f / 255.0f;
		float t = 0.5f * best;
		r = src.x * k * (1 - t) + col.x * t;
		g = src.y * k * (1 - t) + col.y * t;
		b = src.z * k * (1 - t) + col.z * t;
		a = 1.f;
		// contorno de las cajas del tracker (2 px)
		float px = (float)x, py = (float)yUp;
		for (int n = 0; n < N; n++)
		{
			const PersonGPU& p = persons[n];
			float x1 = p.boxX1 * W, x2 = p.boxX2 * W, y1 = p.boxY1 * H, y2 = p.boxY2 * H;
			bool inside = px >= x1 - 1 && px <= x2 + 1 && py >= y1 - 1 && py <= y2 + 1;
			bool edge = fabsf(px - x1) < 1.5f || fabsf(px - x2) < 1.5f || fabsf(py - y1) < 1.5f || fabsf(py - y2) < 1.5f;
			if (inside && edge)
			{
				r = p.r; g = p.g; b = p.b;
			}
		}
	}
	uchar4 o;
	o.x = (unsigned char)(fminf(fmaxf(b, 0.f), 1.f) * 255.f + 0.5f);
	o.y = (unsigned char)(fminf(fmaxf(g, 0.f), 1.f) * 255.f + 0.5f);
	o.z = (unsigned char)(fminf(fmaxf(r, 0.f), 1.f) * 255.f + 0.5f);
	o.w = (unsigned char)(fminf(fmaxf(a, 0.f), 1.f) * 255.f + 0.5f);
	surf2Dwrite(o, out, x * 4, yTD, cudaBoundaryModeZero);
}
} // namespace

void
launchPreprocess(cudaSurfaceObject_t in, int inW, int inH, bool inBgra, float* dst, int dstW, int dstH, cudaStream_t s)
{
	dim3 block(16, 16);
	dim3 grid(divUp(dstW, block.x), divUp(dstH, block.y));
	preprocessKernel<<<grid, block, 0, s>>>(in, inW, inH, inBgra, dst, dstW, dstH);
}

void
launchMaskLogits(const float* proto, int C, int pH, int pW, int modelW, int modelH,
				 const float* coeffs, const PersonGPU* persons, int N, float* logits, int* areaCount, int* bounds, cudaStream_t s)
{
	if (N <= 0)
		return;
	dim3 block(256);
	dim3 grid(divUp(pH * pW, block.x), N);
	maskLogitsKernel<<<grid, block, 0, s>>>(proto, C, pH, pW, (float)modelW / pW, (float)modelH / pH,
											coeffs, persons, logits, areaCount, bounds);
}

void
launchCompose(cudaSurfaceObject_t in, bool inBgra, cudaSurfaceObject_t out, int W, int H,
			  const float* logits, int pH, int pW, int modelW, int modelH,
			  const PersonGPU* persons, int N, int mode, bool segment, bool soft, cudaStream_t s)
{
	dim3 block(16, 16);
	dim3 grid(divUp(W, block.x), divUp(H, block.y));
	composeKernel<<<grid, block, 0, s>>>(in, inBgra, out, W, H, logits, pH, pW, modelW, modelH,
										 persons, N, mode, segment, soft);
}

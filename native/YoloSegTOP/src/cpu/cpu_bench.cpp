// Prueba del backend portable fuera de TD: carga una imagen RGBA cruda, corre N frames, mide y guarda la salida.
// uso: cpu_bench <modelo.onnx> <entrada.rgba> <ancho> <alto> <salida.bgra> [iteraciones] [modo 0-2] [task 0 Segmentation/1 Tracking/2 both] [device 0 Auto/1 CPU]
// En Windows, si no hay ONNX Runtime junto al exe, usar: set YOLOSEG_ORT=<carpeta>\onnxruntime.dll

#include "CpuKernels.h"
#include "CpuPipeline.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <thread>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif

int main(int argc, char** argv)
{
	if (argc < 6)
	{
		printf("usage: cpu_bench model.onnx input.rgba width height output.bgra [iters] [mode 0-2] [task 0 Segmentation/1 Tracking/2 both] [device 0 Auto/1 CPU]\n");
		return 1;
	}
#ifdef _WIN32
	// OrtModel busca onnxruntime.dll ya cargada: precargar la que se indique
	if (const char* ort = getenv("YOLOSEG_ORT"))
		if (!LoadLibraryExA(ort, nullptr, LOAD_WITH_ALTERED_SEARCH_PATH))   // DirectML.dll se busca al lado
			printf("could not load %s\n", ort);
#endif
	const char* model = argv[1];
	int W = atoi(argv[3]), H = atoi(argv[4]);
	int iters = argc > 6 ? atoi(argv[6]) : 50;
	PipelineSettings s;
	s.mode = argc > 7 ? atoi(argv[7]) : ModeOverlay;
	int task = argc > 8 ? atoi(argv[8]) : 2;
	s.segment = task != 1;
	s.track = task != 0;
	int device = argc > 9 ? atoi(argv[9]) : 0;

	std::vector<uint8_t> img((size_t)W * H * 4), flipped((size_t)W * H * 4);
	std::ifstream f(argv[2], std::ios::binary);
	f.read((char*)img.data(), img.size());
	for (int y = 0; y < H; y++)   // a orientacion TD (fila 0 = abajo)
		memcpy(&flipped[(size_t)(H - 1 - y) * W * 4], &img[(size_t)y * W * 4], (size_t)W * 4);

	CpuPipeline p;
	auto tl = std::chrono::high_resolution_clock::now();
	p.requestModel(model, device, true);
	while (p.modelLoading())
		std::this_thread::sleep_for(std::chrono::milliseconds(20));
	printf("model load: %.1f s, backend: %s\n",
		   std::chrono::duration<float>(std::chrono::high_resolution_clock::now() - tl).count(), p.backend().c_str());
	if (!p.loadError().empty())
	{
		printf("ERROR: %s\n", p.loadError().c_str());
		return 1;
	}

	std::shared_ptr<const CpuResult> r;
	double totalMs = 0, inferMs = 0, workMs = 0;
	uint64_t seen = 0;
	for (int i = 0; i < iters + 3; i++)
	{
		auto t0 = std::chrono::high_resolution_clock::now();
		while (!p.submit(flipped.data(), W, H, (size_t)W * 4, false, s))
			std::this_thread::sleep_for(std::chrono::microseconds(200));
		while (!(r = p.latest()) || r->serial == seen)
		{
			if (!p.loadError().empty())
			{
				printf("ERROR: %s\n", p.loadError().c_str());
				return 1;
			}
			std::this_thread::sleep_for(std::chrono::microseconds(200));
		}
		seen = r->serial;
		if (i < 3)
			continue;   // calentamiento
		totalMs += std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t0).count();
		inferMs += r->inferMs;
		workMs += r->totalMs;
	}
	printf("%dx%d mode=%d task=%d: %.2f ms/frame (%.0f fps), network %.2f ms, worker %.2f ms\n", W, H, s.mode, task,
		   totalMs / iters, 1000.0 / (totalMs / iters), inferMs / iters, workMs / iters);
	for (auto& o : r->persons)
		printf("  id=%d cx=%.3f cy=%.3f w=%.3f h=%.3f conf=%.2f area=%.4f\n", o.id, o.cx, o.cy, o.w, o.h, o.conf, o.area);

	// salida BGRA, filas de arriba hacia abajo
	std::vector<uint8_t> out((size_t)W * H * 4);
	for (int y = 0; y < H; y++)
		memcpy(&out[(size_t)y * W * 4], &r->output[(size_t)(H - 1 - y) * W * 4], (size_t)W * 4);
	std::ofstream o(argv[5], std::ios::binary);
	o.write((const char*)out.data(), out.size());

	return 0;
}

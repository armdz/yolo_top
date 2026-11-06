// Prueba del pipeline fuera de TouchDesigner: carga una imagen RGBA cruda, corre N frames, mide y guarda la salida.
// uso: bench <modelo.onnx> <entrada.rgba> <ancho> <alto> <salida.rgba> [iteraciones] [modo 0-2] [task 0 Segmentation/1 Tracking/2 both]

#include "Pipeline.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <thread>
#include <vector>

static cudaSurfaceObject_t makeSurface(cudaArray_t a)
{
	cudaResourceDesc d = {};
	d.resType = cudaResourceTypeArray;
	d.res.array.array = a;
	cudaSurfaceObject_t s = 0;
	cudaCreateSurfaceObject(&s, &d);
	return s;
}

int main(int argc, char** argv)
{
	if (argc < 6)
	{
		printf("usage: bench model.onnx input.rgba width height output.bgra [iters] [mode 0-2] [task 0 Segmentation/1 Tracking/2 both]\n");
		return 1;
	}
	const char* model = argv[1];
	int W = atoi(argv[3]), H = atoi(argv[4]);
	int iters = argc > 6 ? atoi(argv[6]) : 200;
	PipelineSettings s;
	s.mode = argc > 7 ? atoi(argv[7]) : ModeOverlay;
	int task = argc > 8 ? atoi(argv[8]) : 2;
	s.segment = task != 1;
	s.track = task != 0;

	std::vector<uchar4> img((size_t)W * H), flipped((size_t)W * H);
	std::ifstream f(argv[2], std::ios::binary);
	f.read((char*)img.data(), img.size() * 4);
	for (int y = 0; y < H; y++)   // a orientacion TD (fila 0 = abajo)
		memcpy(&flipped[(size_t)(H - 1 - y) * W], &img[(size_t)y * W], W * 4);

	cudaChannelFormatDesc fmt = cudaCreateChannelDesc<uchar4>();
	cudaArray_t inArr, outArr;
	cudaMallocArray(&inArr, &fmt, W, H, cudaArraySurfaceLoadStore);
	cudaMallocArray(&outArr, &fmt, W, H, cudaArraySurfaceLoadStore);
	cudaMemcpy2DToArray(inArr, 0, 0, flipped.data(), W * 4, W * 4, H, cudaMemcpyHostToDevice);
	cudaSurfaceObject_t inS = makeSurface(inArr), outS = makeSurface(outArr);
	cudaStream_t stream;
	cudaStreamCreate(&stream);

	Pipeline p;
	auto tl = std::chrono::high_resolution_clock::now();
	p.requestModel(model, true, 0);
	while (p.modelLoading())
		std::this_thread::sleep_for(std::chrono::milliseconds(50));
	printf("model load: %.1f s\n", std::chrono::duration<float>(std::chrono::high_resolution_clock::now() - tl).count());
	if (!p.loadError().empty())
	{
		printf("ERROR: %s\n", p.loadError().c_str());
		return 1;
	}

	std::vector<PersonOut> persons;
	std::string err;
	for (int i = 0; i < 20; i++)
		p.process(inS, W, H, false, outS, s, stream, persons, err);
	double totalMs = 0, inferMs = 0;
	for (int i = 0; i < iters; i++)
	{
		auto t0 = std::chrono::high_resolution_clock::now();
		bool ok = p.process(inS, W, H, false, outS, s, stream, persons, err);
		cudaStreamSynchronize(stream);
		totalMs += std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t0).count();
		inferMs += p.inferMs;
		if (!ok) { printf("ERROR in process: %s\n", err.c_str()); return 1; }
	}
	printf("%dx%d mode=%d task=%d: total %.2f ms/frame (%.0f fps), network %.2f ms\n", W, H, s.mode, task,
		   totalMs / iters, 1000.0 / (totalMs / iters), inferMs / iters);
	for (auto& o : persons)
		printf("  id=%d cx=%.3f cy=%.3f w=%.3f h=%.3f conf=%.2f area=%.4f\n", o.id, o.cx, o.cy, o.w, o.h, o.conf, o.area);

	cudaMemcpy2DFromArray(flipped.data(), W * 4, outArr, 0, 0, W * 4, H, cudaMemcpyDeviceToHost);
	for (int y = 0; y < H; y++)
		memcpy(&img[(size_t)y * W], &flipped[(size_t)(H - 1 - y) * W], W * 4);
	std::ofstream o(argv[5], std::ios::binary);
	o.write((const char*)img.data(), img.size() * 4);

	return 0;
}

// Pipeline completo en GPU: preproceso -> TensorRT -> ByteTrack -> mascaras -> textura de salida.
// No depende de TouchDesigner, asi lo usa tanto el TOP como el programa de prueba.
#pragma once

#include "ByteTracker.h"
#include <cuda.h>
#include "Kernels.h"
#include "TrtModel.h"

#include <atomic>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

class Pipeline
{
public:
	Pipeline();
	~Pipeline();

	// Pide el modelo; si cambia, lo carga/construye en un hilo aparte (no traba TouchDesigner).
	// ctx: contexto CUDA activo de TouchDesigner (el engine tiene que vivir en el mismo contexto)
	void requestModel(const std::string& onnxPath, bool fp16, int cudaDevice, CUcontext ctx = nullptr);
	bool modelReady() const { return myModel != nullptr; }
	bool modelLoading() const { return myLoad->loading; }
	// segundos que lleva construyendo el engine de TensorRT, o -1 si no esta construyendo uno
	double buildSeconds() const;
	std::string loadError();

	void resetTracker() { myResetPending = true; }

	// Procesa un frame. Con el modelo aun cargando, solo escribe la salida (vacia / passthrough).
	bool process(cudaSurfaceObject_t in, int inW, int inH, bool inBgra, cudaSurfaceObject_t out,
				 const PipelineSettings& s, cudaStream_t stream, std::vector<PersonOut>& persons,
				 std::string& err);

	float inferMs = 0.f;   // preproceso + red (GPU)

	// Ultimo frame procesado (mismo orden que 'persons'). Los logits viven en GPU hasta el proximo process().
	struct LastFrame
	{
		std::vector<PersonGPU> persons;
		bool segment = false;
		int modelW = 1, modelH = 1, protoW = 1, protoH = 1;
		const float* logits = nullptr;   // persons.size() x protoH x protoW
	} last;

private:
	static const int kMaxPersons = 64;

	// Estado compartido con el hilo de carga. Es un shared_ptr: si el operador se borra mientras
	// se construye un engine, el hilo sigue (ya cancelado) con su copia y la libera al terminar,
	// asi borrar el operador nunca espera.
	struct LoaderState
	{
		std::mutex mutex;
		std::condition_variable cv;
		bool quit = false;
		bool hasRequest = false;
		bool busy = false;   // el hilo esta cargando un modelo
		std::string reqPath;
		bool reqFp16 = true;
		int reqDevice = 0;
		CUcontext reqCtx = nullptr;
		std::shared_ptr<TrtModel> pending;
		std::string loadError;
		std::atomic<bool> loading{ false };
		std::atomic<bool> cancel{ false };
		std::atomic<double> buildStart{ -1.0 };   // steady_clock en segundos, -1 = no construye
	};

	void freeBuffers();
	static void loaderLoop(std::shared_ptr<LoaderState> st);

	std::shared_ptr<TrtModel> myModel;
	std::shared_ptr<LoaderState> myLoad = std::make_shared<LoaderState>();
	std::thread myLoader;
	std::string myModelKey;

	ByteTracker myTracker;
	std::atomic<bool> myResetPending{ false };

	float* hDet = nullptr;          // pinned
	float* hCoeffs = nullptr;       // pinned
	PersonGPU* hPersons = nullptr;  // pinned
	int* hArea = nullptr;           // pinned
	int* hBounds = nullptr;         // pinned, 4 por persona
	float* dCoeffs = nullptr;
	PersonGPU* dPersons = nullptr;
	float* dLogits = nullptr;
	int* dArea = nullptr;
	int* dBounds = nullptr;
	size_t myDetCapacity = 0;
	size_t myLogitsCapacity = 0;
	size_t myCoeffCapacity = 0;
	cudaEvent_t myEvStart = nullptr, myEvInfer = nullptr;
};

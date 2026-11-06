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
	bool modelLoading() const { return myLoading; }
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

	void freeBuffers();
	void loaderLoop();

	std::shared_ptr<TrtModel> myModel;
	std::shared_ptr<TrtModel> myPending;
	std::mutex myMutex;
	std::condition_variable myCv;
	std::thread myLoader;
	std::atomic<bool> myLoading{ false };
	std::string myLoadError;
	std::string myModelKey;
	std::string myReqPath;
	bool myReqFp16 = true;
	int myReqDevice = 0;
	CUcontext myReqCtx = nullptr;
	bool myHasRequest = false;
	bool myQuit = false;

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

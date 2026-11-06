// Pipeline del backend portable: preproceso -> ONNX Runtime -> ByteTrack -> mascaras -> imagen de salida.
// Todo corre en un hilo propio: TouchDesigner entrega frames con submit() y lee el ultimo resultado con
// latest(), sin esperar nunca a la red. Si el hilo esta ocupado, el frame se descarta.
// No depende de TouchDesigner (lo usa tambien cpu_bench).
#pragma once

#include "ByteTracker.h"
#include "SegTypes.h"

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

class YoloOrtModel;

// Resultado de un frame. Inmutable una vez publicado.
struct CpuResult
{
	uint64_t serial = 0;
	int inW = 0, inH = 0;
	std::vector<uint8_t> image;    // frame usado (RGBA8/BGRA8, fila 0 = abajo, inW*4 por fila)
	bool imageBgra = false;
	std::vector<uint8_t> output;   // salida BGRA8, fila 0 = abajo
	std::vector<PersonOut> persons;
	std::vector<PersonGPU> boxes;  // mismo orden que persons
	bool segment = false;
	int modelW = 1, modelH = 1, protoW = 1, protoH = 1;
	std::vector<float> logits;     // persons.size() x protoH x protoW
	float inferMs = 0.f, totalMs = 0.f;
};

class CpuPipeline
{
public:
	CpuPipeline();
	~CpuPipeline();

	// Pide el modelo; si cambia, lo carga en el hilo de trabajo (device: OrtDevice)
	void requestModel(const std::string& onnxPath, int device, bool fp16);
	bool modelReady() const { return myReady; }
	bool modelLoading() const { return myLoading; }
	std::string loadError();
	std::string backend();

	void resetTracker() { myResetPending = true; }

	// true si el hilo lo acepto (copia la imagen); false si esta ocupado o sin modelo
	bool idle() const { return !myBusy; }
	bool submit(const uint8_t* img, int W, int H, size_t pitch, bool bgra, const PipelineSettings& s);

	std::shared_ptr<const CpuResult> latest();

private:
	static const int kMaxPersons = 64;

	void loop();
	void process(CpuResult& r, const PipelineSettings& s);

	std::thread myThread;
	std::mutex myMutex;
	std::condition_variable myCv;
	bool myQuit = false;

	// pedido de modelo
	bool myHasRequest = false;
	std::string myReqPath, myModelKey;
	int myReqDevice = 0;
	bool myReqFp16 = true;
	std::atomic<bool> myLoading{ false };
	std::atomic<bool> myReady{ false };
	std::string myLoadError, myBackend;
	bool myFrameError = false;   // myLoadError viene de un frame (se borra con el proximo que salga bien)

	// trabajo pendiente
	std::atomic<bool> myBusy{ false };
	bool myHasJob = false;
	std::unique_ptr<CpuResult> myJob;
	PipelineSettings myJobSettings;
	uint64_t mySerial = 0;

	std::shared_ptr<const CpuResult> myLatest;

	// solo los usa el hilo de trabajo
	std::unique_ptr<YoloOrtModel> myModel;
	ByteTracker myTracker;
	std::atomic<bool> myResetPending{ false };
};

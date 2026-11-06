// Modelo YOLO seg (end2end, sin NMS) corriendo en TensorRT.
// Usa las DLL de TensorRT 10.9 que trae TouchDesigner (nvinfer_10.dll, nvonnxparser_10.dll).
#pragma once

#include <cuda_runtime.h>
#include <atomic>
#include <string>

namespace nvinfer1
{
class IRuntime;
class ICudaEngine;
class IExecutionContext;
}

class TrtModel
{
public:
	~TrtModel();

	// Carga el .engine cacheado al lado del .onnx, o lo construye (varios minutos la primera vez).
	// cancel: si pasa a true, el build se corta en el proximo paso. buildStart: se escribe cuando
	// empieza a construir un engine (segundos de steady_clock), para mostrar cuanto lleva.
	bool load(const std::string& onnxPath, bool fp16, std::string& err,
			  const std::atomic<bool>* cancel = nullptr, std::atomic<double>* buildStart = nullptr);
	bool infer(cudaStream_t stream);

	static std::string enginePathFor(const std::string& onnxPath, bool fp16);
	static std::string lastLog();   // ultimo error que reporto TensorRT

	// Entrada: 1x3xHxW float (RGB 0-1)
	int inW = 0, inH = 0;
	// Salida 0: 1 x numDet x detDim  (x1, y1, x2, y2, score, clase, coeficientes de mascara...)
	int numDet = 0, detDim = 0;
	// Salida 1: 1 x protoC x protoH x protoW (prototipos de mascara). protoC = 0 si es modelo de deteccion
	int protoC = 0, protoH = 0, protoW = 0;

	float* dInput = nullptr;
	float* dDet = nullptr;
	float* dProto = nullptr;

private:
	bool buildEngine(const std::string& onnxPath, const std::string& enginePath, bool fp16, std::string& err,
					 const std::atomic<bool>* cancel, std::atomic<double>* buildStart);

	nvinfer1::IRuntime* myRuntime = nullptr;
	nvinfer1::ICudaEngine* myEngine = nullptr;
	nvinfer1::IExecutionContext* myContext = nullptr;
};

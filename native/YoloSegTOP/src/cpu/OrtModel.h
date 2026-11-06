// Modelo YOLO seg (end2end) corriendo en ONNX Runtime. La libreria se carga en tiempo de ejecucion:
//   macOS  : libonnxruntime.dylib dentro del .plugin (Contents/Frameworks), con CoreML (GPU / Neural Engine)
//   Windows: onnxruntime.dll de la subcarpeta YoloSeg junto al plugin (opcional, ej. la version DirectML)
//            o el onnxruntime.dll que trae TouchDesigner (CPU)
#pragma once

#include <string>
#include <vector>

struct OrtSession;
struct OrtValue;

enum OrtDevice { DeviceAuto = 0, DeviceCpu = 1 };

class YoloOrtModel
{
public:
	~YoloOrtModel();

	bool load(const std::string& onnxPath, int device, bool fp16, std::string& err);
	bool infer(std::string& err);

	// Entrada: 1x3xHxW float (RGB 0-1)
	int inW = 0, inH = 0;
	// Salida 0: 1 x numDet x detDim  (x1, y1, x2, y2, score, clase, coeficientes de mascara...)
	int numDet = 0, detDim = 0;
	// Salida 1: 1 x protoC x protoH x protoW. protoC = 0 si es modelo de deteccion
	int protoC = 0, protoH = 0, protoW = 0;

	std::vector<float> input, det, proto;
	std::string backend;   // "CoreML", "DirectML", "CPU"

private:
	OrtSession* mySession = nullptr;
	std::vector<std::string> myInNames, myOutNames;
	std::vector<OrtValue*> myValues;   // entrada + salidas, sobre los vectores de arriba
	int myDetOut = -1, myProtoOut = -1;
};

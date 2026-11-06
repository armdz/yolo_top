#include "TrtModel.h"

#include <NvInfer.h>
#include <NvOnnxParser.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <system_error>
#include <vector>

namespace fs = std::filesystem;

using namespace nvinfer1;

namespace
{
class Logger : public ILogger
{
public:
	void log(Severity severity, const char* msg) noexcept override
	{
		if (severity <= Severity::kWARNING)
		{
			std::lock_guard<std::mutex> lock(myMutex);
			fprintf(stderr, "[TensorRT] %s\n", msg);
			if (severity <= Severity::kERROR)
				lastError = msg;
		}
	}
	std::string get()
	{
		std::lock_guard<std::mutex> lock(myMutex);
		return lastError;
	}
	std::string lastError;

private:
	std::mutex myMutex;
};

Logger& logger()
{
	static Logger l;
	return l;
}

fs::path u8path(const std::string& p)
{
	// TouchDesigner pasa las rutas en UTF-8 (en Windows hace falta para rutas con acentos)
	return fs::u8path(p);
}

bool fileExists(const std::string& p)
{
	std::error_code ec;
	return fs::is_regular_file(u8path(p), ec);
}

// Cancelacion del build del engine: si 'cancel' se pone en true (se borro el operador o se pidio
// otro modelo), TensorRT corta el build en el proximo paso. (Los pasos que reporta no sirven como
// porcentaje: casi todo el tiempo se va en una sola fase sin avisos.)
class BuildMonitor : public IProgressMonitor
{
public:
	explicit BuildMonitor(const std::atomic<bool>* cancel) : myCancel(cancel) {}

	void phaseStart(const char*, const char*, int32_t) noexcept override {}
	bool stepComplete(const char*, int32_t) noexcept override { return !(myCancel && *myCancel); }
	void phaseFinish(const char*) noexcept override {}

private:
	const std::atomic<bool>* myCancel;
};

double
nowSeconds()
{
	return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// true si 'a' es mas nuevo que 'b' (el .onnx se reexporto despues de construir el engine)
bool newerThan(const std::string& a, const std::string& b)
{
	std::error_code ea, eb;
	auto ta = fs::last_write_time(u8path(a), ea);
	auto tb = fs::last_write_time(u8path(b), eb);
	return !ea && !eb && ta > tb;
}
} // namespace

std::string
TrtModel::lastLog()
{
	return logger().get();
}

TrtModel::~TrtModel()
{
	delete myContext;
	delete myEngine;
	delete myRuntime;
	cudaFree(dInput);
	cudaFree(dDet);
	cudaFree(dProto);
}

std::string
TrtModel::enginePathFor(const std::string& onnxPath, bool fp16)
{
	int dev = 0;
	cudaGetDevice(&dev);
	cudaDeviceProp prop{};
	cudaGetDeviceProperties(&prop, dev);
	std::string base = onnxPath;
	size_t dot = base.find_last_of('.');
	if (dot != std::string::npos)
		base = base.substr(0, dot);
	char suffix[128];
	snprintf(suffix, sizeof(suffix), ".sm%d%d.trt%d.%s.engine", prop.major, prop.minor, getInferLibVersion(), fp16 ? "fp16" : "fp32");
	return base + suffix;
}

bool
TrtModel::buildEngine(const std::string& onnxPath, const std::string& enginePath, bool fp16, std::string& err,
					  const std::atomic<bool>* cancel, std::atomic<double>* buildStart)
{
	if (buildStart)
		*buildStart = nowSeconds();
	std::unique_ptr<IBuilder> builder(createInferBuilder(logger()));
	if (!builder) { err = "Could not create the TensorRT builder"; return false; }
	std::unique_ptr<INetworkDefinition> network(builder->createNetworkV2(0U));
	std::unique_ptr<nvonnxparser::IParser> parser(nvonnxparser::createParser(*network, logger()));
	if (!network || !parser) { err = "Could not create the TensorRT network"; return false; }
	// se lee a memoria: parseFromFile no abre rutas con acentos en Windows
	std::vector<char> onnx;
	{
		std::ifstream f(u8path(onnxPath), std::ios::binary | std::ios::ate);
		std::streamoff size = f.good() ? (std::streamoff)f.tellg() : -1;
		if (size <= 0) { err = "Could not read " + onnxPath; return false; }
		onnx.resize((size_t)size);
		f.seekg(0);
		f.read(onnx.data(), onnx.size());
		if (!f.good()) { err = "Could not read " + onnxPath; return false; }
	}
	if (!parser->parse(onnx.data(), onnx.size()))
	{
		err = "Error parsing the ONNX: ";
		for (int32_t i = 0; i < parser->getNbErrors(); i++)
			err += std::string(parser->getError(i)->desc()) + " ";
		return false;
	}
	std::unique_ptr<IBuilderConfig> config(builder->createBuilderConfig());
	config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, 1ULL << 30);
	if (fp16)
		config->setFlag(BuilderFlag::kFP16);
	BuildMonitor monitor(cancel);
	config->setProgressMonitor(&monitor);
	std::unique_ptr<IHostMemory> plan(builder->buildSerializedNetwork(*network, *config));
	if (cancel && *cancel) { err = "Engine build cancelled"; return false; }
	if (!plan) { err = "TensorRT could not build the engine: " + logger().get(); return false; }

	// se escribe a un temporal y se renombra: si algo falla no queda un engine a medias
	fs::path tmp = u8path(enginePath + ".tmp");
	{
		std::ofstream f(tmp, std::ios::binary);
		f.write(static_cast<const char*>(plan->data()), plan->size());
		if (!f.good())
		{
			f.close();
			std::error_code ec;
			fs::remove(tmp, ec);
			err = "Could not save " + enginePath + " (the model folder must be writable)";
			return false;
		}
	}
	std::error_code ec;
	fs::rename(tmp, u8path(enginePath), ec);
	if (ec)
	{
		fs::remove(tmp, ec);
		err = "Could not save " + enginePath;
		return false;
	}
	return true;
}

bool
TrtModel::load(const std::string& onnxPath, bool fp16, std::string& err, const std::atomic<bool>* cancel,
			   std::atomic<double>* buildStart)
{
	if (!fileExists(onnxPath)) { err = "Model not found: " + onnxPath; return false; }

	std::string enginePath = onnxPath;
	bool isEngine = enginePath.size() > 7 && enginePath.substr(enginePath.size() - 7) == ".engine";
	if (!isEngine)
	{
		enginePath = enginePathFor(onnxPath, fp16);
		bool stale = fileExists(enginePath) && newerThan(onnxPath, enginePath);
		if ((!fileExists(enginePath) || stale) && !buildEngine(onnxPath, enginePath, fp16, err, cancel, buildStart))
			return false;
	}

	std::vector<char> data;
	{
		std::ifstream f(u8path(enginePath), std::ios::binary | std::ios::ate);
		std::streamoff size = f.good() ? (std::streamoff)f.tellg() : -1;
		if (size <= 0) { err = "Could not read " + enginePath; return false; }
		data.resize((size_t)size);
		f.seekg(0);
		f.read(data.data(), data.size());
		if (!f.good()) { err = "Could not read " + enginePath; return false; }
	}

	myRuntime = createInferRuntime(logger());
	myEngine = myRuntime ? myRuntime->deserializeCudaEngine(data.data(), data.size()) : nullptr;
	if (!myEngine) { err = "Could not load the engine (delete " + enginePath + " to rebuild it): " + logger().get(); return false; }
	myContext = myEngine->createExecutionContext();
	if (!myContext) { err = "Could not create the TensorRT context"; return false; }

	for (int32_t i = 0; i < myEngine->getNbIOTensors(); i++)
	{
		const char* name = myEngine->getIOTensorName(i);
		Dims d = myEngine->getTensorShape(name);
		if (myEngine->getTensorDataType(name) != DataType::kFLOAT)
		{
			err = std::string("Tensor ") + name + " is not float32: export the ONNX with half=False (TensorRT does the FP16)";
			return false;
		}
		size_t count = 1;
		for (int32_t k = 0; k < d.nbDims; k++)
		{
			if (d.d[k] <= 0)
			{
				err = "The model has a dynamic shape: export it with dynamic=False and a fixed imgsz, e.g. imgsz=(384, 640)";
				return false;
			}
			count *= (size_t)d.d[k];
		}
		float** dst = nullptr;
		if (myEngine->getTensorIOMode(name) == TensorIOMode::kINPUT)
		{
			if (d.nbDims != 4) { err = "Unexpected input tensor (expected 1x3xHxW)"; return false; }
			inH = (int)d.d[2];
			inW = (int)d.d[3];
			dst = &dInput;
		}
		else if (d.nbDims == 3)
		{
			numDet = (int)d.d[1];
			detDim = (int)d.d[2];
			dst = &dDet;
		}
		else if (d.nbDims == 4)
		{
			protoC = (int)d.d[1];
			protoH = (int)d.d[2];
			protoW = (int)d.d[3];
			dst = &dProto;
		}
		else
			continue;
		if (cudaMalloc((void**)dst, count * sizeof(float)) != cudaSuccess)
		{
			*dst = nullptr;
			err = "Out of GPU memory for the model";
			return false;
		}
		myContext->setTensorAddress(name, *dst);
	}
	// end2end (YOLO26 por defecto): 1 x 300 x (6 + 32). Sin end2end (YOLOv8/11): 1 x 116 x 8400
	if (!dInput || !dDet || detDim < 6 || detDim > numDet)
	{
		err = "Unexpected model format (expected YOLO seg end2end, output 1x300x38). "
			  "Use a YOLO26 seg model exported with format='onnx' (end2end is on by default).";
		return false;
	}
	if (dProto && detDim < 6 + protoC)
		protoC = 0;
	return true;
}

bool
TrtModel::infer(cudaStream_t stream)
{
	return myContext->enqueueV3(stream);
}

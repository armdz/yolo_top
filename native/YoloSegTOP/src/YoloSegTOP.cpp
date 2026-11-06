// YOLO Seg Track TOP - operador nativo (CUDA + TensorRT) para TouchDesigner.
//
// Entrada : textura 8 bit (RGBA8 o BGRA8)
// Salida  : mascara de personas / colores por ID / overlay, misma resolucion que la entrada
// Info DAT: id, cx, cy, w, h, x1, y1, x2, y2, conf, area  (normalizado 0-1, y desde arriba)
// Info CHOP: persons, infer_ms, total_ms, model_ready + p<i>_id/cx/cy/w/h por persona

#include "TOP_CPlusPlusBase.h"
#include "Plugin.h"
#include "OpCommon.h"
#include "Pipeline.h"
#include "SegShared.h"

#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

using namespace TD;

namespace
{
void
setupSurface(cudaSurfaceObject_t* surface, cudaArray_t array)
{
	if (*surface)
	{
		cudaResourceDesc desc;
		cudaGetSurfaceObjectResourceDesc(&desc, *surface);
		if (desc.resType != cudaResourceTypeArray || desc.res.array.array != array)
		{
			cudaDestroySurfaceObject(*surface);
			*surface = 0;
		}
	}
	if (!*surface && array)
	{
		cudaResourceDesc desc = {};
		desc.resType = cudaResourceTypeArray;
		desc.res.array.array = array;
		cudaCreateSurfaceObject(surface, &desc);
	}
}
} // namespace

class YoloSegTOP;

// Registro de instancias por opId, para que otros plugins encuentren los datos del frame
static std::mutex gRegistryMutex;
static std::unordered_map<uint32_t, YoloSegTOP*> gRegistry;

class YoloSegTOP : public TOP_CPlusPlusBase
{
public:
	YoloSegTOP(const OP_NodeInfo* info, TOP_Context* context) : myContext(context), myOpId(info->opId)
	{
		cudaStreamCreate(&myStream);
		std::lock_guard<std::mutex> lock(gRegistryMutex);
		gRegistry[myOpId] = this;
	}

	~YoloSegTOP() override
	{
		{
			std::lock_guard<std::mutex> lock(gRegistryMutex);
			gRegistry.erase(myOpId);
		}
		myPipeline.reset();
		cudaFree(myFrameCopy);
		if (myReady) cudaEventDestroy(myReady);
		if (myInputSurface) cudaDestroySurfaceObject(myInputSurface);
		if (myOutputSurface) cudaDestroySurfaceObject(myOutputSurface);
		if (myStream) cudaStreamDestroy(myStream);
	}

	void getGeneralInfo(TOP_GeneralInfo* ginfo, const OP_Inputs*, void*) override
	{
		ginfo->cookEveryFrameIfAsked = true;
	}

	void execute(TOP_Output* output, const OP_Inputs* inputs, void*) override
	{
		auto t0 = std::chrono::high_resolution_clock::now();
		myError.clear();
		myWarning.clear();
		opcommon::yoloEnablePars(inputs, true);

		if (inputs->getNumInputs() < 1)
		{
			clearFrame();
			myError = "Connect an input (camera / video).";
			return;
		}
		const OP_TOPInput* top = inputs->getInputTOP(0);
		OP_PixelFormat fmt = top->textureDesc.pixelFormat;
		if (fmt != OP_PixelFormat::BGRA8Fixed && fmt != OP_PixelFormat::RGBA8Fixed)
		{
			clearFrame();
			myError = "The input must be 8-bit RGBA (set Pixel Format = 8-bit fixed (RGBA) on the previous TOP).";
			return;
		}
		const bool inBgra = fmt == OP_PixelFormat::BGRA8Fixed;
		const int W = top->textureDesc.width, H = top->textureDesc.height;

		// --- leer parametros (antes de beginCUDAOperations) ---
		PipelineSettings s = opcommon::readYoloSettings(inputs);
		const char* modelPath = inputs->getParFilePath("Modelfile");
		const bool hasModel = modelPath && *modelPath;
		bool fp16 = inputs->getParInt("Fp16") != 0;

		OP_CUDAAcquireInfo acquire;
		acquire.stream = myStream;
		const OP_CUDAArrayInfo* inArray = top->getCUDAArray(acquire, nullptr);

		TOP_CUDAOutputInfo outInfo;
		outInfo.textureDesc.width = W;
		outInfo.textureDesc.height = H;
		outInfo.textureDesc.texDim = OP_TexDim::e2D;
		outInfo.textureDesc.pixelFormat = OP_PixelFormat::BGRA8Fixed;
		outInfo.stream = myStream;
		const OP_CUDAArrayInfo* outArray = output->createCUDAArray(outInfo, nullptr);
		if (!inArray || !outArray)
		{
			clearFrame();
			return;
		}

		if (!myContext->beginCUDAOperations(nullptr))
			return;
		// todo lo de CUDA (incluido crear el pipeline y pedir el modelo) dentro de begin/end,
		// asi usa el mismo contexto CUDA que TouchDesigner
		CUcontext ctx = nullptr;
		cuCtxGetCurrent(&ctx);
		if (!myPipeline)
			myPipeline = std::make_unique<Pipeline>();
		if (hasModel)
			myPipeline->requestModel(modelPath, fp16, myContext->getCUDADeviceIndex(nullptr), ctx);
		setupSurface(&myInputSurface, inArray->cudaArray);
		setupSurface(&myOutputSurface, outArray->cudaArray);
		std::string procErr;
		bool ok = myPipeline->process(myInputSurface, W, H, inBgra, myOutputSurface, s, myStream, myPersons, procErr);
		if (ok)
			ok = copyFrame(inArray->cudaArray, W, H, inBgra, procErr);
		myContext->endCUDAOperations(nullptr);

		std::string loadErr = myPipeline->loadError();
		if (!ok)
			myError = procErr;
		else if (!hasModel)
			myError = opcommon::noModelError();
		else if (myPipeline->modelLoading())
			myWarning = "Preparing TensorRT... the first time it builds the engine (can take several minutes), then it loads in seconds.";
		else if (!loadErr.empty())
			myError = loadErr;

		myFrame++;
		myInW = W;
		myInH = H;
		myInferMs = myPipeline->inferMs;
		myTotalMs = std::chrono::duration<float, std::milli>(std::chrono::high_resolution_clock::now() - t0).count();
	}

	int32_t getNumInfoCHOPChans(void*) override
	{
		return opcommon::yoloInfoChans(myPersons);
	}

	void getInfoCHOPChan(int32_t index, OP_InfoCHOPChan* chan, void*) override
	{
		opcommon::yoloInfoChan(index, chan, myPersons, myInferMs, myTotalMs, myPipeline && myPipeline->modelReady());
	}

	bool getInfoDATSize(OP_InfoDATSize* infoSize, void*) override
	{
		opcommon::yoloInfoDATSize(infoSize, myPersons);
		return true;
	}

	void getInfoDATEntries(int32_t index, int32_t, OP_InfoDATEntries* entries, void*) override
	{
		opcommon::yoloInfoDATEntries(index, entries, myPersons);
	}

	void getErrorString(OP_String* error, void*) override
	{
		if (!myError.empty())
			error->setString(myError.c_str());
	}

	void getWarningString(OP_String* warning, void*) override
	{
		if (!myWarning.empty())
			warning->setString(myWarning.c_str());
	}

	void getInfoPopupString(OP_String* info, void*) override
	{
		char buf[160];
		snprintf(buf, sizeof(buf), "Backend: CUDA + TensorRT\nPersons: %d\nNetwork (GPU): %.2f ms\nTotal TOP: %.2f ms",
				 (int)myPersons.size(), myInferMs, myTotalMs);
		info->setString(buf);
	}

	void setupParameters(OP_ParameterManager* manager, void*) override
	{
		opcommon::setupYoloParams(manager);
	}

	void pulsePressed(const char* name, void*) override
	{
		if (!strcmp(name, "Resettracker") && myPipeline)
			myPipeline->resetTracker();
	}

	bool getShared(YoloSegShared* out) const
	{
		if (!myPipeline || !myHasFrame)
			return false;
		const Pipeline::LastFrame& l = myPipeline->last;
		out->frame = myFrame;
		out->count = (int)std::min(l.persons.size(), myPersons.size());
		out->segment = l.segment && l.logits ? 1 : 0;
		out->inputW = myInW;
		out->inputH = myInH;
		out->modelW = l.modelW;
		out->modelH = l.modelH;
		out->protoW = l.protoW;
		out->protoH = l.protoH;
		out->logits = l.logits;
		out->persons = l.persons.data();
		out->info = myPersons.data();
		out->image = myFrameCopy;
		out->imagePitch = myFramePitch;
		out->imageBgra = myFrameBgra ? 1 : 0;
		out->ready = myReady;
		return true;
	}

private:
	// Sin entrada valida: no dejar personas viejas en la tabla ni en los datos compartidos
	void clearFrame()
	{
		myPersons.clear();
		myHasFrame = false;
	}

	// Copia la entrada (GPU->GPU) para los datos compartidos (RGB del frame procesado)
	bool copyFrame(cudaArray_t in, int W, int H, bool bgra, std::string& err)
	{
		myHasFrame = false;
		if (!myReady)
			cudaEventCreateWithFlags(&myReady, cudaEventDisableTiming);
		if (!myFrameCopy || W != myFrameW || H != myFrameH)
		{
			cudaFree(myFrameCopy);
			myFrameCopy = nullptr;
			myFrameW = myFrameH = 0;
			if (cudaMallocPitch((void**)&myFrameCopy, &myFramePitch, (size_t)W * 4, H) != cudaSuccess)
			{
				myFrameCopy = nullptr;
				err = "Out of GPU memory for the frame copy";
				return false;
			}
			myFrameW = W;
			myFrameH = H;
		}
		myFrameBgra = bgra;
		cudaError_t e = cudaMemcpy2DFromArrayAsync(myFrameCopy, myFramePitch, in, 0, 0, (size_t)W * 4, H,
												   cudaMemcpyDeviceToDevice, myStream);
		if (e == cudaSuccess)
			e = cudaEventRecord(myReady, myStream);
		if (e != cudaSuccess)
		{
			err = std::string("frame copy: ") + cudaGetErrorName(e) + " - " + cudaGetErrorString(e);
			return false;
		}
		myHasFrame = true;
		return true;
	}

private:
	TOP_Context* myContext;
	uint32_t myOpId;
	uchar4* myFrameCopy = nullptr;
	size_t myFramePitch = 0;
	int myFrameW = 0, myFrameH = 0;
	bool myFrameBgra = false;
	bool myHasFrame = false;
	cudaEvent_t myReady = nullptr;
	uint32_t myFrame = 0;
	int myInW = 0, myInH = 0;
	cudaStream_t myStream = nullptr;
	cudaSurfaceObject_t myInputSurface = 0;
	cudaSurfaceObject_t myOutputSurface = 0;
	std::unique_ptr<Pipeline> myPipeline;
	std::vector<PersonOut> myPersons;
	std::string myError, myWarning;
	float myInferMs = 0.f, myTotalMs = 0.f;
};

TOP_CPlusPlusBase*
newYoloSegCuda(const OP_NodeInfo* info, TOP_Context* context)
{
	return new YoloSegTOP(info, context);
}

void
deleteYoloSegCuda(TOP_CPlusPlusBase* instance)
{
	delete (YoloSegTOP*)instance;
}

bool
yoloSegSharedCuda(uint32_t opId, void* out)
{
	std::lock_guard<std::mutex> lock(gRegistryMutex);
	auto it = gRegistry.find(opId);
	return it != gRegistry.end() && it->second->getShared((YoloSegShared*)out);
}

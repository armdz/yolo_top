// YOLO Seg Track TOP - version portable (ONNX Runtime): macOS (CoreML) y Windows sin NVIDIA (DirectML / CPU).
// Mismo tipo de operador y mismos parametros que la version CUDA: un .toe funciona con cualquiera de las dos.
//
// Entrada : cualquier textura 2D (TouchDesigner la convierte a RGBA8 al descargarla)
// Salida  : mascara de personas / colores por ID / overlay, misma resolucion que la entrada
// Info DAT: id, cx, cy, w, h, x1, y1, x2, y2, conf, area  (normalizado 0-1, y desde arriba)
// Info CHOP: persons, infer_ms, total_ms, model_ready + p<i>_id/cx/cy/w/h por persona
//
// La red corre en un hilo aparte: la salida va 1-2 frames atrasada respecto de la entrada, pero
// TouchDesigner nunca espera a la red.

#include "TOP_CPlusPlusBase.h"
#include "Plugin.h"
#include "CpuPipeline.h"
#include "OpCommon.h"
#include "OrtModel.h"
#include "SegSharedCPU.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

using namespace TD;

class YoloSegTOPCPU;

// Registro de instancias por opId, para que otros plugins encuentren los datos del frame
static std::mutex gRegistryMutex;
static std::unordered_map<uint32_t, YoloSegTOPCPU*> gRegistry;

class YoloSegTOPCPU : public TOP_CPlusPlusBase
{
public:
	YoloSegTOPCPU(const OP_NodeInfo* info, TOP_Context* context) : myContext(context), myOpId(info->opId)
	{
		std::lock_guard<std::mutex> lock(gRegistryMutex);
		gRegistry[myOpId] = this;
	}

	~YoloSegTOPCPU() override
	{
		{
			std::lock_guard<std::mutex> lock(gRegistryMutex);
			gRegistry.erase(myOpId);
		}
		myPending.release();
		myPipeline.reset();
	}

	void getGeneralInfo(TOP_GeneralInfo* ginfo, const OP_Inputs*, void*) override
	{
		ginfo->cookEveryFrameIfAsked = true;
		// mientras la red trabaja hay que seguir cocinando para mostrar el resultado cuando llegue
		ginfo->cookEveryFrame = myPipeline && (myPipeline->modelLoading() || !myPipeline->idle() || myPending);
	}

	void execute(TOP_Output* output, const OP_Inputs* inputs, void*) override
	{
		myError.clear();
		myWarning.clear();
		opcommon::yoloEnablePars(inputs, false);

		if (inputs->getNumInputs() < 1)
		{
			clearFrame();
			myError = "Connect an input (camera / video).";
			return;
		}
		const OP_TOPInput* top = inputs->getInputTOP(0);
		if (top->textureDesc.texDim != OP_TexDim::e2D)
		{
			clearFrame();
			myError = "The input must be a 2D texture.";
			return;
		}

		PipelineSettings s = opcommon::readYoloSettings(inputs);
		const char* modelPath = inputs->getParFilePath("Modelfile");
		const bool hasModel = modelPath && *modelPath;
		const bool fp16 = inputs->getParInt("Fp16") != 0;
		const int device = inputs->getParInt("Device");

		if (!myPipeline)
			myPipeline = std::make_unique<CpuPipeline>();
		if (hasModel)
			myPipeline->requestModel(modelPath, device, fp16);

		// La descarga pedida en el cook anterior ya esta lista (no frena a TD): se manda a la red si esta libre
		if (myPending && myPipeline->modelReady() && myPipeline->idle())
		{
			const OP_TextureDesc& d = myPending->textureDesc;
			const void* data = myPending->getData();
			if (data && myPending->size >= (uint64_t)d.width * d.height * 4)
				myPipeline->submit((const uint8_t*)data, (int)d.width, (int)d.height, (size_t)d.width * 4, false, s);
		}
		myPending.release();
		if (hasModel && myPipeline->modelReady())
		{
			OP_TOPInputDownloadOptions opts;
			opts.pixelFormat = OP_PixelFormat::RGBA8Fixed;
			myPending = top->downloadTexture(opts, nullptr);
		}

		// ultimo resultado de la red
		std::shared_ptr<const CpuResult> r = myPipeline->latest();
		if (r && r->serial > mySeenSerial)
		{
			mySeenSerial = r->serial;
			myCurrent = r;
			myPersons = r->persons;
			myFrame++;
		}
		if (!hasModel || !myPipeline->modelReady())
		{
			myCurrent.reset();
			myPersons.clear();
		}

		// --- salida: el ultimo resultado, o negro a la resolucion de la entrada ---
		int W = (int)top->textureDesc.width, H = (int)top->textureDesc.height;
		if (myCurrent)
		{
			W = myCurrent->inW;
			H = myCurrent->inH;
		}
		uint64_t bytes = (uint64_t)W * H * 4;
		OP_SmartRef<TOP_Buffer> buf = myContext->createOutputBuffer(bytes, TOP_BufferFlags::None, nullptr);
		if (buf)
		{
			if (myCurrent && myCurrent->output.size() == bytes)
				memcpy(buf->data, myCurrent->output.data(), bytes);
			else
				memset(buf->data, 0, bytes);
			TOP_UploadInfo info;
			info.textureDesc.width = W;
			info.textureDesc.height = H;
			info.textureDesc.texDim = OP_TexDim::e2D;
			info.textureDesc.pixelFormat = OP_PixelFormat::BGRA8Fixed;
			info.firstPixel = TOP_FirstPixel::BottomLeft;
			output->uploadBuffer(&buf, info, nullptr);
		}

		std::string loadErr = myPipeline->loadError();
		if (!hasModel)
			myError = opcommon::noModelError();
		else if (myPipeline->modelLoading())
			myWarning = "Loading the model... (with CoreML the first load compiles the model and can take a minute)";
		else if (!loadErr.empty())
			myError = loadErr;

		if (myCurrent)
		{
			myInferMs = myCurrent->inferMs;
			myTotalMs = myCurrent->totalMs;
		}
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
		std::string backend = myPipeline ? myPipeline->backend() : "";
		char buf[512];
		snprintf(buf, sizeof(buf), "Backend: ONNX Runtime %s\nPersons: %d\nNetwork: %.2f ms\nTotal (worker thread): %.2f ms",
				 backend.empty() ? "-" : backend.c_str(), (int)myPersons.size(), myInferMs, myTotalMs);
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

	bool getShared(YoloSegSharedCPU* out) const
	{
		if (!myCurrent)
			return false;
		const CpuResult& r = *myCurrent;
		out->frame = myFrame;
		out->count = (int)std::min(r.persons.size(), r.boxes.size());
		out->segment = r.segment && !r.logits.empty() ? 1 : 0;
		out->inputW = r.inW;
		out->inputH = r.inH;
		out->modelW = r.modelW;
		out->modelH = r.modelH;
		out->protoW = r.protoW;
		out->protoH = r.protoH;
		out->logits = out->segment ? r.logits.data() : nullptr;
		out->persons = r.boxes.data();
		out->info = r.persons.data();
		out->image = r.image.data();
		out->imagePitch = (size_t)r.inW * 4;
		out->imageBgra = r.imageBgra ? 1 : 0;
		return true;
	}

private:
	// Sin entrada valida: no dejar personas viejas en la tabla ni en los datos compartidos
	void clearFrame()
	{
		myPending.release();
		myCurrent.reset();
		myPersons.clear();
	}

	TOP_Context* myContext;
	uint32_t myOpId;
	std::unique_ptr<CpuPipeline> myPipeline;
	OP_SmartRef<OP_TOPDownloadResult> myPending;
	std::shared_ptr<const CpuResult> myCurrent;
	uint64_t mySeenSerial = 0;
	uint32_t myFrame = 0;
	std::vector<PersonOut> myPersons;
	std::string myError, myWarning;
	float myInferMs = 0.f, myTotalMs = 0.f;
};

TOP_CPlusPlusBase*
newYoloSegCpu(const OP_NodeInfo* info, TOP_Context* context)
{
	return new YoloSegTOPCPU(info, context);
}

void
deleteYoloSegCpu(TOP_CPlusPlusBase* instance)
{
	delete (YoloSegTOPCPU*)instance;
}

bool
yoloSegSharedCpu(uint32_t opId, void* out)
{
	std::lock_guard<std::mutex> lock(gRegistryMutex);
	auto it = gRegistry.find(opId);
	return it != gRegistry.end() && it->second->getShared((YoloSegSharedCPU*)out);
}

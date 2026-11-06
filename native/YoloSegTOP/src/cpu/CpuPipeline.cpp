#include "CpuPipeline.h"

#include "CpuKernels.h"
#include "OrtModel.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <stdexcept>

CpuPipeline::CpuPipeline()
{
	myThread = std::thread(&CpuPipeline::loop, this);
}

CpuPipeline::~CpuPipeline()
{
	{
		std::lock_guard<std::mutex> lock(myMutex);
		myQuit = true;
	}
	myCv.notify_all();
	if (myThread.joinable())
		myThread.join();   // solo espera si justo esta cargando un modelo o procesando un frame
}

void
CpuPipeline::requestModel(const std::string& onnxPath, int device, bool fp16)
{
	std::string key = onnxPath + "|" + std::to_string(device) + (fp16 ? "|fp16" : "|fp32");
	{
		std::lock_guard<std::mutex> lock(myMutex);
		if (key == myModelKey)
			return;
		myModelKey = key;
		myLoadError.clear();
		myReqPath = onnxPath;
		myReqDevice = device;
		myReqFp16 = fp16;
		myHasRequest = true;
		myLoading = true;
	}
	myCv.notify_all();
}

std::string
CpuPipeline::loadError()
{
	std::lock_guard<std::mutex> lock(myMutex);
	return myLoadError;
}

std::string
CpuPipeline::backend()
{
	std::lock_guard<std::mutex> lock(myMutex);
	return myBackend;
}

bool
CpuPipeline::submit(const uint8_t* img, int W, int H, size_t pitch, bool bgra, const PipelineSettings& s)
{
	if (myBusy || !myReady || !img || W <= 0 || H <= 0)
		return false;
	auto job = std::make_unique<CpuResult>();
	job->inW = W;
	job->inH = H;
	job->imageBgra = bgra;
	job->image.resize((size_t)W * H * 4);
	for (int y = 0; y < H; y++)
		memcpy(job->image.data() + (size_t)y * W * 4, img + (size_t)y * pitch, (size_t)W * 4);
	{
		std::lock_guard<std::mutex> lock(myMutex);
		myJob = std::move(job);
		myJobSettings = s;
		myHasJob = true;
		myBusy = true;
	}
	myCv.notify_all();
	return true;
}

std::shared_ptr<const CpuResult>
CpuPipeline::latest()
{
	std::lock_guard<std::mutex> lock(myMutex);
	return myLatest;
}

void
CpuPipeline::loop()
{
	for (;;)
	{
		std::unique_ptr<CpuResult> job;
		PipelineSettings settings;
		std::string path;
		int device = 0;
		bool fp16 = true, load = false;
		{
			std::unique_lock<std::mutex> lock(myMutex);
			myCv.wait(lock, [this] { return myQuit || myHasRequest || myHasJob; });
			if (myQuit)
				return;
			if (myHasRequest)
			{
				load = true;
				path = myReqPath;
				device = myReqDevice;
				fp16 = myReqFp16;
				myHasRequest = false;
			}
			else
			{
				job = std::move(myJob);
				settings = myJobSettings;
				myHasJob = false;
			}
		}

		if (load)
		{
			myReady = false;
			myModel.reset();
			auto m = std::make_unique<YoloOrtModel>();
			std::string err;
			bool ok = false;
			// una excepcion que salga de este hilo cerraria TouchDesigner
			try
			{
				ok = m->load(path, device, fp16, err);
			}
			catch (const std::exception& e)
			{
				err = std::string("Error loading the model: ") + e.what();
			}
			catch (...)
			{
				err = "Unknown error loading the model";
			}
			std::lock_guard<std::mutex> lock(myMutex);
			if (myHasRequest || myQuit)
				continue;   // mientras cargaba pidieron otro modelo: descartar este
			if (ok)
			{
				myModel = std::move(m);
				myBackend = myModel->backend;
				myTracker.reset();
				myReady = true;
			}
			else
			{
				myLoadError = err;
				myBackend.clear();
			}
			myLoading = false;
			continue;
		}

		if (job && myModel)
		{
			try
			{
				process(*job, settings);
				std::lock_guard<std::mutex> lock(myMutex);
				job->serial = ++mySerial;
				myLatest = std::move(job);
				if (myFrameError)
				{
					myLoadError.clear();
					myFrameError = false;
				}
			}
			catch (const std::exception& e)
			{
				std::lock_guard<std::mutex> lock(myMutex);
				myLoadError = std::string("Error processing the frame: ") + e.what();
				myFrameError = true;
			}
		}
		myBusy = false;
	}
}

void
CpuPipeline::process(CpuResult& r, const PipelineSettings& s)
{
	auto t0 = std::chrono::high_resolution_clock::now();
	YoloOrtModel& model = *myModel;
	const int W = r.inW, H = r.inH;
	const int modelW = model.inW, modelH = model.inH;
	const size_t pitch = (size_t)W * 4;

	if (myResetPending.exchange(false))
		myTracker.reset();

	// --- red ---
	cpuPreprocess(r.image.data(), W, H, pitch, r.imageBgra, model.input.data(), modelW, modelH);
	std::string err;
	if (!model.infer(err))
		throw std::runtime_error(err);
	r.inferMs = std::chrono::duration<float, std::milli>(std::chrono::high_resolution_clock::now() - t0).count();

	// --- tracking ---
	ByteTrackParams tp;
	tp.highThresh = s.confidence;
	tp.newTrackThresh = s.confidence;
	tp.lowThresh = std::min(0.1f, s.confidence);
	tp.trackBuffer = s.trackBuffer;
	tp.matchThresh = s.matchThresh;

	std::vector<Detection> dets;
	for (int i = 0; i < model.numDet; i++)
	{
		const float* d = model.det.data() + (size_t)i * model.detDim;
		if ((int)(d[5] + 0.5f) != 0 || d[4] <= tp.lowThresh)   // solo personas (clase 0)
			continue;
		dets.push_back({ d[0], d[1], d[2], d[3], d[4], i });
	}
	std::vector<TrackResult> tracks;
	if (s.track)
		tracks = myTracker.update(dets, tp);
	else
	{
		myTracker.reset();   // al volver a activar el tracking, arranca de cero
		tracks = ByteTracker::untracked(dets, s.confidence);
	}

	const int N = std::min((int)tracks.size(), kMaxPersons);
	const int C = model.protoC;
	const int pW = C > 0 ? model.protoW : 1, pH = C > 0 ? model.protoH : 1;
	r.modelW = modelW;
	r.modelH = modelH;
	r.protoW = pW;
	r.protoH = pH;
	r.segment = s.segment && C > 0;
	r.boxes.resize(N);
	r.persons.resize(N);
	if (r.segment)
		r.logits.resize((size_t)N * pW * pH);
	for (int k = 0; k < N; k++)
	{
		const TrackResult& t = tracks[k];
		const float* d = model.det.data() + (size_t)t.detIndex * model.detDim;
		PersonGPU& p = r.boxes[k];
		p.cropX1 = d[0]; p.cropY1 = d[1]; p.cropX2 = d[2]; p.cropY2 = d[3];
		p.boxX1 = clamp01(t.x1 / modelW); p.boxY1 = clamp01(t.y1 / modelH);
		p.boxX2 = clamp01(t.x2 / modelW); p.boxY2 = clamp01(t.y2 / modelH);
		idColor(t.id, p.r, p.g, p.b);
		p.pad = 0;
		p.maskX1 = p.boxX1; p.maskY1 = p.boxY1; p.maskX2 = p.boxX2; p.maskY2 = p.boxY2;

		PersonOut& o = r.persons[k];
		o.id = t.id;
		o.x1 = p.boxX1; o.y1 = p.boxY1; o.x2 = p.boxX2; o.y2 = p.boxY2;
		o.w = o.x2 - o.x1; o.h = o.y2 - o.y1;
		o.cx = (o.x1 + o.x2) * 0.5f; o.cy = (o.y1 + o.y2) * 0.5f;
		o.conf = t.score;
		o.area = o.w * o.h;

		if (r.segment)
		{
			int area = 0, b[4];
			cpuMaskLogits(model.proto.data(), C, pH, pW, modelW, modelH, d + 6, p,
						  r.logits.data() + (size_t)k * pW * pH, area, b);
			o.area = (float)area / (pW * pH);
			if (area > 0)
			{
				// pixeles del proto -> normalizado, con medio pixel de margen (el borde suave es bilineal)
				// y recortado a la caja de la deteccion (la mascara no se dibuja fuera de ella)
				p.maskX1 = clamp01(std::max((-b[0] - 0.5f) / pW, p.cropX1 / modelW));
				p.maskY1 = clamp01(std::max((-b[1] - 0.5f) / pH, p.cropY1 / modelH));
				p.maskX2 = clamp01(std::min((b[2] + 1.5f) / pW, p.cropX2 / modelW));
				p.maskY2 = clamp01(std::min((b[3] + 1.5f) / pH, p.cropY2 / modelH));
			}
		}
	}

	// --- salida ---
	r.output.resize((size_t)W * H * 4);
	cpuCompose(r.image.data(), pitch, r.imageBgra, r.output.data(), W, H, r.segment ? r.logits.data() : nullptr,
			   pH, pW, modelW, modelH, r.boxes.data(), N, s.mode, r.segment, s.soft);
	r.totalMs = std::chrono::duration<float, std::milli>(std::chrono::high_resolution_clock::now() - t0).count();
}

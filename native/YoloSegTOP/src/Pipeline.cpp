#include "Pipeline.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>

namespace
{
bool isSticky(cudaError_t e)
{
	switch (e)
	{
		case cudaErrorIllegalAddress: case cudaErrorLaunchFailure: case cudaErrorIllegalInstruction:
		case cudaErrorMisalignedAddress: case cudaErrorInvalidAddressSpace: case cudaErrorInvalidPc:
		case cudaErrorHardwareStackError: case cudaErrorAssert: case cudaErrorLaunchTimeout:
			return true;
		default:
			return false;
	}
}

// Arma el mensaje de error con la etapa y el codigo de CUDA
bool fail(std::string& err, const char* stage, cudaError_t e)
{
	err = std::string(stage) + ": " + cudaGetErrorName(e) + " - " + cudaGetErrorString(e);
	if (isSticky(e))
		err += ". Fatal CUDA error: restart TouchDesigner.";
	return false;
}

#define CHECK(stage, call) do { cudaError_t e_ = (call); if (e_ != cudaSuccess) return fail(err, stage, e_); } while (0)
} // namespace

Pipeline::Pipeline()
{
	cudaMallocHost((void**)&hPersons, kMaxPersons * sizeof(PersonGPU));
	cudaMallocHost((void**)&hArea, kMaxPersons * sizeof(int));
	cudaMallocHost((void**)&hBounds, kMaxPersons * 4 * sizeof(int));
	cudaMalloc((void**)&dPersons, kMaxPersons * sizeof(PersonGPU));
	cudaMalloc((void**)&dArea, kMaxPersons * sizeof(int));
	cudaMalloc((void**)&dBounds, kMaxPersons * 4 * sizeof(int));
	cudaEventCreate(&myEvStart);
	cudaEventCreate(&myEvInfer);
}

Pipeline::~Pipeline()
{
	bool busy;
	{
		std::lock_guard<std::mutex> lock(myLoad->mutex);
		myLoad->quit = true;
		myLoad->hasRequest = false;
		myLoad->cancel = true;   // si se esta construyendo un engine, TensorRT lo corta en el proximo paso
		busy = myLoad->busy;
	}
	myLoad->cv.notify_all();
	if (myLoader.joinable())
	{
		// cargando un modelo: no esperar (el build puede tardar), el hilo termina solo con su copia del estado
		if (busy)
			myLoader.detach();
		else
			myLoader.join();
	}
	myModel.reset();
	freeBuffers();
	cudaFreeHost(hPersons);
	cudaFreeHost(hArea);
	cudaFreeHost(hBounds);
	cudaFree(dPersons);
	cudaFree(dArea);
	cudaFree(dBounds);
	cudaEventDestroy(myEvStart);
	cudaEventDestroy(myEvInfer);
}

void
Pipeline::freeBuffers()
{
	cudaFreeHost(hDet);
	cudaFreeHost(hCoeffs);
	cudaFree(dCoeffs);
	cudaFree(dLogits);
	hDet = hCoeffs = dCoeffs = dLogits = nullptr;
	myDetCapacity = myLogitsCapacity = myCoeffCapacity = 0;
}

void
Pipeline::requestModel(const std::string& onnxPath, bool fp16, int cudaDevice, CUcontext ctx)
{
	std::string key = onnxPath + (fp16 ? "|fp16" : "|fp32");
	if (key == myModelKey)
		return;
	myModelKey = key;
	{
		// Nunca bloquea: el hilo de carga toma siempre el ultimo pedido
		std::lock_guard<std::mutex> lock(myLoad->mutex);
		myLoad->loadError.clear();
		myLoad->reqPath = onnxPath;
		myLoad->reqFp16 = fp16;
		myLoad->reqDevice = cudaDevice;
		myLoad->reqCtx = ctx;
		myLoad->hasRequest = true;
		myLoad->cancel = true;   // cortar el build en curso: se pidio otro modelo
		myLoad->loading = true;
	}
	myLoad->cv.notify_all();
	if (!myLoader.joinable())
		myLoader = std::thread(&Pipeline::loaderLoop, myLoad);
}

void
Pipeline::loaderLoop(std::shared_ptr<LoaderState> st)
{
	for (;;)
	{
		std::string path;
		bool fp16;
		int device;
		CUcontext ctx;
		{
			std::unique_lock<std::mutex> lock(st->mutex);
			st->busy = false;
			st->cv.wait(lock, [&] { return st->quit || st->hasRequest; });
			if (st->quit)
				return;
			path = st->reqPath;
			fp16 = st->reqFp16;
			device = st->reqDevice;
			ctx = st->reqCtx;
			st->hasRequest = false;
			st->busy = true;
			st->cancel = false;
			st->buildStart = -1.0;
		}
		if (ctx)
			cuCtxSetCurrent(ctx);
		else
			cudaSetDevice(device);
		std::shared_ptr<TrtModel> m;
		std::string err;
		bool ok = false;
		// una excepcion que salga de este hilo cerraria TouchDesigner
		try
		{
			m = std::make_shared<TrtModel>();
			ok = m->load(path, fp16, err, &st->cancel, &st->buildStart);
		}
		catch (const std::exception& e)
		{
			err = std::string("Error loading the model: ") + e.what();
		}
		catch (...)
		{
			err = "Unknown error loading the model";
		}
		if (!ok)
			m.reset();
		std::lock_guard<std::mutex> lock(st->mutex);
		if (st->hasRequest || st->quit)
			continue;   // mientras cargaba pidieron otro modelo (o se borro el operador): descartar este
		if (ok)
			st->pending = m;
		else
			st->loadError = err;
		st->loading = false;
	}
}

double
Pipeline::buildSeconds() const
{
	double start = myLoad->buildStart;
	if (start < 0 || !myLoad->loading)
		return -1.0;
	return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count() - start;
}

std::string
Pipeline::loadError()
{
	std::lock_guard<std::mutex> lock(myLoad->mutex);
	return myLoad->loadError;
}

bool
Pipeline::process(cudaSurfaceObject_t in, int inW, int inH, bool inBgra, cudaSurfaceObject_t out,
				  const PipelineSettings& s, cudaStream_t stream, std::vector<PersonOut>& persons,
				  std::string& err)
{
	err.clear();
	cudaGetLastError();   // limpiar errores viejos que no son nuestros (otros OPs en este hilo)
	{
		std::lock_guard<std::mutex> lock(myLoad->mutex);
		if (myLoad->pending)
		{
			myModel = myLoad->pending;
			myLoad->pending.reset();
			myTracker.reset();
		}
	}
	if (myResetPending.exchange(false))
		myTracker.reset();

	persons.clear();
	last.persons.clear();
	last.logits = nullptr;
	std::shared_ptr<TrtModel> model = myModel;
	int N = 0;
	bool segment = false;
	int modelW = 1, modelH = 1, pW = 1, pH = 1;

	if (model && in)
	{
		modelW = model->inW;
		modelH = model->inH;
		size_t detCount = (size_t)model->numDet * model->detDim;
		// (si una reserva falla, el puntero queda en null y la capacidad en 0: el proximo frame reintenta)
		if (detCount > myDetCapacity)
		{
			cudaFreeHost(hDet);
			hDet = nullptr;
			myDetCapacity = 0;
			CHECK("allocating memory", cudaMallocHost((void**)&hDet, detCount * sizeof(float)));
			myDetCapacity = detCount;
		}
		if (model->protoC > 0)
		{
			pW = model->protoW;
			pH = model->protoH;
			size_t logits = (size_t)kMaxPersons * pW * pH;
			if (logits > myLogitsCapacity)
			{
				cudaFree(dLogits);
				dLogits = nullptr;
				myLogitsCapacity = 0;
				CHECK("allocating memory", cudaMalloc((void**)&dLogits, logits * sizeof(float)));
				myLogitsCapacity = logits;
			}
			size_t coeffs = (size_t)kMaxPersons * model->protoC;
			if (coeffs > myCoeffCapacity)
			{
				cudaFreeHost(hCoeffs);
				cudaFree(dCoeffs);
				hCoeffs = dCoeffs = nullptr;
				myCoeffCapacity = 0;
				CHECK("allocating memory", cudaMallocHost((void**)&hCoeffs, coeffs * sizeof(float)));
				CHECK("allocating memory", cudaMalloc((void**)&dCoeffs, coeffs * sizeof(float)));
				myCoeffCapacity = coeffs;
			}
		}

		// --- red ---
		CHECK("event", cudaEventRecord(myEvStart, stream));
		launchPreprocess(in, inW, inH, inBgra, model->dInput, model->inW, model->inH, stream);
		CHECK("preprocess", cudaGetLastError());
		if (!model->infer(stream))
		{
			err = "TensorRT could not run the network: " + TrtModel::lastLog();
			return false;
		}
		CHECK("event", cudaEventRecord(myEvInfer, stream));
		CHECK("copying detections", cudaMemcpyAsync(hDet, model->dDet, detCount * sizeof(float), cudaMemcpyDeviceToHost, stream));
		CHECK("inference", cudaStreamSynchronize(stream));
		cudaEventElapsedTime(&inferMs, myEvStart, myEvInfer);

		// --- tracking (CPU, pocas decenas de cajas) ---
		ByteTrackParams tp;
		tp.highThresh = s.confidence;
		tp.newTrackThresh = s.confidence;
		tp.lowThresh = std::min(0.1f, s.confidence);
		tp.trackBuffer = s.trackBuffer;
		tp.matchThresh = s.matchThresh;

		std::vector<Detection> dets;
		for (int i = 0; i < model->numDet; i++)
		{
			const float* d = hDet + (size_t)i * model->detDim;
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

		N = std::min((int)tracks.size(), kMaxPersons);
		const int C = model->protoC;
		for (int k = 0; k < N; k++)
		{
			const TrackResult& t = tracks[k];
			const float* d = hDet + (size_t)t.detIndex * model->detDim;
			PersonGPU& p = hPersons[k];
			p.cropX1 = d[0]; p.cropY1 = d[1]; p.cropX2 = d[2]; p.cropY2 = d[3];
			p.boxX1 = clamp01(t.x1 / modelW); p.boxY1 = clamp01(t.y1 / modelH);
			p.boxX2 = clamp01(t.x2 / modelW); p.boxY2 = clamp01(t.y2 / modelH);
			idColor(t.id, p.r, p.g, p.b);
			p.pad = 0;
			p.maskX1 = p.boxX1; p.maskY1 = p.boxY1; p.maskX2 = p.boxX2; p.maskY2 = p.boxY2;
			if (C > 0)
				memcpy(hCoeffs + k * C, d + 6, C * sizeof(float));

			PersonOut o;
			o.id = t.id;
			o.x1 = p.boxX1; o.y1 = p.boxY1; o.x2 = p.boxX2; o.y2 = p.boxY2;
			o.w = o.x2 - o.x1; o.h = o.y2 - o.y1;
			o.cx = (o.x1 + o.x2) * 0.5f; o.cy = (o.y1 + o.y2) * 0.5f;
			o.conf = t.score;
			o.area = o.w * o.h;
			persons.push_back(o);
		}
		if (N > 0)
			cudaMemcpyAsync(dPersons, hPersons, N * sizeof(PersonGPU), cudaMemcpyHostToDevice, stream);

		segment = s.segment && C > 0;
		if (segment && N > 0)
		{
			cudaMemcpyAsync(dCoeffs, hCoeffs, N * C * sizeof(float), cudaMemcpyHostToDevice, stream);
			cudaMemsetAsync(dArea, 0, N * sizeof(int), stream);
			cudaMemsetAsync(dBounds, 0x80, N * 4 * sizeof(int), stream);
			launchMaskLogits(model->dProto, C, pH, pW, modelW, modelH, dCoeffs, dPersons, N, dLogits, dArea, dBounds, stream);
			CHECK("masks", cudaGetLastError());
		}
	}

	launchCompose(in, inBgra, out, inW, inH, dLogits, pH, pW, modelW, modelH, dPersons, N,
				  s.mode, segment, s.soft, stream);
	CHECK("output", cudaGetLastError());

	if (segment && N > 0)
	{
		CHECK("copying areas", cudaMemcpyAsync(hArea, dArea, N * sizeof(int), cudaMemcpyDeviceToHost, stream));
		CHECK("copying bounds", cudaMemcpyAsync(hBounds, dBounds, N * 4 * sizeof(int), cudaMemcpyDeviceToHost, stream));
		CHECK("masks/output", cudaStreamSynchronize(stream));
		for (int k = 0; k < N; k++)
		{
			persons[k].area = (float)hArea[k] / (pW * pH);
			if (hArea[k] == 0)
				continue;   // sin pixeles de mascara: queda la caja del tracker
			// pixeles del proto -> normalizado, con medio pixel de margen (el borde suave es bilineal)
			// y recortado a la caja de la deteccion (la mascara no se dibuja fuera de ella)
			const int* b = hBounds + k * 4;
			PersonGPU& p = hPersons[k];
			p.maskX1 = clamp01(std::max((-b[0] - 0.5f) / pW, p.cropX1 / modelW));
			p.maskY1 = clamp01(std::max((-b[1] - 0.5f) / pH, p.cropY1 / modelH));
			p.maskX2 = clamp01(std::min((b[2] + 1.5f) / pW, p.cropX2 / modelW));
			p.maskY2 = clamp01(std::min((b[3] + 1.5f) / pH, p.cropY2 / modelH));
		}
	}

	// datos del frame para otros operadores (ver SegShared.h)
	last.persons.assign(hPersons, hPersons + N);
	last.segment = segment;
	last.modelW = modelW;
	last.modelH = modelH;
	last.protoW = pW;
	last.protoH = pH;
	last.logits = segment ? dLogits : nullptr;
	return true;
}

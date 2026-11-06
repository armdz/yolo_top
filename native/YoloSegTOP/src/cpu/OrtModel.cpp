#include "OrtModel.h"

#include "onnxruntime_c_api.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <thread>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace fs = std::filesystem;

namespace
{
// ---------------------------------------------------------------- carga de la libreria

#ifdef _WIN32
std::wstring
moduleDir(HMODULE m)
{
	wchar_t buf[MAX_PATH * 2];
	DWORD n = GetModuleFileNameW(m, buf, (DWORD)(sizeof(buf) / sizeof(buf[0])));
	std::wstring p(buf, n);
	size_t slash = p.find_last_of(L"\\/");
	return slash == std::wstring::npos ? L"." : p.substr(0, slash);
}

HMODULE
selfModule()
{
	HMODULE m = nullptr;
	GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
					   (LPCWSTR)&selfModule, &m);
	return m;
}

void*
findSymbol(void* lib, const char* name)
{
	return lib ? (void*)GetProcAddress((HMODULE)lib, name) : nullptr;
}

void*
openLibrary(std::string& where)
{
	// 1) subcarpeta YoloSeg junto al plugin: permite usar otra build de ONNX Runtime (ej. DirectML)
	//    sin pisar la que usa TouchDesigner
	std::wstring own = moduleDir(selfModule()) + L"\\YoloSeg\\onnxruntime.dll";
	if (HMODULE m = LoadLibraryExW(own.c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH))
	{
		where = "plugin YoloSeg folder";
		return m;
	}
	// 2) la que trae TouchDesigner (bin\onnxruntime.dll, junto al .exe)
	std::wstring td = moduleDir(nullptr) + L"\\onnxruntime.dll";
	if (HMODULE m = LoadLibraryExW(td.c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH))
	{
		where = "TouchDesigner";
		return m;
	}
	if (HMODULE m = GetModuleHandleW(L"onnxruntime.dll"))
	{
		where = "already loaded";
		return m;
	}
	return nullptr;
}
#else
std::string
selfDir()
{
	Dl_info info;
	if (!dladdr((void*)&selfDir, &info) || !info.dli_fname)
		return ".";
	return fs::path(info.dli_fname).parent_path().string();
}

void*
findSymbol(void* lib, const char* name)
{
	return lib ? dlsym(lib, name) : nullptr;
}

void*
openLibrary(std::string& where)
{
	// el .plugin trae su propia copia: YoloSegTOP.plugin/Contents/Frameworks/libonnxruntime.dylib
	const std::string dir = selfDir();
	const std::string candidates[] = {
		dir + "/../Frameworks/libonnxruntime.dylib",
		dir + "/libonnxruntime.dylib",
		"libonnxruntime.dylib",
		"libonnxruntime.so",
	};
	for (const std::string& c : candidates)
		if (void* h = dlopen(c.c_str(), RTLD_NOW | RTLD_LOCAL))
		{
			where = c;
			return h;
		}
	return nullptr;
}
#endif

struct OrtLib
{
	void* handle = nullptr;
	const OrtApi* api = nullptr;
	OrtEnv* env = nullptr;
	std::string where;
	std::string error;
};

OrtLib&
ortLib()
{
	static OrtLib lib;
	static std::once_flag once;
	std::call_once(once, [] {
		lib.handle = openLibrary(lib.where);
		if (!lib.handle)
		{
#ifdef _WIN32
			lib.error = "onnxruntime.dll not found (it ships with TouchDesigner 2023 and newer). "
						"You can also install it with scripts\\get_directml.bat.";
#else
			lib.error = "libonnxruntime.dylib not found inside the plugin (rebuild with build_mac.sh).";
#endif
			return;
		}
		typedef const OrtApiBase*(ORT_API_CALL * GetApiBaseFn)();
		auto getBase = (GetApiBaseFn)findSymbol(lib.handle, "OrtGetApiBase");
		const OrtApiBase* base = getBase ? getBase() : nullptr;
		// la API es compatible hacia atras: se pide la version de los headers, o la mas nueva que tenga la libreria
		for (uint32_t v = ORT_API_VERSION; base && !lib.api && v >= 14; v--)
			lib.api = base->GetApi(v);
		if (!lib.api)
		{
			lib.error = "ONNX Runtime is too old (1.14 or newer is needed): "
						"update TouchDesigner or install a newer one with scripts\\get_directml.bat.";
			return;
		}
		OrtStatus* s = lib.api->CreateEnv(ORT_LOGGING_LEVEL_ERROR, "YoloSeg", &lib.env);
		if (s)
		{
			lib.error = std::string("ONNX Runtime: ") + lib.api->GetErrorMessage(s);
			lib.api->ReleaseStatus(s);
			lib.api = nullptr;
		}
	});
	return lib;
}

const OrtApi&
api()
{
	return *ortLib().api;
}

// Si hay error, lo copia a 'err' y libera el status
bool
check(OrtStatus* s, std::string& err, const char* what)
{
	if (!s)
		return true;
	err = std::string(what) + ": " + api().GetErrorMessage(s);
	api().ReleaseStatus(s);
	return false;
}

// ---------------------------------------------------------------- backends

enum class Ep { Cpu, CoreMLCached, CoreML, DirectML };

const char*
epName(Ep e)
{
	switch (e)
	{
		case Ep::CoreMLCached:
		case Ep::CoreML: return "CoreML";
		case Ep::DirectML: return "DirectML";
		default: return "CPU";
	}
}

// Arma las opciones de sesion para un backend. false = ese backend no esta disponible en esta libreria
bool
makeOptions(Ep ep, bool fp16, const std::string& cacheDir, OrtSessionOptions** out, std::string& err)
{
	const OrtApi& a = api();
	OrtSessionOptions* o = nullptr;
	if (!check(a.CreateSessionOptions(&o), err, "session options"))
		return false;
	// la mitad de los nucleos: el resto queda para TouchDesigner
	int threads = (int)std::max(std::thread::hardware_concurrency() / 2, 1u);
	a.SetIntraOpNumThreads(o, threads);
	a.SetInterOpNumThreads(o, 1);
	a.SetSessionGraphOptimizationLevel(o, ORT_ENABLE_ALL);

	bool ok = true;
	if (ep == Ep::CoreMLCached || ep == Ep::CoreML)
	{
		std::vector<const char*> keys = { "ModelFormat", "MLComputeUnits", "RequireStaticInputShapes", "AllowLowPrecisionAccumulationOnGPU" };
		std::vector<const char*> vals = { "MLProgram", "ALL", "1", fp16 ? "1" : "0" };
		if (ep == Ep::CoreMLCached)
		{
			keys.push_back("ModelCacheDirectory");
			vals.push_back(cacheDir.c_str());
		}
		ok = check(a.SessionOptionsAppendExecutionProvider(o, "CoreML", keys.data(), vals.data(), keys.size()), err, "CoreML");
		if (!ok && ep == Ep::CoreML)
		{
			// ONNX Runtime viejo: funcion con flags (0x010 = COREML_FLAG_CREATE_MLPROGRAM)
			typedef OrtStatus*(ORT_API_CALL * CoreMLFn)(OrtSessionOptions*, uint32_t);
			if (auto fn = (CoreMLFn)findSymbol(ortLib().handle, "OrtSessionOptionsAppendExecutionProvider_CoreML"))
				ok = check(fn(o, 0x010), err, "CoreML");
		}
	}
	else if (ep == Ep::DirectML)
	{
		typedef OrtStatus*(ORT_API_CALL * DmlFn)(OrtSessionOptions*, int);
		auto fn = (DmlFn)findSymbol(ortLib().handle, "OrtSessionOptionsAppendExecutionProvider_DML");
		ok = fn != nullptr;
		if (ok)
		{
			// DirectML no soporta memory pattern ni ejecucion en paralelo
			a.DisableMemPattern(o);
			a.SetSessionExecutionMode(o, ORT_SEQUENTIAL);
			ok = check(fn(o, 0), err, "DirectML");
		}
	}
	if (!ok)
	{
		a.ReleaseSessionOptions(o);
		return false;
	}
	*out = o;
	return true;
}

bool
readFile(const std::string& path, std::vector<char>& data)
{
	std::ifstream f(fs::u8path(path), std::ios::binary | std::ios::ate);
	if (!f.good())
		return false;
	std::streamoff size = f.tellg();
	if (size <= 0)
		return false;
	data.resize((size_t)size);
	f.seekg(0);
	f.read(data.data(), data.size());
	return f.good();
}

// Forma de un tensor de entrada/salida; false si no es float
bool
tensorShape(OrtTypeInfo* ti, std::vector<int64_t>& shape)
{
	const OrtApi& a = api();
	const OrtTensorTypeAndShapeInfo* t = nullptr;
	ONNXTensorElementDataType type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
	size_t n = 0;
	bool ok = !a.CastTypeInfoToTensorInfo(ti, &t) && t && !a.GetTensorElementType(t, &type) && !a.GetDimensionsCount(t, &n);
	if (ok)
	{
		shape.resize(n);
		ok = !a.GetDimensions(t, shape.data(), n);
	}
	a.ReleaseTypeInfo(ti);
	return ok && type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
}
} // namespace

YoloOrtModel::~YoloOrtModel()
{
	if (!ortLib().api)
		return;
	for (OrtValue* v : myValues)
		if (v)
			api().ReleaseValue(v);
	if (mySession)
		api().ReleaseSession(mySession);
}

bool
YoloOrtModel::load(const std::string& onnxPath, int device, bool fp16, std::string& err)
{
	OrtLib& lib = ortLib();
	if (!lib.api)
	{
		err = lib.error;
		return false;
	}
	const OrtApi& a = api();

	std::vector<char> data;
	if (!readFile(onnxPath, data))
	{
		err = "Could not read the model: " + onnxPath;
		return false;
	}

	// backends en orden de preferencia; si uno falla se prueba el siguiente
	std::vector<Ep> eps;
	if (device == DeviceAuto)
	{
#ifdef __APPLE__
		eps = { Ep::CoreMLCached, Ep::CoreML };
#elif defined(_WIN32)
		eps = { Ep::DirectML };
#endif
	}
	eps.push_back(Ep::Cpu);

	// cache del modelo compilado de CoreML, al lado del .onnx
	fs::path p = fs::u8path(onnxPath);
	std::string cacheDir = (p.parent_path() / (p.stem().u8string() + ".coreml")).u8string();

	std::string firstErr;
	for (Ep ep : eps)
	{
		OrtSessionOptions* o = nullptr;
		std::string e;
		if (!makeOptions(ep, fp16, cacheDir, &o, e))
		{
			if (firstErr.empty() && ep != Ep::DirectML)   // DirectML no disponible es lo normal
				firstErr = e;
			continue;
		}
		if (ep == Ep::CoreMLCached)
		{
			std::error_code ec;
			fs::create_directories(fs::u8path(cacheDir), ec);
		}
		OrtStatus* s = a.CreateSessionFromArray(lib.env, data.data(), data.size(), o, &mySession);
		a.ReleaseSessionOptions(o);
		if (check(s, e, epName(ep)))
		{
			backend = epName(ep);
			break;
		}
		mySession = nullptr;
		if (firstErr.empty())
			firstErr = e;
		if (ep == Ep::Cpu)
		{
			err = "ONNX Runtime could not open the model: " + e;
			return false;
		}
	}
	if (!mySession)
	{
		err = "ONNX Runtime could not open the model: " + firstErr;
		return false;
	}
	if (device == DeviceAuto && backend == "CPU" && !firstErr.empty())
		backend = "CPU (" + firstErr + ")";

	// --- entradas y salidas ---
	OrtAllocator* alloc = nullptr;
	if (!check(a.GetAllocatorWithDefaultOptions(&alloc), err, "allocator"))
		return false;
	auto name = [&](bool input, size_t i) {
		char* n = nullptr;
		OrtStatus* s = input ? a.SessionGetInputName(mySession, i, alloc, &n) : a.SessionGetOutputName(mySession, i, alloc, &n);
		std::string r;
		if (!s && n)
			r = n;
		if (s)
			a.ReleaseStatus(s);
		if (n)
			a.AllocatorFree(alloc, n);
		return r;
	};

	size_t nIn = 0, nOut = 0;
	a.SessionGetInputCount(mySession, &nIn);
	a.SessionGetOutputCount(mySession, &nOut);
	if (nIn != 1)
	{
		err = "The model must have a single input (image 1x3xHxW)";
		return false;
	}

	std::vector<int64_t> inShape, detShape, protoShape;
	OrtTypeInfo* ti = nullptr;
	if (!check(a.SessionGetInputTypeInfo(mySession, 0, &ti), err, "input") || !tensorShape(ti, inShape) || inShape.size() != 4)
	{
		err = "Unexpected input: it must be float 1x3xHxW (export with half=False)";
		return false;
	}
	myInNames.push_back(name(true, 0));
	for (size_t i = 0; i < nOut; i++)
	{
		std::vector<int64_t> shape;
		ti = nullptr;
		if (!check(a.SessionGetOutputTypeInfo(mySession, i, &ti), err, "output") || !tensorShape(ti, shape))
			continue;
		if (shape.size() == 3 && detShape.empty())
		{
			detShape = shape;
			myDetOut = (int)myOutNames.size();
			myOutNames.push_back(name(false, i));
		}
		else if (shape.size() == 4 && protoShape.empty())
		{
			protoShape = shape;
			myProtoOut = (int)myOutNames.size();
			myOutNames.push_back(name(false, i));
		}
	}
	for (auto* s : { &inShape, &detShape, &protoShape })
		for (int64_t d : *s)
			if (d <= 0)
			{
				err = "The model has a dynamic shape: export it with dynamic=False and a fixed imgsz, e.g. imgsz=(384, 640)";
				return false;
			}
	if (detShape.empty())
	{
		err = "The model has no detection output";
		return false;
	}
	inH = (int)inShape[2];
	inW = (int)inShape[3];
	numDet = (int)detShape[1];
	detDim = (int)detShape[2];
	// end2end (YOLO26 por defecto): 1 x 300 x (6 + 32). Sin end2end (YOLOv8/11): 1 x 116 x 8400
	if (inShape[1] != 3 || detDim < 6 || detDim > numDet)
	{
		err = "Unexpected model format (expected YOLO seg end2end, output 1x300x38). "
			  "Use a YOLO26 seg model exported with format='onnx' (end2end is on by default).";
		return false;
	}
	if (!protoShape.empty())
	{
		protoC = (int)protoShape[1];
		protoH = (int)protoShape[2];
		protoW = (int)protoShape[3];
		if (detDim < 6 + protoC)
			protoC = 0;
	}

	// --- tensores sobre nuestros buffers (sin copias en cada frame) ---
	input.assign((size_t)3 * inW * inH, 0.f);
	det.assign((size_t)numDet * detDim, 0.f);
	if (!protoShape.empty())
		proto.assign((size_t)protoShape[1] * protoH * protoW, 0.f);
	OrtMemoryInfo* mem = nullptr;
	if (!check(a.CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &mem), err, "memory info"))
		return false;
	auto tensor = [&](std::vector<float>& buf, const std::vector<int64_t>& shape) {
		OrtValue* v = nullptr;
		std::string e;
		check(a.CreateTensorWithDataAsOrtValue(mem, buf.data(), buf.size() * sizeof(float), shape.data(), shape.size(),
											   ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &v), e, "tensor");
		myValues.push_back(v);
	};
	tensor(input, inShape);
	tensor(det, detShape);   // myValues[1 + myDetOut]
	if (!protoShape.empty())
		tensor(proto, protoShape);
	a.ReleaseMemoryInfo(mem);
	for (OrtValue* v : myValues)
		if (!v)
		{
			err = "Could not create the model tensors";
			return false;
		}
	if (myDetOut != 0)   // las salidas tienen que quedar en el mismo orden que myOutNames
		std::swap(myValues[1], myValues[2]);
	return true;
}

bool
YoloOrtModel::infer(std::string& err)
{
	std::vector<const char*> in = { myInNames[0].c_str() };
	std::vector<const char*> out;
	for (auto& n : myOutNames)
		out.push_back(n.c_str());
	return check(api().Run(mySession, nullptr, in.data(), &myValues[0], 1, out.data(), out.size(), &myValues[1]), err,
				 "ONNX Runtime could not run the network");
}

// Punto de entrada del plugin YOLO Seg Track: elige CUDA + TensorRT o ONNX Runtime al cargar.
#include "Plugin.h"

using namespace TD;

extern "C"
{
PLUGIN_EXPORT void
FillTOPPluginInfo(TOP_PluginInfo* info)
{
	if (!info->setAPIVersion(TOPCPlusPlusAPIVersion))
		return;
	info->executeMode = cudaBackend() ? TOP_ExecuteMode::CUDA : TOP_ExecuteMode::CPUMem;
	info->customOPInfo.opType->setString("Yolosegtrack");
	info->customOPInfo.opLabel->setString("YOLO Seg Track");
	info->customOPInfo.opIcon->setString("YST");
	info->customOPInfo.authorName->setString("Lolo Armdz");
	info->customOPInfo.authorEmail->setString("");
	if (info->customOPInfo.opHelpURL)
		info->customOPInfo.opHelpURL->setString("https://nuevosmiedos.xyz");
	info->customOPInfo.minInputs = 1;
	info->customOPInfo.maxInputs = 1;
}

PLUGIN_EXPORT TOP_CPlusPlusBase*
CreateTOPInstance(const OP_NodeInfo* info, TOP_Context* context)
{
#ifdef YOLOSEG_WITH_CUDA
	if (cudaBackend())
		return newYoloSegCuda(info, context);
#endif
	return newYoloSegCpu(info, context);
}

PLUGIN_EXPORT void
DestroyTOPInstance(TOP_CPlusPlusBase* instance, TOP_Context* context)
{
#ifdef YOLOSEG_WITH_CUDA
	if (cudaBackend())
	{
		deleteYoloSegCuda(instance);
		return;
	}
#endif
	deleteYoloSegCpu(instance);
}

// Datos del frame para otros plugins (ver SegShared.h / cpu/SegSharedCPU.h)
#ifdef YOLOSEG_WITH_CUDA
PLUGIN_EXPORT bool
YoloSegGetShared(uint32_t opId, void* out)
{
	return cudaBackend() && yoloSegSharedCuda(opId, out);
}
#endif

PLUGIN_EXPORT bool
YoloSegGetSharedCPU(uint32_t opId, void* out)
{
	return !cudaBackend() && yoloSegSharedCpu(opId, out);
}
}

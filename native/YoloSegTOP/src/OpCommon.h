// Parametros e Info DAT/CHOP de YOLO Seg Track, comunes a los dos backends
// (CUDA/TensorRT y ONNX Runtime). Mismos nombres de parametros => un .toe funciona con cualquiera.
// Los textos que ve el usuario (labels, errores, warnings) estan en ingles.
#pragma once

#include "CPlusPlus_Common.h"
#include "SegTypes.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

namespace opcommon
{
using namespace TD;

// ---------------------------------------------------------------- YOLO Seg Track

// Que calcula el operador
enum Task { TaskSegment = 0, TaskTrack = 1, TaskSegmentTrack = 2 };

const int kTableCols = 11;
inline const char* tableHeader(int c)
{
	static const char* h[] = { "id", "cx", "cy", "w", "h", "x1", "y1", "x2", "y2", "conf", "area" };
	return h[c];
}

inline const char** taskNames() { static const char* n[] = { "Segment", "Track", "Segmenttrack" }; return n; }

// Menus: getParInt devuelve el indice de la opcion elegida
inline int taskOf(const OP_Inputs* inputs) { return inputs->getParInt("Task"); }

// Titulo de seccion dentro de la pagina (todos los parametros van en una sola pagina)
inline void
header(OP_ParameterManager* manager, const char* name, const char* label)
{
	OP_StringParameter sp;
	sp.name = name;
	sp.label = label;
	manager->appendHeader(sp);
}

inline void
setupYoloParams(OP_ParameterManager* manager)
{
	static const char* taskLabels[] = { "Segmentation", "Tracking", "Segmentation + Tracking" };
	static const char* outputNames[] = { "Mask", "Colors", "Overlay" };
	static const char* outputLabels[] = { "White Mask", "Colors per ID", "Debug Overlay" };
	static const char* deviceNames[] = { "Auto", "Cpu" };
#ifdef __APPLE__
	static const char* deviceLabels[] = { "Auto (CoreML: GPU / Neural Engine)", "CPU Only" };
#else
	static const char* deviceLabels[] = { "Auto (NVIDIA: TensorRT, others: DirectML)", "CPU Only" };
#endif

	header(manager, "Headermodel", "Model");
	{
		OP_StringParameter sp;
		sp.name = "Modelfile";
		sp.label = "Model File (.onnx)";
		sp.defaultValue = "";
		manager->appendFile(sp);
	}
	{
		// solo en el modo ONNX Runtime; con CUDA queda en gris
		OP_StringParameter sp;
		sp.name = "Device";
		sp.label = "Device";
		sp.defaultValue = deviceNames[0];
		manager->appendMenu(sp, 2, deviceNames, deviceLabels);
	}
	{
		OP_NumericParameter np;
		np.name = "Fp16";
		np.label = "FP16";
		np.defaultValues[0] = 1;
		manager->appendToggle(np);
	}

	header(manager, "Headerdetection", "Detection");
	{
		OP_StringParameter sp;
		sp.name = "Task";
		sp.label = "Task";
		sp.defaultValue = taskNames()[TaskSegmentTrack];
		manager->appendMenu(sp, 3, taskNames(), taskLabels);
	}
	{
		OP_NumericParameter np;
		np.name = "Confidence";
		np.label = "Confidence";
		np.defaultValues[0] = 0.4;
		np.minSliders[0] = 0.0;
		np.maxSliders[0] = 1.0;
		np.minValues[0] = 0.0;
		np.maxValues[0] = 1.0;
		np.clampMins[0] = np.clampMaxes[0] = true;
		manager->appendFloat(np);
	}
	{
		OP_StringParameter sp;
		sp.name = "Outputmode";
		sp.label = "Output";
		sp.defaultValue = outputNames[0];
		manager->appendMenu(sp, 3, outputNames, outputLabels);
	}

	header(manager, "Headersegmentation", "Segmentation");
	{
		OP_NumericParameter np;
		np.name = "Softedges";
		np.label = "Soft Edges";
		np.defaultValues[0] = 1;
		manager->appendToggle(np);
	}

	header(manager, "Headertracking", "Tracking");
	{
		OP_NumericParameter np;
		np.name = "Trackbuffer";
		np.label = "Track Buffer (frames)";
		np.defaultValues[0] = 30;
		np.minSliders[0] = 1;
		np.maxSliders[0] = 300;
		np.minValues[0] = 1;
		np.clampMins[0] = true;
		manager->appendInt(np);
	}
	{
		OP_NumericParameter np;
		np.name = "Matchthresh";
		np.label = "Match Threshold";
		np.defaultValues[0] = 0.8;
		np.minSliders[0] = 0.0;
		np.maxSliders[0] = 1.0;
		np.minValues[0] = 0.0;
		np.maxValues[0] = 1.0;
		np.clampMins[0] = np.clampMaxes[0] = true;
		manager->appendFloat(np);
	}
	{
		OP_NumericParameter np;
		np.name = "Resettracker";
		np.label = "Reset Tracker";
		manager->appendPulse(np);
	}
}

inline PipelineSettings
readYoloSettings(const OP_Inputs* inputs)
{
	PipelineSettings s;
	const int task = taskOf(inputs);
	s.segment = task != TaskTrack;
	s.track = task != TaskSegment;
	s.confidence = (float)inputs->getParDouble("Confidence");
	s.mode = std::min(std::max(inputs->getParInt("Outputmode"), 0), 2);
	s.soft = inputs->getParInt("Softedges") != 0;
	s.trackBuffer = std::max(inputs->getParInt("Trackbuffer"), 1);
	s.matchThresh = (float)inputs->getParDouble("Matchthresh");
	return s;
}

// Gris en los parametros que no aplican a la tarea elegida. Se llama al principio de execute():
// leer los menus desde setParameterEnableStates no devolvia la opcion elegida (quedaba todo en gris).
inline void
yoloEnablePars(const OP_Inputs* inputs, bool cuda)
{
	const int task = taskOf(inputs);
	const bool segment = task != TaskTrack, track = task != TaskSegment;
	inputs->enablePar("Softedges", segment);
	inputs->enablePar("Device", !cuda);
#ifndef __APPLE__
	inputs->enablePar("Fp16", cuda);   // en Windows solo lo usa TensorRT (en Mac, CoreML)
#endif
	inputs->enablePar("Trackbuffer", track);
	inputs->enablePar("Matchthresh", track);
	inputs->enablePar("Resettracker", track);
}

inline const char* noModelError() { return "Choose a YOLO26 seg .onnx model in 'Model File' (e.g. models/yolo26n-seg.onnx)."; }

// Info CHOP: persons, infer_ms, total_ms, model_ready + p<i>_id/cx/cy/w/h
inline int32_t yoloInfoChans(const std::vector<PersonOut>& persons) { return 4 + 5 * (int32_t)persons.size(); }

inline void
yoloInfoChan(int32_t index, OP_InfoCHOPChan* chan, const std::vector<PersonOut>& persons, float inferMs, float totalMs, bool ready)
{
	static const char* fixed[] = { "persons", "infer_ms", "total_ms", "model_ready" };
	if (index < 4)
	{
		chan->name->setString(fixed[index]);
		float v[] = { (float)persons.size(), inferMs, totalMs, ready ? 1.f : 0.f };
		chan->value = v[index];
		return;
	}
	int k = (index - 4) / 5, f = (index - 4) % 5;
	static const char* names[] = { "id", "cx", "cy", "w", "h" };
	char buf[64];
	snprintf(buf, sizeof(buf), "p%d_%s", k, names[f]);
	chan->name->setString(buf);
	if (k >= (int)persons.size())
	{
		chan->value = 0.f;
		return;
	}
	const PersonOut& p = persons[k];
	float v[] = { (float)p.id, p.cx, p.cy, p.w, p.h };
	chan->value = v[f];
}

inline void
yoloInfoDATSize(OP_InfoDATSize* infoSize, const std::vector<PersonOut>& persons)
{
	infoSize->rows = 1 + (int32_t)persons.size();
	infoSize->cols = kTableCols;
	infoSize->byColumn = false;
}

inline void
yoloInfoDATEntries(int32_t index, OP_InfoDATEntries* entries, const std::vector<PersonOut>& persons)
{
	if (index == 0)
	{
		for (int c = 0; c < kTableCols; c++)
			entries->values[c]->setString(tableHeader(c));
		return;
	}
	if (index - 1 >= (int)persons.size())
		return;
	const PersonOut& p = persons[index - 1];
	float v[] = { p.cx, p.cy, p.w, p.h, p.x1, p.y1, p.x2, p.y2, p.conf, p.area };
	char buf[32];
	snprintf(buf, sizeof(buf), "%d", p.id);
	entries->values[0]->setString(buf);
	for (int c = 1; c < kTableCols; c++)
	{
		snprintf(buf, sizeof(buf), "%.4f", v[c - 1]);
		entries->values[c]->setString(buf);
	}
}

} // namespace opcommon

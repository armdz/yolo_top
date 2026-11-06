// ByteTrack - port en C++ del tracker de Ultralytics (ultralytics/trackers/byte_tracker.py).
// Mismos pasos y umbrales, para que los IDs se comporten igual que en el script de Python.
#pragma once

#include <array>
#include <memory>
#include <vector>

struct Detection
{
	float x1, y1, x2, y2;   // cualquier unidad (pixeles del modelo)
	float score;
	int   index;            // indice en la lista original de detecciones
};

struct TrackResult
{
	int   id;
	float x1, y1, x2, y2;   // caja suavizada por el filtro de Kalman
	float score;
	int   detIndex;         // deteccion que actualizo este track en este frame
};

struct ByteTrackParams
{
	float highThresh = 0.25f;     // track_high_thresh
	float lowThresh = 0.1f;       // track_low_thresh
	float newTrackThresh = 0.25f; // new_track_thresh
	int   trackBuffer = 30;       // frames que un track perdido sobrevive
	float matchThresh = 0.8f;     // match_thresh
	bool  fuseScore = true;
};

class ByteTracker
{
public:
	std::vector<TrackResult> update(const std::vector<Detection>& dets, const ByteTrackParams& p);
	void reset();

	// Sin tracking: las detecciones con score >= thresh, de mayor a menor confianza, con id = 1, 2, 3...
	// (los ids no se mantienen entre frames)
	static std::vector<TrackResult> untracked(const std::vector<Detection>& dets, float thresh);

	struct Track;
	using TrackPtr = std::shared_ptr<Track>;

private:
	std::vector<TrackPtr> myTracked;
	std::vector<TrackPtr> myLost;
	int myFrameId = 0;
	int myNextId = 0;
};
